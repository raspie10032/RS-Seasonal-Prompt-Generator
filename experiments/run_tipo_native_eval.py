"""Record frozen development inputs through an existing local ComfyUI server.

No model install, server restart, training, or final-set inference. Submits one
request at a time only while the queue is empty. A timeout leaves its prompt ID
on disk for inspection; it never retries an outcome-unknown generation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request
import uuid

from .tipo_evaluation import read_rows, score_results, validate


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def local_url(value):
    parsed = urllib.parse.urlparse(value)
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ('', '/')):
        raise ValueError('Use a local HTTP ComfyUI origin, without credentials or a path')
    return value.rstrip('/') + '/'


def api(base, route, payload=None):
    request = urllib.request.Request(base + route,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    # Never follow a redirect carrying private prompts to a different server.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    with urllib.request.build_opener(NoRedirect).open(request, timeout=20) as response:
        return json.load(response)


def idle(base):
    queue = api(base, 'queue')
    if queue['queue_running'] or queue['queue_pending']:
        raise RuntimeError('ComfyUI queue is occupied; stopping before the next case')


def graph_for(row, model, settings, image=None):
    inputs = row['input']
    graph = {
        '1': {'class_type': 'CLIPLoader', 'inputs': {
            'clip_name': model, 'type': 'stable_diffusion', 'device': 'default'}},
        '2': {'class_type': 'GemmaTipoNative', 'inputs': {
            'clip': ['1', 0], 'prompt': inputs['prompt'],
            'tag_length': inputs.get('tag_length', 'short'),
            'ban_tags': inputs.get('ban_tags', ''), **settings}},
        '3': {'class_type': 'PreviewAny', 'inputs': {'source': ['2', 0]}},
        '4': {'class_type': 'PreviewAny', 'inputs': {'source': ['2', 1]}},
    }
    if bool(inputs.get('image')) != bool(image):
        raise ValueError('Image input must be staged; never downgrade to text')
    if image:
        graph['5'] = {'class_type': 'LoadImage', 'inputs': {'image': image}}
        graph['2']['inputs']['image'] = ['5', 0]
    return graph


def extract_output(history):
    if history['status'].get('status_str') != 'success' or not history['status'].get('completed'):
        raise RuntimeError('Native generation failed; inspect saved history')
    outputs = history['outputs']
    result = {}
    for node, name in (('3', 'postprocessed'), ('4', 'raw')):
        parts = outputs[node]['text']
        if len(parts) != 1 or not isinstance(parts[0], str) or not parts[0].strip():
            raise ValueError('Missing or ambiguous native output')
        result[name] = parts[0]
    # PreviewAny does not expose generated token IDs or the actual stop reason.
    result.update(truncated=None, termination_status='not exposed by native API')
    return result


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--comfy-input', type=Path, required=True)
    parser.add_argument('--model-file', type=Path, required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8188')
    parser.add_argument('--max-tokens', type=int, default=384)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    base = local_url(args.url)
    if not 16 <= args.max_tokens <= 2048 or not 0 <= args.temperature <= 1.5 or args.timeout <= 0:
        parser.error('Invalid generation settings')
    rows = read_rows(args.cases)
    validate(rows)
    if any(row['split'] != 'dev' for row in rows):
        parser.error('This baseline runner accepts development inputs only')
    manifest = json.loads((args.cases.parent / 'manifest.json').read_text())
    if file_hash(args.cases) != manifest['files'][args.cases.name]:
        raise ValueError('Frozen evaluation checksum differs')
    idle(base)
    info = api(base, 'object_info/GemmaTipoNative')
    if 'GemmaTipoNative' not in info:
        raise RuntimeError('Native node is not loaded')
    if not args.comfy_input.is_dir() or not args.model_file.is_file():
        raise ValueError('Missing local input directory or model')
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {'max_tokens': args.max_tokens, 'temperature': args.temperature, 'seed': args.seed}
    run_id = 'tipo-eval-' + uuid.uuid4().hex
    start = time.monotonic()
    write_json(args.output/'run.json', {
        'schema': 'tipo-native-evaluation/1', 'client_id': run_id, 'settings': settings,
        'cases_sha256': file_hash(args.cases), 'model_file': str(args.model_file.resolve()),
        'model_file_sha256': file_hash(args.model_file), 'model_name': args.model_name,
        'model_identity_limit': 'hash of caller-specified file; confirm server loader maps to this file',
        'node_schema': info, 'system_stats_before': api(base, 'system_stats'),
        'termination_status': 'API does not expose token IDs or stop reason',
        'scope': 'Development baseline only. Condition checks are not creative quality.'})
    staged_dir = args.comfy_input / run_id
    staged_dir.mkdir(exist_ok=False)
    outputs = []
    try:
        with (args.output/'outputs.jsonl').open('x', encoding='utf-8') as stream:
            for index, row in enumerate(rows):
                staged = None
                source = row['input'].get('image')
                if source:
                    from PIL import Image, ImageOps
                    if file_hash(source) != row['source']['image_sha256']:
                        raise ValueError('Reference image checksum differs')
                    dst = staged_dir / (str(index) + '.png')
                    with Image.open(source) as im:
                        pixels = ImageOps.exif_transpose(im).convert('RGB')
                        pixels.info.clear()
                        pixels.save(dst)
                    staged = run_id + '/' + dst.name
                idle(base)
                graph = graph_for(row, args.model_name, settings, staged)
                tick = time.monotonic()
                submitted = api(base, 'prompt', {'prompt': graph, 'client_id': run_id})
                write_json(args.output/f'{index:03}-submitted.json', {'id': row['id'], **submitted})
                prompt_id = submitted['prompt_id']
                while True:
                    result = api(base, 'history/' + urllib.parse.quote(prompt_id, safe=''))
                    if prompt_id in result:
                        history = result[prompt_id]
                        break
                    if time.monotonic() - tick > args.timeout:
                        raise TimeoutError('Outcome unknown; inspect saved prompt ID before resuming')
                    time.sleep(0.25)
                write_json(args.output/f'{index:03}-history.json', history)
                output = {'id': row['id'], 'task': row['task'], 'prompt_id': prompt_id,
                          'wall_seconds': time.monotonic() - tick, **extract_output(history)}
                stream.write(json.dumps(output, ensure_ascii=False) + '\n')
                stream.flush()
                outputs.append(output)
                print(json.dumps({'completed': len(outputs), 'total': len(rows),
                                  'id': row['id'], 'seconds': output['wall_seconds']}), flush=True)
    finally:
        report = score_results(rows, outputs)
        report.update(wall_seconds=time.monotonic()-start,
                      termination_status_unknown=len(outputs),
                      staged_image_directory=str(staged_dir),
                      postprocessed_constraint_passes=sum(
                          r['postprocessed']['constraint_pass'] for r in report['results']))
        write_json(args.output/'score.json', report)


if __name__ == '__main__':
    main()
