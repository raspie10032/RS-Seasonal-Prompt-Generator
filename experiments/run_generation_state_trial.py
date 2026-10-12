"""GPU-only paired TIPO experiment from private saved VRS recommendations.

python -m experiments.run_generation_state_trial --help
No downloads, store mutation, source images, or output constraint repair.
"""
import argparse
from collections import defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import resource
import time

from vrs_int.generation_state import build_state, messages_for, reference_tags
from vrs_int.tag_policy import canonical, filter_recommendation_tags


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    # Only call inside a newly created experiment directory.
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def score_tags(text, state):
    """Literal tag adherence, not semantic/image quality. Never repairs output."""
    from tipo_engine import _extract_tags
    tags = {canonical(t) for t in _extract_tags(text).split(",") if t.strip()}
    controls = state["controls"]
    missing = {kind: [t for t in controls[kind] if t not in tags] for kind in ("keep", "target")}
    excluded = sorted(tags.intersection(controls["exclude"]))
    identities = {row["tag"]: [kind] for kind, rows in state["identity_candidates"].items() for row in rows}
    _, quality = filter_recommendation_tags(sorted(tags), identities)
    used = sorted(tags.intersection(reference_tags(state)) - set(controls["keep"] + controls["target"]))
    mixed = {}
    for kind, rows in state["identity_candidates"].items():
        present = sorted(tags.intersection(row["tag"] for row in rows))
        if len(present) > 1:
            mixed[kind] = present
    return {"metric": "literal tag checks only; no semantic quality verdict",
            "tag_count": len(tags), "missing": missing, "excluded_present": excluded,
            "quality_year_present": sorted(quality), "extra_reference_tags": used,
            "mixed_candidate_names": mixed,
            "required_total": sum(len(controls[k]) for k in ("keep", "target")),
            "required_found": sum(len(controls[k]) - len(missing[k]) for k in ("keep", "target")),
            "literal_pass": not any(missing.values()) and not excluded and not quality and not mixed}


def render_report(out, report):
    esc = lambda value: html.escape(str(value))
    parts = ['<!doctype html><html lang="ko"><meta charset="utf-8">',
             '<title>VRS-INT → TIPO paired state trial</title>',
             '<style>body{font:16px system-ui;max-width:1450px;margin:35px auto;padding:0 20px;background:#101723;color:#e3ecf7}table{width:100%;table-layout:fixed;border-collapse:collapse}td,th{border:1px solid #41516a;padding:14px;vertical-align:top;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.65 system-ui}.bad{color:#ffb5ac}summary{cursor:pointer}small{color:#bacbe0}</style>',
             '<h1>VRS-INT → TIPO: request / flat tags / structured state</h1>',
             '<p>Same model, seed, controls and reference-tag set. No source image is passed to TIPO. Raw outputs are shown without constraint repair. Literal checks are not prompt-quality scores.</p>',
             '<pre>' + esc(json.dumps(report['summary'], ensure_ascii=False, indent=2)) + '</pre>']
    grouped = defaultdict(dict)
    for row in report['rows']:
        grouped[(row['case'], row['seed'])][row['mode']] = row
    for (case, seed), rows in grouped.items():
        parts.append(f'<h2>{esc(case)} / seed {seed}</h2><p>{esc(next(iter(rows.values()))["request"])}</p><table><tr>')
        parts += [f'<th>{mode}</th>' for mode in ('request', 'flat', 'state')]
        parts.append('</tr><tr>')
        for mode in ('request', 'flat', 'state'):
            row = rows.get(mode, {})
            parts.append('<td><pre>' + esc(row.get('raw', row.get('error', 'pending'))) + '</pre><details><summary>Timing &amp; literal checks</summary><pre>' + esc(json.dumps({k: row.get(k) for k in ('seconds', 'usage', 'checks', 'finish_reason')}, ensure_ascii=False, indent=2)) + '</pre></details></td>')
        parts.append('</tr></table>')
    (out / 'report.html').write_text(''.join(parts) + '</html>', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True, help='Private JSON list: name, recommendation, request, keep, target, exclude')
    parser.add_argument('--model', type=Path, required=True, help='Existing local GGUF, never downloaded')
    parser.add_argument('--output', type=Path, required=True, help='Must not already exist')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 91])
    parser.add_argument('--framing', choices=('creative', 'tipo-partial'), default='creative')
    parser.add_argument('--tag-dictionary', type=Path, help='Optional local tag dictionary CSV')
    parser.add_argument('--resolve', action='store_true', help='Select compatible single-subject references')
    args = parser.parse_args()
    if not args.model.is_file() or 'mmproj' in args.model.name.lower():
        parser.error('Supply an existing language-model GGUF')
    if not os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
        parser.error('Select one GPU UUID with CUDA_VISIBLE_DEVICES before launch')
    import llama_cpp
    import tipo_engine as te
    from experiments.run_vrs_trial import Monitor
    te.require_gpu_offload()
    if llama_cpp.llama_max_devices() < 1:
        raise RuntimeError('No GPU device available')
    cases = json.loads(args.cases.read_text(encoding='utf-8'))
    from vrs_int.tag_dictionary import TagDictionary
    dictionary = TagDictionary(args.tag_dictionary) if args.tag_dictionary else None
    states = []
    for case in cases:
        recommendation_path = Path(case['recommendation'])
        recommendation = json.loads(recommendation_path.read_text(encoding='utf-8'))
        state = build_state(recommendation, case['request'], dictionary=dictionary, resolve=args.resolve, **{
            k: case.get(k, []) for k in ('keep', 'target', 'exclude')})
        states.append((case, state, sha256(recommendation_path)))
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'Text state experiment only; no image re-encoding, memory rebuild or hidden-state injection',
              'model': str(args.model.resolve()), 'model_sha256': sha256(args.model),
              'llama_cpp_version': llama_cpp.__version__, 'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'],
              'settings': {'n_ctx': 4096, 'gpu_layers': -1, 'cpu_threads': 2,
                           'max_tokens': 512, 'temperature': .6, 'top_p': .9, 'top_k': 40,
                           'repeat_penalty': 1.05, 'seeds': args.seeds, 'framing': args.framing,
                           'resolve': args.resolve},
              'tag_dictionary': dictionary.provenance if dictionary else None,
              'source_sha256': {str(p): sha256(p) for p in (Path(__file__), Path('vrs_int/generation_state.py'),
                  Path('vrs_int/tag_dictionary.py'), Path('vrs_int/text_knowledge.py'),
                  Path('vrs_int/state_selection.py'), Path('tipo_engine.py'))},
              'cases_sha256': sha256(args.cases), 'rows': [], 'summary': {}}
    save(args.output / 'predeclared.json', report)
    for index, (case, state, source_hash) in enumerate(states):
        save(args.output / f'state-{index}.json', {'case': case['name'], 'source_file_sha256': source_hash, 'state': state})
    started = time.perf_counter()
    with Monitor() as monitor:
        load_start = time.perf_counter()
        llm = te.load_tipo(str(args.model), -1, n_ctx=4096, n_threads=2, cache=False)
        if llm is None:
            raise RuntimeError('Could not load TIPO')
        report['load_seconds'] = time.perf_counter() - load_start
        try:
            for case_index, (case, state, _) in enumerate(states):
                for seed_index, seed in enumerate(args.seeds):
                    # Rotate order to avoid always favoring the last/cached arm.
                    modes = ['request', 'flat', 'state']
                    offset = (case_index + seed_index) % 3
                    for mode in modes[offset:] + modes[:offset]:
                        llm.reset()
                        messages = messages_for(state, mode, framing=args.framing)
                        row = {'case': case['name'], 'request': case['request'], 'seed': seed, 'mode': mode, 'messages': messages}
                        tick = time.perf_counter()
                        try:
                            response = llm.create_chat_completion(messages=messages, max_tokens=512,
                                temperature=.6, top_p=.9, top_k=40, repeat_penalty=1.05, seed=seed)
                            choice = response['choices'][0]
                            row.update(raw=choice['message']['content'] or '',
                                       finish_reason=choice.get('finish_reason'), usage=response.get('usage'))
                            row['checks'] = score_tags(row['raw'], state)
                            row['completed'] = row['finish_reason'] == 'stop' and bool(row['raw'].strip())
                        except Exception as exc:
                            row.update(error=f'{type(exc).__name__}: {exc}', completed=False)
                        row['seconds'] = round(time.perf_counter() - tick, 3)
                        report['rows'].append(row)
                        save(args.output / 'progress.json', report)
                        print(json.dumps({k: row.get(k) for k in ('case', 'mode', 'seed', 'seconds', 'completed')}, ensure_ascii=False), flush=True)
        finally:
            llm.close()
    report.update(total_seconds=time.perf_counter() - started,
                  peak_process_vram_mib=max((s['mib'] for s in monitor.samples), default=None),
                  max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save(args.output / 'gpu-samples.json', monitor.samples)
    for mode in ('request', 'flat', 'state'):
        rows = [r for r in report['rows'] if r['mode'] == mode]
        checks = [r['checks'] for r in rows if 'checks' in r]
        report['summary'][mode] = {
            'runs': len(rows), 'completed': sum(r['completed'] for r in rows),
            'literal_pass': sum(r['completed'] and r.get('checks', {}).get('literal_pass', False) for r in rows),
            'required_found': sum(c['required_found'] for c in checks),
            'required_total_scored': sum(c['required_total'] for c in checks),
            'mean_seconds': round(sum(r['seconds'] for r in rows) / len(rows), 3),
            'mean_extra_reference_tags': round(sum(len(c['extra_reference_tags']) for c in checks) / max(1, len(checks)), 2)}
    save(args.output / 'report.json', report)
    render_report(args.output, report)
    print(json.dumps({'summary': report['summary'], 'peak_process_vram_mib': report['peak_process_vram_mib']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
