"""Compare local GGUFs on fixed text/image cases with one loopback server at a time.

No download or inference service is used. Cases/results must be outside the repo.
Device totals and this server PID's GPU memory are recorded separately.
"""
import argparse
import base64
import json
import mimetypes
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

from .gemma4_gguf_audit import sha256
from .gemma4_tuning_data import score_output


def request(url, body=None, timeout=10):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def sample_device(uuid, pid=None):
    result = subprocess.run(['nvidia-smi', '-i', uuid, '-q', '-x'],
        capture_output=True, text=True, timeout=5, check=True)
    gpu = ET.fromstring(result.stdout).find('gpu')
    sample = {'time': time.time(), 'uuid': gpu.findtext('uuid'), 'name': gpu.findtext('product_name'),
              'device_memory_mib': int(gpu.findtext('fb_memory_usage/used').split()[0]),
              'utilization_percent': int(gpu.findtext('utilization/gpu_util').split()[0])}
    for process in gpu.findall('processes/process_info'):
        if process.findtext('pid') == str(pid):
            sample['process_gpu_memory_mib'] = int(process.findtext('used_memory').split()[0])
            sample['process_gpu_type'] = process.findtext('type')
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True, type=Path)
    parser.add_argument('--models', required=True, nargs='+', type=Path)
    parser.add_argument('--mmproj', required=True, type=Path)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', required=True, help='llama-server device name, verified using --list-devices')
    parser.add_argument('--gpu-uuid', required=True, help='UUID for device and server PID memory sampling')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    cases = json.loads(args.cases.read_text())
    frozen = {'cases': cases, 'cases_sha256': sha256(args.cases), 'seed': 42, 'temperature': 0,
              'ctx': 4096, 'max_tokens': 320, 'batch': 1024, 'ubatch': 1024,
              'device': args.device, 'gpu_uuid': args.gpu_uuid,
              'models': {str(p): sha256(p) for p in args.models}, 'mmproj_sha256': sha256(args.mmproj),
              'server_sha256': sha256(args.server), 'script_sha256': sha256(__file__),
              'images': {c['image']: sha256(c['image']) for c in cases if c.get('image')},
              'scope': 'Small deterministic output/constraint comparison, not semantic quality equivalence'}
    (args.output / 'predeclared.json').write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    reports = []
    for model in args.models:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        command = [str(args.server.resolve()), '-m', str(model.resolve()), '--mmproj', str(args.mmproj.resolve()),
                   '--device', args.device, '-ngl', '99', '--ctx-size', '4096', '-t', '2', '-tb', '2',
                   '-b', '1024', '-ub', '1024', '--parallel', '1', '--host', '127.0.0.1', '--port', str(port),
                   '--jinja', '--reasoning', 'off', '--no-warmup', '--fit', 'off', '--threads-http', '2']
        report = {'model': str(model), 'command': command, 'baseline_device_sample': sample_device(args.gpu_uuid),
                  'rows': [], 'samples': [], 'monitor_errors': []}
        reports.append(report)
        stop = threading.Event()
        def monitor():
            while not stop.is_set():
                try:
                    sample = sample_device(args.gpu_uuid, process.pid)
                    status = Path(f'/proc/{process.pid}/status').read_text()
                    for line in status.splitlines():
                        if line.startswith('VmHWM:'):
                            sample['process_rss_hwm_kib'] = int(line.split()[1])
                    report['samples'].append(sample)
                except (OSError, ValueError, ET.ParseError, subprocess.SubprocessError) as exc:
                    report['monitor_errors'].append(type(exc).__name__)
                stop.wait(0.5)
        started = time.monotonic()
        with (args.output / f'{model.stem}.log').open('w') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            thread = threading.Thread(target=monitor)
            thread.start()
            try:
                base_url = f'http://127.0.0.1:{port}'
                deadline = time.monotonic() + 180
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'Server exited ({process.returncode}); see {log.name}')
                    try:
                        if request(base_url + '/health')['status'] == 'ok':
                            break
                    except (OSError, urllib.error.URLError):
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError('Server did not become ready')
                    time.sleep(0.5)
                report['load_seconds'] = time.monotonic() - started
                for case in cases:
                    messages = json.loads(json.dumps(case['messages']))
                    if case.get('image'):
                        image = Path(case['image'])
                        mime = mimetypes.guess_type(image)[0] or 'image/png'
                        data = base64.b64encode(image.read_bytes()).decode()
                        messages[-1]['content'] = [{'type': 'text', 'text': messages[-1]['content']},
                            {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{data}'}}]
                    begin = time.monotonic()
                    response = request(base_url + '/v1/chat/completions', {
                        'messages': messages, 'temperature': 0, 'seed': 42, 'max_tokens': 320,
                        'cache_prompt': False, 'chat_template_kwargs': {'enable_thinking': False}}, timeout=180)
                    choice = response['choices'][0]
                    raw = choice['message'].get('content') or ''
                    if not raw.strip():
                        raise RuntimeError(f'Empty output for {case["name"]}')
                    row = {'name': case['name'], 'image': case.get('image'), 'raw': raw,
                           'seconds': time.monotonic() - begin, 'finish_reason': choice['finish_reason'],
                           'usage': response.get('usage'), 'timings': response.get('timings'),
                           'score': score_output(raw, case.get('required', []), case.get('excluded', []))}
                    report['rows'].append(row)
                    (args.output / 'report.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2))
                    print(json.dumps({'model': model.name, 'case': case['name'], 'seconds': row['seconds'],
                                      'pass': row['score']['pass']}, ensure_ascii=False), flush=True)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                stop.set()
                thread.join(timeout=10)
                report['peak_device_memory_mib'] = max((s['device_memory_mib'] for s in report['samples']), default=None)
                report['peak_process_gpu_memory_mib'] = max((s['process_gpu_memory_mib'] for s in report['samples']
                                                             if 'process_gpu_memory_mib' in s), default=None)
                report['peak_process_rss_mib'] = max((s.get('process_rss_hwm_kib', 0) for s in report['samples']), default=0) / 1024
                report['after_device_sample'] = sample_device(args.gpu_uuid)
                (args.output / 'report.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
