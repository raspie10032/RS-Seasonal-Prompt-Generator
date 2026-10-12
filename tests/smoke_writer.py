"""Opt-in real writer inference; downloads ~2.5 GB on first use.

Checks a small fixed set of explicit scene details, not general prompt quality.
Run: python tests/smoke_writer.py --report models/writer-smoke.json
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prompt_writer_node import RSPromptWriter, MODEL_REPO, MODEL_REVISION, MODEL_SHA256

# Inputs fixed before the first writer inference. Regex checks support common English
# wording; read the saved outputs too, because substring checks cannot judge
# binding, overall visual coherence, or the truth of a negated phrase.
CASES = [
    ('park', '소녀 한 명이 하얀 셔츠와 파란 치마를 입고 공원에 서 있다.',
     ['white shirt', 'blue skirt', 'park', r'stand(?:ing|s)']),
    ('neon', '비 오는 밤, 검은 코트를 입은 여성이 네온사인이 빛나는 골목을 걷고 있다.',
     ['black coat', 'night', 'rain', 'neon', 'alley', r'walk(?:ing|s)']),
    ('dress', '해질녘 해변에서 빨간 원피스를 입은 성인 여성이 걷고 있다.',
     ['red dress', r'beach|shore', r'sunset|dusk|setting sun', r'walk(?:ing|s)']),
    ('pair', '성인 여성 두 명이 카페에서 마주 앉아 있다. 한 명은 하얀 스웨터, 다른 한 명은 검은 재킷을 입었다.',
     [r'2girls|2women|two (?:adult )?women|two female', r'caf[eé]', 'white sweater', 'black jacket', r'\bsit(?:ting|s)?\b|seated', 'facing each other|opposite each other|across from each other']),
    ('still_life', '사람이 없는 조용한 카페. 나무 테이블 위에 커피잔 하나, 창밖에는 비가 내린다.',
     [r'caf[eé]', r'wood(?:en)? table', 'coffee', 'rain', 'window', r'no people|empty|unoccupied|no human|without people']),
    ('tag_input', '1girl, purple dress, autumn forest, standing, full body, golden hour',
     ['purple dress', 'autumn', 'forest', r'stand(?:ing|s)', r'full.body|head.to.toe|fully visible', 'golden hour']),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--gpu-layers', type=int, default=-1,
                        help='Default: all GPU layers; 0 explicitly allows CPU inference.')
    parser.add_argument('--cpu-threads', type=int, default=2)
    args = parser.parse_args()
    rows = []
    for name, prompt, patterns in CASES:
        for output_format in ('tags', 'natural_language'):
            started = time.perf_counter()
            output, = RSPromptWriter().run(
                prompt, output_format=output_format, creativity='balanced',
                length='medium', seed=42, gpu_layers=args.gpu_layers, cpu_threads=args.cpu_threads)
            normalized = output.lower().replace('_', ' ')
            checks = {pattern: bool(re.search(pattern, normalized)) for pattern in patterns}
            checks['nonempty'] = bool(output.strip())
            checks['no_korean'] = re.search('[가-힣]', output) is None
            row = dict(case=name, output_format=output_format, input=prompt,
                       output=output, checks=checks, passed=all(checks.values()),
                       seconds=round(time.perf_counter() - started, 2))
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    versions = {name: importlib.metadata.version(name) for name in ('llama-cpp-python',)}
    import llama_cpp
    try:
        gpu_rows = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
            '--format=csv,noheader'], text=True, timeout=5).splitlines()
        gpu_rows = [row for row in gpu_rows if row.split(',')[0].strip() == str(os.getpid())]
    except (OSError, subprocess.SubprocessError):
        gpu_rows = []
    report = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  model=dict(repo=MODEL_REPO, revision=MODEL_REVISION, sha256=MODEL_SHA256),
                  python=platform.python_version(), platform=platform.platform(), packages=versions,
                  gpu_layers_requested=args.gpu_layers,
                  runtime=dict(cpu_threads=args.cpu_threads,
                               cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                               system_info=llama_cpp.llama_print_system_info().decode(),
                               nvidia_process_rows=gpu_rows),
                  settings=dict(temperature=0.6, seed=42, creativity='balanced', length='medium'),
                  source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                 for name in ('prompt_writer_node.py', 'tipo_engine.py', 'tests/smoke_writer.py')},
                  results=rows, passed=all(row['passed'] for row in rows),
                  scope='Six fixed inputs in two formats; keyword regressions plus human review, not a general quality or GPU benchmark.')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
