"""Opt-in real inference check; may install dependencies/download several GB.

Run from the repo: python tests/smoke_inference.py --mode text
Vision: python tests/smoke_inference.py --mode vision --image /path/to/image.png
Offline unittest discovery intentionally excludes this script.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gemma_tipo_node import AUTO_MODEL, GemmaTipoNode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('text', 'vision'), default='text')
    parser.add_argument('--image', type=Path)
    parser.add_argument('--gpu-layers', type=int, default=0)
    parser.add_argument('--report', type=Path,
                        help='Optional JSON report (use models/ to keep local reports ignored)')
    args = parser.parse_args()
    if args.mode == 'vision' and (not args.image or not args.image.is_file()):
        parser.error('--mode vision requires an existing --image')

    image = None
    if args.mode == 'vision':
        import numpy as np
        from PIL import Image, ImageOps
        with Image.open(args.image) as source:
            source = ImageOps.exif_transpose(source).convert('RGB')
            image = np.asarray(source, dtype=np.float32)[None, ...] / 255.0

    prompt = '소녀 한 명이 하얀 셔츠와 파란 치마를 입고 공원에 서 있다.'
    started = time.perf_counter()
    output, = GemmaTipoNode().run(
        prompt, AUTO_MODEL, 'very_short', 'danbooru', 0.5,
        'blurry, low quality', 42, image=image, gpu_layers=args.gpu_layers)
    checks = {
        'nonempty': bool(output.strip()),
        'not_text_fallback': output != prompt,
        'no_korean_prose': re.search(r'[가-힣]', output) is None,
        'banned_tags_removed': not ({'blurry', 'low quality'} & {
            tag.strip().lower().replace('_', ' ') for tag in output.split(',')}),
    }
    versions = {}
    for name in ('llama-cpp-python', 'tipo-kgen', 'numpy', 'Pillow'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'mode': args.mode, 'gpu_layers_requested': args.gpu_layers,
        'image': str(args.image) if args.image else None,
        'seconds_including_first_use_setup': round(time.perf_counter() - started, 2),
        'platform': platform.platform(), 'python': platform.python_version(),
        'packages': versions,
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ('gemma_tipo_node.py', 'tipo_engine.py',
                                       'tipo_vision_engine.py')},
        'output': output, 'checks': checks, 'passed': all(checks.values()),
        'scope': 'One local inference; no quality benchmark or GPU-use assertion.',
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                               encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
