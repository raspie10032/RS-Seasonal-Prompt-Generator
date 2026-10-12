"""Two-GPU private import: Gemma encoding and PixAI fallback, then merge.

Run the metadata scan first. This runner starts only local subprocesses and
preserves checkpoints. It never downloads, publishes, or modifies images.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import signal
import sqlite3
import subprocess
import sys
import threading
import time

from .generated import write_report
from .store import Store


def merge_stores(parts, destination):
    """Reject incompatible contracts and duplicate keys before publishing."""
    if not parts:
        raise ValueError("No completed parts")
    sources = [(Store(p['store']), p['snapshot']) for p in parts]
    contract = sources[0][0].manifest(sources[0][1])['contract']
    for store, snapshot in sources:
        if store.manifest(snapshot)['contract'] != contract:
            raise ValueError("Incompatible image encoders")
        store.verify(snapshot)
    def entries():
        for store, snapshot in sources:
            yield from store.entries(snapshot)
    store = Store(destination)
    snapshot = store.build(entries(), contract, "personal-generated-metadata-and-pixai", bucket_count=1024,
        provenance={"parts": parts, "images_included": False, "rating_policy": "u: unknown",
                    "captions": "generation intent or explicit PixAI visual predictions"})
    result = dict(store.verify(snapshot), store=str(Path(destination).resolve()))
    store.activate(snapshot, store.current())
    return result


def run(args):
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(args.catalog.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        if not source.execute("SELECT value FROM config WHERE key='scan_complete' AND value='true'").fetchone():
            raise ValueError("Finish the metadata scan first")
    finally:
        source.close()
    allowed = sorted(os.sched_getaffinity(0))[:args.cpu_threads]
    if len(allowed) < 2:
        raise ValueError("Two workers need at least two CPU cores")
    cpus = [allowed[:len(allowed)//2], allowed[len(allowed)//2:]]
    work = queue.Queue()
    for shard in range(args.shards):
        work.put(('metadata', args.catalog, shard, args.shards))
    producer_done, stopping = threading.Event(), threading.Event()
    active, parts, errors = {}, [], []
    lock = threading.Lock()

    def stop(*_):
        stopping.set()
        with lock:
            for process in active.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}

    def launch(name, command, worker, encoder=False):
        env = os.environ.copy()
        env['HF_HUB_OFFLINE'] = '1'
        if encoder and args.encoder_pythonpath:
            env['PYTHONPATH'] = str(args.encoder_pythonpath) + os.pathsep + str(Path.cwd())
        command = ['taskset','-c',','.join(map(str,cpus[worker])),sys.executable,'-u',*command]
        with (root / (name + '.log')).open('a', encoding='utf-8') as stream:
            with lock:
                if stopping.is_set():
                    raise RuntimeError("Import interrupted; checkpoints preserved")
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                           env=env, start_new_session=True)
                active[name] = process
            code = process.wait()
        with lock:
            active.pop(name, None)
        if code:
            raise RuntimeError(name + ' exited ' + str(code) + '; see its log')

    def worker(index):
        try:
            if index == 1:
                tagged = root / 'pixai.sqlite3'
                launch('pixai', ['-m','vrs_int.generated_tagger','--catalog',str(args.catalog),
                    '--output',str(tagged),'--model',str(args.pixai),'--gpu',args.gpu[index],
                    '--tagger','pixai','--workers',str(len(cpus[index])), '--batch-size','8'], index)
                for shard in range(2):
                    work.put(('pixai',tagged,shard,2))
                producer_done.set()
            while not stopping.is_set():
                try:
                    family, catalog, shard, count = work.get(timeout=.5)
                except queue.Empty:
                    if producer_done.is_set(): break
                    continue
                name = family + '-' + str(shard).zfill(2)
                checkpoint = root / (name + '.sqlite3')
                launch(name, ['-m','vrs_int.generated','build','--catalog',str(catalog),
                    '--encoder',str(args.encoder),'--store',str(root / (name + '-store')),
                    '--checkpoint',str(checkpoint),'--gpu',args.gpu[index],'--gpu-fraction','.90',
                    '--cpu-threads',str(len(cpus[index])),'--preprocess-workers',str(len(cpus[index])),
                    '--batch-size','64','--delay','0','--shard-index',str(shard),'--shard-count',str(count)],
                    index, encoder=True)
                report = json.loads(checkpoint.with_suffix('.build.json').read_text())
                if not report['state'].startswith('complete'):
                    raise ValueError('Incomplete shard ' + name)
                with lock:
                    parts.append(dict(name=name, store=report['store'], snapshot=report['snapshot'],
                                      errors=report['errors']))
                work.task_done()
        except Exception as exc:
            with lock:
                errors.append(str(exc))
            stop()

    started = time.time()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker,i) for i in range(2)]
            while not all(f.done() for f in futures):
                with lock:
                    state = dict(state='running', elapsed_seconds=round(time.time()-started),
                        active={name:p.pid for name,p in active.items()}, completed_parts=list(parts),
                        pending=work.qsize(), errors=list(errors), images_modified=False,
                        cpu_cores=cpus, gpu=args.gpu)
                write_report(root / 'progress.json', state)
                try:
                    gpu = subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
                    with (root/'resources.jsonl').open('a') as log:
                        log.write(json.dumps(dict(time=time.time(),gpus=gpu.stdout.strip().splitlines()))+'\n')
                except (subprocess.SubprocessError, OSError):
                    pass
                time.sleep(5)
            for future in futures: future.result()
        if errors or stopping.is_set():
            raise RuntimeError('; '.join(errors) or 'Import interrupted')
        write_report(root/'progress.json',dict(state='merging',completed_parts=parts))
        result = merge_stores(sorted(parts,key=lambda p:p['name']), root/'memory')
        tagging = json.loads((root/'pixai.tagger.json').read_text())
        has_errors = any(p['errors'] for p in parts) or tagging['files'].get('error',0) > 0
        result.update(state='complete-with-errors' if has_errors else 'complete',
                      elapsed_seconds=round(time.time()-started), parts=parts, images_modified=False)
        result['tagging'] = tagging
        write_report(root/'report.json',result)
        write_report(root/'progress.json',result)
        print(json.dumps(result),flush=True)
        return result
    except Exception as exc:
        write_report(root/'progress.json',dict(state='failed',error=str(exc),completed_parts=parts,
                                               checkpoints_preserved=True,images_modified=False))
        raise
    finally:
        stop()
        for sig, handler in previous.items(): signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('catalog','encoder','pixai','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gpu',action='append',required=True)
    parser.add_argument('--cpu-threads',type=int,default=14)
    parser.add_argument('--encoder-pythonpath',type=Path)
    parser.add_argument('--shards',type=int,choices=(2,4,8,16),default=16)
    args=parser.parse_args()
    if len(args.gpu)!=2 or len(set(args.gpu))!=2 or any(not g.startswith('GPU-') for g in args.gpu):
        parser.error('Specify two different GPU UUIDs')
    if args.cpu_threads<2:
        parser.error('At least two CPU cores required')
    run(args)


if __name__=='__main__':
    main()
