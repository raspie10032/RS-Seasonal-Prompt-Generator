"""Compare a completed generated-image store with both private source catalogs."""
import argparse
from collections import Counter
from contextlib import ExitStack, closing
import json
from pathlib import Path
import sqlite3
import time

from .generated import expand, write_report
from .store import Store
from .tag_policy import canonical


def audit_run(catalog, run_directory):
    root = Path(run_directory)
    report = json.loads((root / 'report.json').read_text())
    if report['state'] not in ('complete', 'complete-with-errors'):
        raise ValueError('Import has not finished')
    memory = Store(report['store'])
    verification = memory.verify(report['snapshot'])
    counts, names, seen = Counter(), Counter(), set()
    with ExitStack() as stack:
        sources = [stack.enter_context(closing(sqlite3.connect(
            Path(path).resolve().as_uri() + '?mode=ro', uri=True)))
            for path in (catalog, root / 'pixai.sqlite3')]
        expected_count = sum(db.execute('SELECT count(*) FROM records').fetchone()[0] for db in sources)
        for entry in memory.entries(report['snapshot']):
            sha = entry['document']['image_sha256']
            if sha in seen:
                raise ValueError('Duplicate image in merged memory')
            seen.add(sha)
            matches = [(index, row[0]) for index, db in enumerate(sources)
                       if (row := db.execute('SELECT record FROM records WHERE sha=?', (sha,)).fetchone())]
            if len(matches) != 1:
                raise ValueError('Missing or ambiguous source observation')
            source_index, payload = matches[0]
            original = expand(payload)
            if len(entry['audit']) != 1 or entry['audit'][0]['source_record'] != original:
                raise ValueError('Original caption or metadata differs from the catalog')
            if entry['document']['rating'] != 'u' or not original.get('caption'):
                raise ValueError('Unexpected rating or empty caption')
            counts['metadata' if source_index == 0 else 'pixai'] += 1
            counts['vectors'] += entry['vector'] is not None
            for field, kind in (('tags_artist', 'artist'), ('tags_character', 'character'),
                                ('tags_copyright', 'copyright')):
                for name in original.get(field, []):
                    name = canonical(name)
                    if name not in entry['document']['tags'] or kind not in entry['categories'].get(name, []):
                        raise ValueError('Core name or its category was lost')
                    names[kind] += 1
        if len(seen) != expected_count:
            raise ValueError('Catalog observations are missing from memory')
    return dict(state=report['state'], snapshot=report['snapshot'], verification=verification,
                source_records_compared=len(seen), sources=dict(counts),
                preserved_name_occurrences=dict(names),
                original_caption_and_metadata_equal=True,
                scope='Storage/source preservation audit; not visual accuracy evaluation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-pid', type=int, help='Linux importer PID; watch its current process instance')
    args = parser.parse_args()
    started = time.monotonic()
    try:
        if args.wait_pid:
            status = Path('/proc') / str(args.wait_pid) / 'stat'
            identity = status.read_text().rsplit(')', 1)[1].split()[19] if status.exists() else None
            while not (args.run / 'report.json').exists():
                if time.monotonic() - started > 86400:
                    raise RuntimeError('Import wait exceeded one day')
                if (not status.exists() or
                        status.read_text().rsplit(')', 1)[1].split()[19] != identity):
                    raise RuntimeError('Importer exited without a completion report')
                progress = json.loads((args.run / 'progress.json').read_text())
                if progress.get('state') == 'failed':
                    raise RuntimeError('Import failed; inspect progress.json')
                time.sleep(15)
        audit_started = time.monotonic()
        result = audit_run(args.catalog, args.run)
        result['audit_seconds_excluding_wait'] = round(time.monotonic() - audit_started, 2)
        write_report(args.output, result)
        print(json.dumps(result), flush=True)
    except Exception as exc:
        write_report(args.output, dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    main()
