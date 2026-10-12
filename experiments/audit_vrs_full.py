import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import sparse

from experiments.tag_policy import IDENTITY, typed_tags
from experiments.run_vrs_trial import Monitor, setup_gpu
from experiments.vrs_full import FullMemory, write_json
from experiments.vrs_memory import digest_file
from experiments.vrs_cuda import AssociationCUDA

root = Path(sys.argv[1])
destination = Path(sys.argv[2])
receipt = json.loads((root / 'receipt.json').read_text())
assert receipt['state'] == 'complete'
torch = setup_gpu()
with Monitor() as monitor:
    memory = FullMemory(root)
    x = memory.x
    frequency = np.asarray(x.sum(axis=0)).flatten().astype(np.int64)
    result = {'snapshot': str(root), 'receipt_sha256': digest_file(root / 'receipt.json'),
              'artifact_hashes_verified': len(receipt['files_sha256'])}
    code = Path('experiments')
    for name, sha in receipt['code_sha256'].items():
        assert digest_file(code / name) == sha, name
    result['build_code_hashes_match_current'] = True
    names = {c: set() for c in IDENTITY}
    vocab = set(memory.vocab)
    source_ids = set()
    core_occurrences = 0
    removed = Counter()
    db = sqlite3.connect((root / 'observations.sqlite3').as_uri() + '?mode=ro', uri=True)
    assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    for key, raw, kept_raw, removed_raw in db.execute(
            'SELECT id,source_record,kept_tags,removed_tags FROM observations'):
        source_ids.add(key)
        source = json.loads(raw)
        kept = set(json.loads(kept_raw))
        typed = typed_tags(source)
        for cat in IDENTITY:
            assert typed[cat] <= kept
            assert typed[cat] <= vocab
            names[cat].update(typed[cat])
            core_occurrences += len(typed[cat])
        removed.update(json.loads(removed_raw).keys())
    db.close()
    assert len(source_ids) == receipt['ingest']['source_records']
    assert dict(removed) == receipt['ingest']['removed_tags']
    assert {i for r in memory.rows for i in r['source_ids']} == source_ids
    assert len({r['image_sha256'] for r in memory.rows}) == len(memory.rows)
    result['source_records_checked'] = len(source_ids)
    result['explicit_core_occurrences_preserved'] = core_occurrences
    result['explicit_identity_types_preserved'] = {k: len(v) for k, v in names.items()}
    result['all_source_ids_and_exact_image_dedup_verified'] = True

    graph = json.loads((root / 'pairs.json').read_text())
    expected_total = sum(int(n) ** 2 for n in np.diff(x.indptr))
    actual_total, nonzero = 0, 0
    diagonal_checked = 0
    for shard in graph['shards']:
        with np.load(root / shard['file'], allow_pickle=False) as data:
            a, b = int(data['row_start']), int(data['row_stop'])
            counts, indices, ptr = data['counts'], data['indices'], data['indptr']
            assert np.all(counts > 0)
            assert np.all(data['status'] == 1)
            assert np.all(indices >= 0) and np.all(indices < x.shape[1])
            actual_total += int(counts.sum())
            nonzero += len(counts)
            for i in range(a, b):
                start, stop = ptr[i-a:i-a+2]
                own = counts[start:stop][indices[start:stop] == i]
                assert len(own) == 1 and int(own[0]) == frequency[i]
                diagonal_checked += 1
    assert actual_total == expected_total
    assert nonzero - x.shape[1] == graph['directed_pairs_excluding_self']
    result['graph'] = {'diagonals_checked': diagonal_checked,
                       'all_positive_statuses_and_column_bounds_checked': True,
                       'aggregate_cooccurrence_matches_document_squares': True,
                       'directed_pairs_excluding_self': nonzero - x.shape[1]}

    mask = np.load(root / 'vector_mask.npy', allow_pickle=False)
    vectors = np.load(root / 'vectors.npy', mmap_mode='r', allow_pickle=False)
    assert vectors.shape == (len(memory.rows), 768)
    assert int(mask.sum()) == receipt['features']['pooled']
    checked_vectors, max_error = 0, 0.0
    for start in range(0, len(mask), 2048):
        block = np.array(vectors[start:start+2048], dtype=np.float32)
        enabled = mask[start:start+2048]
        assert np.all(np.isfinite(block))
        assert not np.any(block[~enabled])
        norm = np.linalg.norm(block[enabled], axis=1)
        if len(norm):
            max_error = max(max_error, float(np.max(np.abs(norm-1))))
        checked_vectors += int(enabled.sum())
    assert max_error < .005
    result['vectors'] = {'checked': checked_vectors, 'maximum_unit_norm_error': max_error,
                         'missing_vectors_are_zero_and_masked': True}

    queries = []
    with AssociationCUDA(receipt['core_root']) as judge:
        for cat in IDENTITY:
            candidates = [t for t in names[cat] if frequency[memory.index[t]] > 0]
            tag = min(candidates, key=lambda t: (-frequency[memory.index[t]], t))
            queries.append({'case': 'core_' + cat, 'result': memory.query([tag], judge=judge)})
            singletons = sorted(t for t in candidates if frequency[memory.index[t]] == 1)
            if singletons:
                singleton = memory.query([singletons[0]], min_support=1, judge=judge)
                assert singleton['matching_records'] == 1 and singleton['candidates']
                queries.append({'case': 'singleton_' + cat, 'result': singleton})
        excluded = memory.query(['hatsune miku'], ['twintails'], ratings=('g','s'), judge=judge)
        queries.append({'case': 'excluded_attribute', 'result': excluded})
        unknown = memory.query(['hatsune miku', 'nonexistent rs character 2099'], judge=judge)
        assert unknown['verdict'] == 'U' and not unknown['candidates']
        queries.append({'case': 'unknown_required', 'result': unknown})
        combined_found = False
        for row in memory.rows:
            grouped = {cat: [t for t in row['tags'] if t in names[cat]] for cat in IDENTITY}
            if all(grouped.values()):
                seeds = [grouped[cat][0] for cat in IDENTITY]
                compound = memory.query(seeds, min_support=1, ratings=(row['rating'],), judge=judge)
                assert compound['matching_records'] > 0 and compound['candidates']
                queries.append({'case': 'artist_character_copyright_together', 'result': compound})
                combined_found = True
                break
        assert combined_found, 'No observation could test all three identity categories together'
    by_id = {r['id']: r for r in memory.rows}
    refs = 0
    for entry in queries:
        q = entry['result']
        for candidate in q['candidates']:
            assert candidate['tag'] not in q['excluded']
            for key in candidate['image_ids']:
                row = by_id[key]
                assert set(q['input_tags'] + [candidate['tag']]) <= set(row['tags'])
                assert not set(q['excluded']).intersection(row['tags'])
                assert not q['ratings'] or row['rating'] in q['ratings']
                refs += 1
    result['query_evidence_references_checked'] = refs
    result['queries'] = queries
    result['audit_process_peak_vram_sample_mib'] = max((s['mib'] for s in monitor.samples), default=None)
    result['all_passed'] = True
write_json(destination, result)
print(json.dumps({k:v for k,v in result.items() if k != 'queries'}, ensure_ascii=False))
