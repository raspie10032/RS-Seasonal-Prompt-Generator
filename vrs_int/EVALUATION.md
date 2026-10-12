# VRS-INT held-out recommendation evaluation

[한국어 기술 보고서](../docs/VRS_INT_HOLDOUT_REPORT_KOR.md)

`evaluate` reads one exact immutable snapshot. It does not build, update, activate,
decode images, load an encoder, or download weights. NumPy is the only required
third-party package for the default offline mode.

```bash
python -m vrs_int evaluate --store ./vrs-int-data/memory --snapshot EXACT_SNAPSHOT_SHA256 --split val --ks 5,10,20 --neighbors 32 --rare-max 5 --backend numpy --output ./vrs-int-data/evaluation-val.json
python -m vrs_int evaluate --store ./vrs-int-data/memory --snapshot EXACT_SNAPSHOT_SHA256 --split test --ks 5,10,20 --neighbors 32 --rare-max 5 --backend numpy --include-cases --output ./vrs-int-data/evaluation-test.json
```

Set `--ratings g,s` to restrict **both** queries and training evidence. Existing
output files are never overwritten; outputs inside `objects/`, `snapshots/`, or
at `CURRENT` are rejected. Outputs can expose personal metadata; keep them local.
The publication checker rejects unreviewed JSON regardless of its filename.

## Protocol and leakage controls

1. Assign train/val/test before building features. Keep near-duplicates, image
   variants and shared source identities in the same split. Do not select splits
   based on this evaluator's results.
2. The existing `build` command encodes all listed splits. Pin its returned hash.
   An older migrated train-only feature subset needs holdout encoding before it
   can be evaluated; missing vectors are reported, not guessed.
3. Training evidence must have a vector and pass the ratings filter. Every
   val/test hash and source ID in that snapshot is blocked, including holdouts
   in the other split or outside the ratings filter. A query's own hash is also
   excluded by the shared recommender. Duplicate train image hashes get one vote.
4. Evaluate val to choose neighbors/k/ratings/rare threshold. Freeze these choices
   and evaluate test once for the final result. The tool cannot prevent a person
   from repeatedly tuning on test; retain reports and configuration.
5. Gold labels are read for metrics after retrieval, never passed into its vote.
   Category truth uses explicit source categories, not inferred names. Ground
   truth is the snapshot's retained tags; raw audit tags are used only to inspect
   storage preservation.

Duplicate query image hashes within the requested split are evaluated once in
lexicographic observation-key order. Missing query vectors and duplicate queries
are listed in `coverage.skipped`. An empty split or no evaluable vectors fails.
An empty training pool yields zero hits, not a successful quality claim.

## Metrics and denominators

| Output | Definition |
| --- | --- |
| `top_k` | P@k = correct tags / k, R@k = correct tags / retained truth count |
| `precision_micro`, `recall_micro` | Sum hits / sum slots or sum truth across queries |
| `precision_macro`, `recall_macro` | Average per-query scores; empty truth excluded from macro recall |
| `category_top_k` | Artist/character/copyright hits in the global top-k list, explicit category required on prediction and truth |
| `identity_list_at_3` | Separate recall/precision for the recommender's three candidates per identity category |
| `rare.top_k` | Recall for tags occurring in 1..rare_max eligible training images |
| `unseen_top_k` | Recall for truth tags with training image frequency zero; kept separate from known rare tags |
| `rare` preservation | Original audit vs retained tags on deduplicated eligible train images |
| `storage_preservation` | Original audit vs retained tags on all live train observations passing ratings, including missing vectors and blocked duplicates |
| `errors` | False-positive/negative counts, queries with either, no recommendations, zero-hit recommendation lists |

Short lists still divide precision by **k**, so returning one correct tag for
k=20 does not receive perfect P@20. Zero-denominator ratios serialize as `null`,
not zero or NaN. Counts, denominators and sums accompany averages. The precision
fields on rare/unseen/category subsets use k slots as well; use their **recall**
fields to assess subset recovery. `identity_list_at_3` uses three slots.

Rarity is fixed using train only. Preservation includes intentional filtering
(e.g. highres); a removed noise label is a storage removal, not an algorithm bug.
Missing/unreadable source-tag audit falls back to retained tags and audit coverage
is reported, so 100% preservation with incomplete audit is not a losslessness
proof. Storage rarity counts observations; retrieval rarity counts distinct
eligible images. These denominators must not be compared as identical quantities.

`--include-cases` adds local per-query ranked candidates, neighbor observation
keys, false positives/negatives, rare misses, unseen truth and identity errors.
`per_tag` records train frequency, audit frequency/preserved occurrences, truth
support, hits and recall for each k. False positives mean annotation disagreements,
not proof a tag is visually or semantically wrong.

## Reproducibility and separate validation stages

The report pins snapshot/parent/encoder contract, parameters, implementation file
hashes, backend and NumPy version, evaluated-key digest and its own canonical
SHA-256 (excluding `report_sha256` itself). No timestamp or moving `latest` alias
is used. Run the same environment/configuration twice to reproduce the report.
Different BLAS/device floating-point reductions can reorder very close ties;
NumPy and CUDA are not asserted to be bit-identical.

Offline calculation checks:

```bash
python -m unittest discover -s tests -p "test_vrs_int*.py" -v
python scripts/check_code_only.py
```

Tests generate fictional labels and f32le/i8 features in temporary stores. They
check hand-calculated metrics, leakage, ranking votes, category/rare/unseen
recovery, missing vectors, mutation isolation, reproducibility, CLI and existing
snapshot/lineage/transfer behavior. No image corpus or model is required.

For **actual image quality**, use privately held correctly annotated images and
the real encoder in the existing `build` step first. Verify the encoder contract
and snapshot. Then evaluate the stored real features. To validate the CUDA
retrieval implementation on those same features:

```bash
python -m vrs_int evaluate --store ./vrs-int-data/memory --snapshot EXACT_SNAPSHOT_SHA256 --split test --ks 5,10,20 --neighbors 32 --rare-max 5 --backend cuda --gpu GPU-YOUR-UUID --include-cases --output ./vrs-int-data/evaluation-test-cuda.json
```

CUDA mode needs CUDA-enabled PyTorch and explicit GPU selection, and fails
without it. It does not encode images either. Real encoder extraction,
CUDA/NumPy ranking agreement and real-label retrieval quality are distinct
checks. The offline suite alone proves none of them. Record dependency/driver
versions and GPU, examine close ties and private error cases, and compare raw
counts before claiming quality. No universal pass threshold is supplied.

Exact hashes/source IDs do not detect unknown near-duplicates, and a snapshot
contract cannot establish that upstream encoder training never saw holdout
images. Million-image evaluation speed/RAM use is unmeasured; search remains
exact streaming search with a host-side feature cache and additional metadata
passes. The weighted vote formula and existing `visual_verdict` output are
unchanged: scores are neither probabilities nor SWEGCA A/R/U judgments.
