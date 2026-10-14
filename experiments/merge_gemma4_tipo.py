"""Merge a completed language-only pilot into its original full HF checkpoint.

Runs on CPU with two threads. Preserves vision/audio/PLE weights, writes only a
new private directory, and records every changed tensor before native conversion.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from .train_gemma4_tipo import checkpoint_manifest


def fingerprint(tensor):
    import torch
    value = tensor.detach().cpu().contiguous().reshape(-1)
    return hashlib.sha256(memoryview(value.view(torch.uint8).numpy())).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.pilot/'run-config.json').read_text())
    report = json.loads((args.pilot/'report.json').read_text())
    if report.get('adapter_roundtrip') != 'pass':
        raise ValueError('A completed pilot with adapter reload validation is required')
    base = Path(run['arguments']['base']).resolve()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(base) or output.is_relative_to(args.pilot.resolve()):
        raise ValueError('Use a new output directory outside the base and pilot')
    if checkpoint_manifest(base) != run['checkpoint_files']:
        raise ValueError('The base weights differ from the training checkpoint')
    targets = run['target_modules']
    if not targets or any(not x.startswith('model.language_model.') for x in targets):
        raise ValueError('Only language projection adapters are supported')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from transformers import Gemma4ForConditionalGeneration
    from peft import LoraConfig, PeftModel
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    model, loading = Gemma4ForConditionalGeneration.from_pretrained(
        base, device_map={'': 'cpu'}, dtype='auto', local_files_only=True,
        output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs', 'unexpected_keys')):
        raise ValueError(f'Checkpoint loading is incomplete: {loading}')
    before = {k: fingerprint(v) for k, v in model.state_dict().items()}
    # PEFT may shorten saved targets to q_proj/etc. The training process removed
    # vision/audio modules, so those suffixes must NOT be reused on a full model.
    adapter_config = LoraConfig.from_pretrained(args.pilot/'adapter')
    adapter_config.target_modules = set(targets)
    wrapped = PeftModel.from_pretrained(model, args.pilot/'adapter',
                                        config=adapter_config, local_files_only=True)
    merged = wrapped.merge_and_unload(safe_merge=True)
    after = {k: fingerprint(v) for k, v in merged.state_dict().items()}
    if before.keys() != after.keys():
        raise ValueError('Tensor inventory changed during merge')
    changed = sorted(k for k in before if before[k] != after[k])
    allowed = {name+'.weight' for name in targets}
    if not changed or not set(changed) <= allowed:
        raise ValueError('Merge changed unexpected tensors or no tensors')
    output.mkdir(parents=True, exist_ok=False)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size='12GB')
    import shutil
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                 'preprocessor_config.json', 'processor_config.json'):
        if (base/name).is_file():
            shutil.copyfile(base/name, output/name)
    audit = {'base': str(base), 'pilot': str(args.pilot.resolve()),
             'adapter_files': checkpoint_manifest(args.pilot/'adapter'),
             'base_files': run['checkpoint_files'], 'output_files': checkpoint_manifest(output),
             'tensor_count': len(before), 'changed_tensors': changed,
             'unchanged_tensors': len(before)-len(changed),
             'all_non_target_tensors_byte_identical': True,
             'scope': 'Unquantized BF16 base plus language LoRA; this differs from the NF4 training runtime. Native outputs require separate evaluation.'}
    (output/'merge-audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps({'output': str(output), 'changed_tensors': len(changed),
                      'all_non_target_tensors_byte_identical': True}), flush=True)


if __name__ == '__main__':
    main()
