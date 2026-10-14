"""Bounded CUDA QLoRA pilot; saves an adapter and raw before/after evidence.

No deployment, GGUF conversion or source-store mutation. --multimodal keeps
frozen vision modules and trains the language LoRA on actual image inputs.
All outputs must be explicitly directed to a private experiment directory.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from .gemma4_tuning_data import score_output, validate_splits
from vrs_int.generation_state import messages_for
from tipo_prompt import instruction_messages, row_messages


def state_cases(paths):
    """Private retrieval states are evaluation inputs, never supervision."""
    rows = []
    for path in paths:
        raw = json.loads(path.read_text())
        state = raw.get("state", raw)
        controls = state["controls"]
        rows.append({"family": path.stem, "mode": "state", "source": str(path),
                     "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "required": list(dict.fromkeys(controls["keep"] + controls["target"])),
                     "excluded": controls["exclude"], "messages": messages_for(state, "state")})
    return rows


def encode_example(tokenizer, row, max_length):
    """Mask the entire instruction; reject truncation instead of corrupting labels."""
    messages = row_messages(row)
    if messages[-1]["role"] != "assistant":
        raise ValueError("Missing assistant target")
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    whole = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=False, enable_thinking=False)
    pids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = tokenizer.encode(whole, add_special_tokens=False)
    if ids[:len(pids)] != pids:
        raise ValueError("Chat template/tokenizer changed the assistant prefix")
    if len(ids) > max_length:
        raise ValueError(f"Example exceeds {max_length} tokens: {len(ids)}; refusing silent truncation")
    if len(ids) <= len(pids):
        raise ValueError("Missing answer tokens")
    return {"input_ids": ids, "labels": [-100] * len(pids) + ids[len(pids):],
            "prompt_length": len(pids)}


def eval_subset(rows, maximum):
    """Round-robin across family/mode groups, not just the first adjacent rows."""
    groups = {}
    for row in rows:
        groups.setdefault((row["family"], row["mode"]), []).append(row)
    chosen = []
    while groups and len(chosen) < maximum:
        for key in list(groups):
            chosen.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
            if len(chosen) == maximum:
                break
    return chosen


def training_order(size, count, seed=42):
    """Shuffle whole epochs without dropping or repeating rows inside an epoch."""
    if size < 1 or count < 1:
        raise ValueError("Training size and count must be positive")
    rng = random.Random(seed)
    result = []
    while len(result) < count:
        epoch = list(range(size))
        rng.shuffle(epoch)
        result.extend(epoch)
    return result[:count]


def summarize(rows):
    count = len(rows)
    return {"n": count, "all_constraints_pass": sum(r["score"]["pass"] for r in rows),
            "required_hits": sum(len(r["required"]) - len(r["score"]["missing"]) for r in rows),
            "required_total": sum(len(r["required"]) for r in rows),
            "outputs_with_forbidden_tags": sum(bool(r["score"]["forbidden"]) for r in rows),
            "outputs_with_conflicts": sum(bool(r["score"]["conflicts"]) for r in rows),
            "scope": "Exact tag matching on constructed held-out instructions; not human-rated prompt quality."}


def checkpoint_manifest(base):
    files = {}
    for path in sorted(base.glob("*.safetensors")):
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
                sha.update(chunk)
        files[path.name] = {"bytes": path.stat().st_size, "sha256": sha.hexdigest()}
    if not files:
        raise ValueError("No Safetensors weights found in local --base")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--eval-count", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument('--multimodal', action='store_true')
    parser.add_argument('--initial-adapter', type=Path,
                        help='Continue a completed pilot adapter with matching original base hashes')
    parser.add_argument("--states", type=Path, nargs="*", default=[], help="Private state files for evaluation only")
    parser.add_argument("--reserved-evaluation", type=Path, nargs="*", default=[],
                        help="Frozen dev/final JSONL; mandatory for native-contract training")
    args = parser.parse_args()
    if not args.gpu_uuid.startswith("GPU-"):
        parser.error("Specify an NVIDIA UUID; do not rely on CUDA index ordering")
    if min(args.steps, args.grad_accum, args.rank, args.eval_count, args.max_new_tokens) < 1 or args.lr <= 0:
        parser.error("Training/evaluation sizes and learning rate must be positive")
    # Reject bad data before CUDA initialization or checkpoint loading.
    read = lambda name: [json.loads(line) for line in (args.data / (name + ".jsonl")).read_text().splitlines() if line.strip()]
    train, evaluation = read("train"), read("eval")
    if not args.multimodal:
        validate_splits(train, evaluation)
    elif not args.reserved_evaluation:
        parser.error('Multimodal training requires reserved dev/final sources')
    if args.multimodal and not any(r.get('input', {}).get('image') for r in train):
        parser.error('Multimodal training requires actual image training rows')
    if any(row.get("prompt_contract") for row in train) and not args.reserved_evaluation:
        parser.error("Native-contract training requires --reserved-evaluation dev.jsonl final.jsonl")
    if args.reserved_evaluation:
        from experiments.tipo_evaluation import assert_training_disjoint, read_rows, validate
        reserved = [row for path in args.reserved_evaluation for row in read_rows(path)]
        validate(reserved)
        if {row['split'] for row in reserved} != {'dev', 'final'}:
            parser.error("Both dev and final evaluation sources must be reserved")
        assert_training_disjoint([*train, *evaluation], reserved)
        from experiments.tipo_supervision import SCHEMA, validate_supervision
        manifest = json.loads((args.data / "manifest.json").read_text())
        if any(row.get('schema') == SCHEMA for row in [*train, *evaluation]):
            if manifest.get('schema') != SCHEMA:
                raise ValueError("Authored supervision requires its frozen manifest")
            validate_supervision(train, evaluation, reserved, allow_images=args.multimodal)
            for name in ('train.jsonl', 'eval.jsonl'):
                actual = hashlib.sha256((args.data / name).read_bytes()).hexdigest()
                if manifest['files'].get(name) != actual:
                    raise ValueError("Frozen supervision checksum differs: " + name)
        elif args.multimodal:
            raise ValueError('Multimodal training requires frozen authored supervision')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma4ForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(42)
    random.seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    free, total = torch.cuda.mem_get_info()
    if free < 11 * 1024**3:
        raise RuntimeError("This Gemma 4 E2B pilot needs at least 11 GiB free VRAM")
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        path = args.output / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
        tmp.replace(path)
    def progress(stage, **kw):
        value = {"stage": stage, "elapsed_seconds": time.monotonic() - start, **kw}
        save("progress.json", value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    start = time.monotonic()
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    processor = None
    if args.multimodal:
        from .tipo_multimodal import make_processor, encode_multimodal, image_inputs, IMAGE_CONTRACT
        processor = make_processor(args.base)
        tok = processor.tokenizer
    encode = (lambda row: encode_multimodal(processor, row, args.max_length)) if processor else (
              lambda row: encode_example(tok, row, args.max_length))
    encoded = [encode(row) for row in train]
    order = training_order(len(encoded), args.steps * args.grad_accum)
    heldout = eval_subset(evaluation, args.eval_count)
    if args.multimodal:
        images = eval_subset([r for r in evaluation if r['input'].get('image')], (args.eval_count+1)//2)
        texts = eval_subset([r for r in evaluation if not r['input'].get('image')], args.eval_count//2)
        if not images:
            raise ValueError('Multimodal validation must include held-out images')
        heldout = [r for pair in zip(images, texts) for r in pair]
        heldout += images[len(texts):] + texts[len(images):]
    external = state_cases(args.states)
    for row in heldout:
        encode(row)
    for row in external:
        prompt = tok.apply_chat_template(row["messages"], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        count = len(tok.encode(prompt, add_special_tokens=False))
        if count > args.max_length:
            raise ValueError(f"External state exceeds --max-length: {row['source']}, {count}")
    progress("hashing_checkpoint")
    checkpoint_files = checkpoint_manifest(args.base)
    initial_run = None
    if args.initial_adapter:
        initial_run = json.loads((args.initial_adapter.parent/'run-config.json').read_text())
        if initial_run['checkpoint_files'] != checkpoint_files:
            raise ValueError('Initial adapter was trained on a different base')
        if json.loads((args.initial_adapter.parent/'report.json').read_text()).get('adapter_roundtrip') != 'pass':
            raise ValueError('Initial pilot must be completed and reload-verified')
    progress("loading", training_rows=len(train), evaluation_rows=len(heldout),
             max_training_tokens=max(len(x["input_ids"]) for x in encoded))
    model, loading = Gemma4ForConditionalGeneration.from_pretrained(
        args.base, local_files_only=True, device_map={"": 0}, dtype=torch.bfloat16,
        attn_implementation="sdpa", output_loading_info=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=['vision_tower', 'embed_vision', 'lm_head'] if args.multimodal else None,
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    loading = {key: sorted(value) if isinstance(value, set) else value
               for key, value in loading.items()}
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError(f"Incomplete pretrained checkpoint load: {loading}")
    # Remove only unused modalities from THIS process. Original weights stay intact.
    for attr in (("audio_tower", "embed_audio") if args.multimodal else
                 ("vision_tower", "audio_tower", "embed_vision", "embed_audio")):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    vision_evidence = {'calls': 0, 'shapes': []}
    if args.multimodal:
        # Preserve the full-precision frozen visual path; no vision adapters.
        model.model.vision_tower.to(dtype=torch.float16)
        model.model.embed_vision.to(dtype=torch.bfloat16)
        def seen_vision(module, inputs, kwargs):
            vision_evidence['calls'] += 1
            pixels = kwargs.get('pixel_values')
            shape = list(pixels.shape) if pixels is not None else None
            if shape not in vision_evidence['shapes']:
                vision_evidence['shapes'].append(shape)
        model.model.vision_tower.register_forward_pre_hook(seen_vision, with_kwargs=True)
    for param in model.parameters():
        param.requires_grad_(False)
    # Keep frozen multi-billion-parameter PLE embeddings in BF16: PEFT's default
    # prepare_model_for_kbit_training upcasts these to FP32 and exhausts 16 GiB.
    for name, module in model.named_modules():
        if "norm" in name and (not args.multimodal or 'language_model' in name):
            module.to(dtype=torch.float32)
    targets = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and "language_model" in name
               and any(name.endswith(proj) or name.endswith(proj + ".linear")
                       for proj in ("q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))]
    if not targets:
        raise RuntimeError("No language-only projection modules found")
    if args.initial_adapter:
        from peft import PeftModel
        config = LoraConfig.from_pretrained(args.initial_adapter)
        if config.r != args.rank or set(initial_run['target_modules']) != set(targets):
            raise ValueError('Initial adapter rank/targets differ')
        config.target_modules = set(targets)
        model = PeftModel.from_pretrained(model, args.initial_adapter, config=config, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank,
                                lora_dropout=0.0, target_modules=targets, task_type="CAUSAL_LM", bias="none"))
    model.enable_input_require_grads()
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not params or any("lora_" not in name or "language_model" not in name for name, _ in params):
        raise RuntimeError("Trainable parameters escaped language-only LoRA")
    model.config.get_text_config().use_cache = False
    def save_adapter(path):
        model.save_pretrained(path)
        # Preserve full module paths. PEFT suffix compression is unsafe when a
        # language adapter is reloaded into a model with matching vision names.
        config_path = path/'adapter_config.json'
        config = json.loads(config_path.read_text())
        config['target_modules'] = targets
        config_path.write_text(json.dumps(config, indent=2))
    save("run-config.json", {"arguments": {k: str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v for k,v in vars(args).items()},
         "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
         "gpu_name": torch.cuda.get_device_name(), "free_vram_at_start": free, "total_vram": total,
         "trainable_parameters": sum(p.numel() for _,p in params), "target_modules": targets,
         "loading": loading, "data_manifest": json.loads((args.data / "manifest.json").read_text()),
         "checkpoint_files": checkpoint_files,
         "reserved_evaluation_files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in args.reserved_evaluation},
         "training_order": {"seed": 42, "indices": order,
                            "unique_rows": len(set(order)), "epochs": len(order)/len(train)},
         "external_states": [{"path": r["source"], "sha256": r["source_sha256"]} for r in external],
         "initial_adapter_files": checkpoint_manifest(args.initial_adapter) if args.initial_adapter else None,
         "multimodal": bool(processor),
         "image_processor": processor.image_processor.to_dict() if processor else None,
         "image_input_contract": IMAGE_CONTRACT if processor else None,
         "image_preprocessing": 'native uint8 bicubic resize; processor resize/rescale disabled; native media newline placement' if processor else None,
         "multimodal_source_sha256": hashlib.sha256(Path(__file__).with_name('tipo_multimodal.py').read_bytes()).hexdigest() if processor else None,
         "vision_precision": 'fp16 frozen weights; BF16 training autocast' if processor else None,
         "lineage": 'Continuation of recorded initial adapter' if args.initial_adapter else 'New language-only LoRA on the supplied base',
         "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    def generate(row):
        if processor and row.get('input', {}).get('image'):
            inputs = image_inputs(processor, row).to('cuda')
            request = row['input']['prompt']
        else:
            prompt = tok.apply_chat_template(instruction_messages(row), tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
            inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            request = instruction_messages(row)[-1]['content']
        t = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            ids = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                 use_cache=True, pad_token_id=tok.pad_token_id)
        raw = tok.decode(ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        endings = model.generation_config.eos_token_id
        endings = endings if isinstance(endings, list) else [endings]
        truncated = ids.shape[1] - inputs.input_ids.shape[1] >= args.max_new_tokens and ids[0, -1].item() not in endings
        score = score_output(raw, row["required"], row["excluded"])
        score["truncated"] = truncated
        score["pass"] = score["pass"] and not truncated
        return {"id": row.get('id'), "family": row["family"], "mode": row["mode"], "request": request,
                "raw": raw, "seconds": time.monotonic() - t,
                "required": row["required"], "excluded": row["excluded"],
                "score": score}

    def evaluate(stage, disabled=False, rows=None):
        rows = heldout if rows is None else rows
        model.eval()
        model.gradient_checkpointing_disable()
        results = []
        with model.disable_adapter() if disabled else nullcontext():
            for i, row in enumerate(rows):
                results.append(generate(row))
                progress(stage, completed=i+1, total=len(rows))
                save(stage + ".json", {"outputs": results, "summary": summarize(results)})
        return results

    before = evaluate("before", disabled=not bool(args.initial_adapter))
    before_external = evaluate("external-before", disabled=not bool(args.initial_adapter), rows=external)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW([p for _,p in params], lr=args.lr)
    optimizer.zero_grad(set_to_none=True)
    history = []
    torch.cuda.reset_peak_memory_stats()
    progress("training", total_steps=args.steps)
    vision_before_training = vision_evidence['calls']
    image_rows_seen = 0
    for step in range(args.steps):
        losses = []
        # Constant LR after a short warmup; this is an explicitly bounded pilot.
        for group in optimizer.param_groups:
            group["lr"] = args.lr * min(1.0, (step + 1) / 4)
        for micro in range(args.grad_accum):
            row = encoded[order[step * args.grad_accum + micro]]
            ids = torch.tensor([row["input_ids"]], device="cuda")
            model_inputs = {k: v.to('cuda') for k,v in row['model_inputs'].items()} if row.get('model_inputs') else {
                'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
            if row.get('model_inputs'):
                image_rows_seen += 1
            n = len(row["input_ids"]) - row["prompt_length"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**model_inputs, use_cache=False, logits_to_keep=n+1)
                logits = outputs.logits[:, :-1].float()
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, -n:].reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            (loss / args.grad_accum).backward()
            losses.append(loss.item())
            del outputs, logits, loss, ids, model_inputs
        grad = torch.nn.utils.clip_grad_norm_([p for _,p in params], 1.0)
        if not math.isfinite(float(grad)) or float(grad) == 0:
            raise RuntimeError("Missing or nonfinite LoRA gradients")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        history.append({"step": step+1, "loss": sum(losses)/len(losses), "gradient_norm": float(grad)})
        progress("training", **history[-1], total_steps=args.steps,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated())
        if (step+1) % 16 == 0 or step+1 == args.steps:
            save_adapter(args.output / f"checkpoint-{step+1}")
            save("loss.json", history)
    peak = torch.cuda.max_memory_allocated()
    vision_training_calls = vision_evidence['calls'] - vision_before_training
    if processor and (not image_rows_seen or vision_training_calls < image_rows_seen):
        raise RuntimeError('Image training did not execute the vision tower')
    save_adapter(args.output / "adapter")
    tok.save_pretrained(args.output / "adapter")
    del optimizer
    torch.cuda.empty_cache()
    after = evaluate("after")
    after_external = evaluate("external-after", rows=external)
    # Verify serializing/reloading the adapter preserves a deterministic output.
    model.load_adapter(args.output / "adapter", adapter_name="roundtrip", is_trainable=False)
    model.set_adapter("roundtrip")
    model.eval()
    roundtrip = generate(heldout[0])
    if roundtrip["raw"] != after[0]["raw"]:
        raise RuntimeError("Reloaded adapter changed deterministic generation")
    summary = {"before": summarize(before), "after": summarize(after), "adapter_roundtrip": "pass",
               "external_before": summarize(before_external), "external_after": summarize(after_external),
               "training_steps": args.steps, "training_examples_seen": args.steps * args.grad_accum,
               "training_peak_allocated_bytes": peak, "seconds": time.monotonic() - start,
               "training_unique_rows_seen": len(set(order)),
               "training_epochs": len(order)/len(train),
               "vision_training_calls": vision_training_calls, "image_training_rows_seen": image_rows_seen,
               "vision_evidence": vision_evidence,
               "scope": "Mixed image/text language LoRA with frozen vision" if processor else "Text-only language LoRA",
               "quality_limit": 'Held-out tag checks are not native runtime, visual fidelity or creative quality validation.',
               "final_loss": history[-1]["loss"]}
    save("report.json", summary)
    progress("complete", **summary)


if __name__ == "__main__":
    main()
