"""Bounded CUDA QLoRA pilot; saves an adapter and raw before/after evidence.

No deployment, GGUF conversion, image training or source-store mutation.
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
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(messages[:2], tokenize=False,
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
    parser.add_argument("--states", type=Path, nargs="*", default=[], help="Private state files for evaluation only")
    args = parser.parse_args()
    if not args.gpu_uuid.startswith("GPU-"):
        parser.error("Specify an NVIDIA UUID; do not rely on CUDA index ordering")
    if min(args.steps, args.grad_accum, args.rank, args.eval_count, args.max_new_tokens) < 1 or args.lr <= 0:
        parser.error("Training/evaluation sizes and learning rate must be positive")
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
    read = lambda name: [json.loads(line) for line in (args.data / (name + ".jsonl")).read_text().splitlines() if line.strip()]
    train, evaluation = read("train"), read("eval")
    validate_splits(train, evaluation)
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    encoded = [encode_example(tok, row, args.max_length) for row in train]
    heldout = eval_subset(evaluation, args.eval_count)
    external = state_cases(args.states)
    for row in heldout:
        encode_example(tok, row, args.max_length)
    for row in external:
        prompt = tok.apply_chat_template(row["messages"], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        count = len(tok.encode(prompt, add_special_tokens=False))
        if count > args.max_length:
            raise ValueError(f"External state exceeds --max-length: {row['source']}, {count}")
    progress("hashing_checkpoint")
    checkpoint_files = checkpoint_manifest(args.base)
    progress("loading", training_rows=len(train), evaluation_rows=len(heldout),
             max_training_tokens=max(len(x["input_ids"]) for x in encoded))
    model, loading = Gemma4ForConditionalGeneration.from_pretrained(
        args.base, local_files_only=True, device_map={"": 0}, dtype=torch.bfloat16,
        attn_implementation="sdpa", output_loading_info=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    loading = {key: sorted(value) if isinstance(value, set) else value
               for key, value in loading.items()}
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError(f"Incomplete pretrained checkpoint load: {loading}")
    # The pretrained vision/audio weights are unused, removed from THIS process.
    # No original checkpoint or shared image encoder is written.
    for attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    for param in model.parameters():
        param.requires_grad_(False)
    # Keep frozen multi-billion-parameter PLE embeddings in BF16: PEFT's default
    # prepare_model_for_kbit_training upcasts these to FP32 and exhausts 16 GiB.
    for name, module in model.named_modules():
        if "norm" in name:
            module.to(dtype=torch.float32)
    targets = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and "language_model" in name
               and any(name.endswith(proj) or name.endswith(proj + ".linear")
                       for proj in ("q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))]
    if not targets:
        raise RuntimeError("No language-only projection modules found")
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank,
                            lora_dropout=0.0, target_modules=targets, task_type="CAUSAL_LM", bias="none"))
    model.enable_input_require_grads()
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not params or any("lora_" not in name or "language_model" not in name for name, _ in params):
        raise RuntimeError("Trainable parameters escaped language-only LoRA")
    model.config.get_text_config().use_cache = False
    save("run-config.json", {"arguments": {k: str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v for k,v in vars(args).items()},
         "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
         "gpu_name": torch.cuda.get_device_name(), "free_vram_at_start": free, "total_vram": total,
         "trainable_parameters": sum(p.numel() for _,p in params), "target_modules": targets,
         "loading": loading, "data_manifest": json.loads((args.data / "manifest.json").read_text()),
         "checkpoint_files": checkpoint_files,
         "external_states": [{"path": r["source"], "sha256": r["source_sha256"]} for r in external],
         "lineage": "New language-only LoRA on the supplied Gemma 4 E2B checkpoint; not a continuation of TIPO v2 GGUF.",
         "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    def generate(row):
        prompt = tok.apply_chat_template(row["messages"][:2], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        t = time.monotonic()
        with torch.inference_mode():
            ids = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                 use_cache=True, pad_token_id=tok.pad_token_id)
        raw = tok.decode(ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        endings = model.generation_config.eos_token_id
        endings = endings if isinstance(endings, list) else [endings]
        truncated = ids.shape[1] - inputs.input_ids.shape[1] >= args.max_new_tokens and ids[0, -1].item() not in endings
        score = score_output(raw, row["required"], row["excluded"])
        score["truncated"] = truncated
        score["pass"] = score["pass"] and not truncated
        return {"family": row["family"], "mode": row["mode"], "request": row["messages"][1]["content"],
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

    before = evaluate("before", disabled=True)
    before_external = evaluate("external-before", disabled=True, rows=external)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW([p for _,p in params], lr=args.lr)
    optimizer.zero_grad(set_to_none=True)
    history = []
    torch.cuda.reset_peak_memory_stats()
    progress("training", total_steps=args.steps)
    for step in range(args.steps):
        losses = []
        # Constant LR after a short warmup; this is an explicitly bounded pilot.
        for group in optimizer.param_groups:
            group["lr"] = args.lr * min(1.0, (step + 1) / 4)
        for micro in range(args.grad_accum):
            row = encoded[(step * args.grad_accum + micro) % len(encoded)]
            ids = torch.tensor([row["input_ids"]], device="cuda")
            n = len(row["input_ids"]) - row["prompt_length"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                use_cache=False, logits_to_keep=n+1)
                logits = outputs.logits[:, :-1].float()
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, -n:].reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            (loss / args.grad_accum).backward()
            losses.append(loss.item())
            del outputs, logits, loss, ids
        grad = torch.nn.utils.clip_grad_norm_([p for _,p in params], 1.0)
        if not math.isfinite(float(grad)) or float(grad) == 0:
            raise RuntimeError("Missing or nonfinite LoRA gradients")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        history.append({"step": step+1, "loss": sum(losses)/len(losses), "gradient_norm": float(grad)})
        progress("training", **history[-1], total_steps=args.steps,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated())
        if (step+1) % 16 == 0 or step+1 == args.steps:
            model.save_pretrained(args.output / f"checkpoint-{step+1}")
            save("loss.json", history)
    peak = torch.cuda.max_memory_allocated()
    model.save_pretrained(args.output / "adapter")
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
               "scope": "Pilot on synthetic instructions. No image-generation quality evaluation or deployment.",
               "final_loss": history[-1]["loss"]}
    save("report.json", summary)
    progress("complete", **summary)


if __name__ == "__main__":
    main()
