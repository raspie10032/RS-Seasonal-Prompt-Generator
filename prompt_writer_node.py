"""Instruction-following local prompt writer, separate from TIPO expansion.

Node discovery imports only the standard library. Models and inference
dependencies are resolved only when the user runs a nonempty prompt.
"""
import hashlib
import os
import re
import threading

AUTO_WRITER = "(auto: Qwen3-4B-Instruct-2507)"
MODEL_REPO = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
MODEL_REVISION = "a06e946bb6b655725eafa393f4a9745d460374c9"
MODEL_FILE = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
MODEL_SHA256 = "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597"
_verified = set()
_verify_lock = threading.Lock()

CREATIVITY = {
    "faithful": "Translate and reorganize only the supplied visual details. Do not invent any details.",
    "balanced": "Add a few compatible details of lighting, atmosphere and composition only. Do not invent an outfit, character identity, hair/eye color, body feature, era or art style.",
    "creative": "Enrich unspecified details into a coherent, imaginative scene. Explicit details remain binding; never replace them.",
}
LENGTHS = {
    "short": ("about 10-20 tags or 30-50 words", 256),
    "medium": ("about 20-35 tags or 60-100 words", 512),
    "long": ("about 35-55 tags or 100-160 words", 900),
}


def _engine():
    try:
        from . import tipo_engine
    except ImportError:
        import tipo_engine
    return tipo_engine


def _resolve_model(model):
    te = _engine()
    if model != AUTO_WRITER:
        # Model selections are basenames from the node's dropdown.
        if os.path.basename(model) != model or not te._is_model_file(model):
            raise ValueError("Select an instruction-tuned GGUF language model from the dropdown")
        path = os.path.join(te._gguf_dir(), model)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path
    path = os.path.join(te._gguf_dir(), MODEL_FILE)
    te._stream_download(
        f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILE}", path)
    with _verify_lock:
        stat = os.stat(path)
        signature = (path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if signature not in _verified:
            digest = hashlib.sha256()
            with open(path, "rb") as source:
                for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != MODEL_SHA256:
                raise RuntimeError(f"Default writer model failed SHA-256 verification: {path}. Move the invalid file aside and retry.")
            _verified.add(signature)
    return path


def _messages(prompt, output_format, creativity, length):
    if output_format not in ("tags", "natural_language"):
        raise ValueError("Unknown output format")
    guide, _ = LENGTHS[length]
    output = (
        "Output ONLY one comma-separated list of concise English image-generation tags. "
        "Use Danbooru-style tags where applicable, and plain English visual phrases otherwise. "
        "Use exactly one count tag for the TOTAL of each gender, e.g. two women = 2girls. "
        "Never write 1girl twice to represent two women. Preserve adult ages. "
        "For multiple people, preserve who wears which outfit and their spatial relationship, "
        "using longer phrases when needed (one woman wearing ..., the other wearing ...)."
        if output_format == "tags" else
        "Output ONLY one fluent English paragraph suitable as an image-generation prompt. "
        "Write complete connected sentences, not a comma-separated list of fragments. "
        "Each sentence needs a subject and a verb, for example: A woman stands in a park. "
        "Convert any input tags into prose: for example 1girl becomes a girl."
    )
    system = (
        "You write image-generation prompts from Korean or English descriptions or tags. "
        "Preserve every explicit subject, number of people, color, clothing, action, location, "
        "weather and time. Never drop explicit quantities or who wears which item. "
        "Translate Korean clothing precisely: 치마 = skirt, 바지 = pants, "
        "원피스 = dress, 수영복 = swimsuit, 셔츠 = shirt, 재킷 = jacket. "
        "Preserve spatial relations: 마주 앉아 = sitting facing each other, "
        "나란히 = side by side, 뒤에 = behind. Do not replace facing each other with together. "
        "Respect negations. Do not turn a dress into a swimsuit. "
        "Never introduce contradictions. " + CREATIVITY[creativity] + " " + output +
        " No preamble, explanation, headings, quotes or markdown. Aim for " + guide +
        "; preserving the supplied details matters more than the target length."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": prompt.strip()}]


def _completion_text(response):
    choice = response["choices"][0]
    if choice.get("finish_reason") == "length":
        raise RuntimeError("Model output was truncated. Use a shorter description or a larger length setting.")
    text = _engine()._strip_reasoning(choice["message"]["content"]).strip()
    if not text:
        raise RuntimeError("Model returned no usable prompt")
    return text


def _clean_tags(text):
    # Instruction models can use longer visual phrases to bind actions/outfits
    # to different subjects. TIPO's prose-rejection heuristic would lose them.
    text = re.sub(r"^```[^\n]*\n|\n```$", "", text.strip()).strip()
    text = re.sub(r"^(?:tags|prompt|output)\s*:\s*", "", text, flags=re.I)
    tags = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*]\s+|\d+[.)]\s+)", "", line)
        tags.extend(part.strip().strip('"').rstrip('.') for part in line.split(','))
    return _engine()._dedup(', '.join(tag for tag in tags if tag))


def write_messages(messages, model=AUTO_WRITER, output_format="tags", length="medium",
                   temperature=0.6, seed=0, cpu_threads=2):
    """GPU-only one-shot writer for reference-based prompts; always close its model.

    The instance is private, so releasing it never closes another node's cached
    model. No CPU environment override or automatic fallback applies here.
    """
    if output_format not in ("tags", "natural_language") or length not in LENGTHS:
        raise ValueError("Invalid output format or length")
    te = _engine()
    if not te._ensure_deps():
        raise RuntimeError("llama-cpp-python is unavailable")
    te.require_gpu_offload()
    path = _resolve_model(model)
    with te._inference_lock:
        llm = te.load_tipo(path, -1, n_ctx=4096, n_threads=cpu_threads, cache=False)
        if llm is None:
            raise RuntimeError(f"Could not load model: {path}")
        try:
            response = llm.create_chat_completion(
                messages=messages, max_tokens=LENGTHS[length][1],
                temperature=float(temperature), top_p=0.8, top_k=20,
                repeat_penalty=1.05, seed=int(seed) % (2 ** 31))
            text = _completion_text(response)
            text = _clean_tags(text) if output_format == "tags" else text.strip().strip('"')
            if not text:
                raise RuntimeError("Model returned no usable prompt")
            return text
        finally:
            llm.close()


class RSPromptWriter:
    @classmethod
    def INPUT_TYPES(cls):
        te = _engine()
        try:
            models = sorted(name for name in os.listdir(te._gguf_dir())
                            if te._is_model_file(name)
                            and os.path.isfile(os.path.join(te._gguf_dir(), name)))
        except OSError:
            models = []
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "원하는 장면을 한국어/영어로 설명하거나 Seasonal 노드의 출력을 연결하세요."}),
                "model": ([AUTO_WRITER] + models, {"default": AUTO_WRITER}),
                "output_format": (["tags", "natural_language"], {"default": "tags"}),
                "creativity": (list(CREATIVITY), {"default": "balanced"}),
                "length": (list(LENGTHS), {"default": "medium"}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.5, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 100,
                    "tooltip": "-1 = all GPU layers (default); 0 = explicit CPU mode. A GPU build is required; no automatic CPU fallback."}),
                "cpu_threads": ("INT", {"default": 2, "min": 1, "max": 8,
                    "tooltip": "CPU helper threads, including prompt processing. Keep low to preserve desktop responsiveness."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = "prompt"

    def run(self, prompt, model=AUTO_WRITER, output_format="tags",
            creativity="balanced", length="medium", temperature=0.6, seed=0,
            gpu_layers=-1, cpu_threads=2):
        if not prompt or not prompt.strip():
            return ("",)
        te = _engine()
        messages = _messages(prompt, output_format, creativity, length)
        try:
            if not te._ensure_deps():
                raise RuntimeError("llama-cpp-python is unavailable. Check the [TIPO] dependency error in the console and install a compatible build and runtime.")
            layers = 0 if os.environ.get("RS_TIPO_FORCE_CPU", "").lower() in ("1", "true", "yes", "on") else int(gpu_layers)
            if layers != 0:
                te.require_gpu_offload()
            path = _resolve_model(model)
            with te.tipo_inference():
                llm = te.load_tipo(path, layers, n_ctx=4096, n_threads=cpu_threads)
                if llm is None:
                    raise RuntimeError(f"Could not load model: {path}")
                response = llm.create_chat_completion(
                    messages=messages, max_tokens=LENGTHS[length][1],
                    temperature=float(temperature), top_p=0.8, top_k=20,
                    repeat_penalty=1.05, seed=int(seed) % (2 ** 31))
            text = _completion_text(response)
            if output_format == "tags":
                text = _clean_tags(text)
            else:
                text = re.sub(r"^```[^\n]*\n|\n```$", "", text).strip().strip('"')
            if not text:
                raise RuntimeError("Model returned no usable prompt")
            return (text,)
        except Exception as exc:
            # Show a failed node instead of silently passing unchanged text
            # downstream and making an inference failure look successful.
            raise RuntimeError(f"[RS Prompt Writer] Generation failed: {exc}") from exc
