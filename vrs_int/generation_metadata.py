"""Read generation intent without executing workflows or guessing image content.

Only selected prompt/settings fields are retained. Workflow dumps, arbitrary
PNG fields, credentials and model paths are deliberately not copied to records.
"""
import json
import re

from .tag_policy import canonical

VERSION = "generation-metadata-v3"
SETTINGS = ("seed", "steps", "sampler", "sampler_name", "scheduler", "noise_schedule",
            "scale", "cfg", "cfg_scale", "width", "height", "model_name", "model_hash")


def object_value(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            pass
    return {}


def unique(values):
    return list(dict.fromkeys(v.strip() for v in values if isinstance(v, str) and v.strip()))


def caption_parts(value):
    caption = object_value(object_value(value).get("caption"))
    base = caption.get("base_caption", "")
    characters = []
    for row in caption.get("char_captions", []):
        if isinstance(row, dict) and isinstance(row.get("char_caption"), str):
            characters.append({"prompt": row["char_caption"],
                               "centers": row.get("centers", [])})
    return base if isinstance(base, str) else "", characters


def comfy_prompts(graph, filename):
    """Follow saved-image ancestry and explicit positive/negative input ports."""
    nodes = {str(k): v for k, v in graph.items() if isinstance(v, dict) and "class_type" in v}
    if not nodes:
        return None

    def link(value):
        return isinstance(value, list) and len(value) == 2 and str(value[0]) in nodes and type(value[1]) is int

    unresolved = set()

    def text(value, stack=()):
        if isinstance(value, str):
            return value
        if not link(value):
            return ""
        key = str(value[0])
        if key in stack or len(stack) > 40:
            unresolved.add("cycle-or-depth")
            return ""
        node = nodes[key]; kind = node["class_type"]; inputs = node.get("inputs", {})
        stack = (*stack, key)
        resolve = lambda v: text(v, stack)
        if kind == "ImpactWildcardProcessor":
            return inputs.get("populated_text", "")
        if kind == "ShowText|pysssss":
            return inputs.get("text_0") or resolve(inputs.get("text"))
        if kind.startswith("CLIPTextEncode"):
            return ", ".join(unique(resolve(inputs.get(k)) for k in ("text", "text_g", "text_l")))
        if kind in ("Text Prompt (JPS)", "PrimitiveString", "String", "Text Multiline"):
            return resolve(inputs.get("text", inputs.get("value")))
        if kind == "Text Concatenate (JPS)":
            return str(inputs.get("delimiter", ", ")).join(
                unique(resolve(inputs.get(f"text{i}")) for i in range(1, 6)))
        if kind == "WD14Tagger|pysssss" and isinstance(inputs.get("tags"), str):
            return inputs["tags"]
        if kind in ("ConditioningCombine", "ConditioningConcat"):
            return ", ".join(unique(resolve(v) for k, v in inputs.items() if k.startswith("conditioning")))
        unresolved.add(kind)
        return ""

    saves = [k for k, v in nodes.items() if v["class_type"] == "SaveImage"]
    matching = [k for k in saves if filename.startswith(
        str(nodes[k].get("inputs", {}).get("filename_prefix", "")).replace("\\", "/").split("/")[-1])]
    if len(matching) == 1:
        saves = matching
    if len(saves) != 1:
        # Renamed archives often contain several save nodes. Keep only prompt
        # strings shared by every possible output, never their union.
        alternatives = []
        for save in saves:
            selected = {k: v for k, v in nodes.items() if k == save or v["class_type"] != "SaveImage"}
            candidate = comfy_prompts(selected, filename)
            if candidate is None:
                return None
            alternatives.append(candidate)
        if not alternatives:
            return None
        positive = [p for p in alternatives[0]["positive"] if all(p in x["positive"] for x in alternatives)]
        characters = [p for p in alternatives[0]["characters"] if all(p in x["characters"] for x in alternatives)]
        if not positive and not characters:
            return None
        return {"format": "comfyui", "positive": positive, "characters": characters,
                "negative": unique(p for x in alternatives for p in x["negative"]),
                "settings": [], "output_selection": "common-prompts-across-outputs",
                "warnings": ["saved-output-ambiguous:only-common-prompts-retained"],
                "output_prompt_alternatives": alternatives}
    reachable = set()

    def visit(key):
        if key in reachable:
            return
        reachable.add(key)
        for value in nodes[key].get("inputs", {}).values():
            if link(value):
                visit(str(value[0]))
    visit(saves[0])
    positive, negative, characters, settings = [], [], [], []
    for key in sorted(reachable):
        node = nodes[key]; inputs = node.get("inputs", {}); kind = node["class_type"]
        if "positive" in inputs and "negative" in inputs:
            positive.append(text(inputs["positive"]))
            negative.append(text(inputs["negative"]))
        elif kind in ("NovelAIGenerator", "NAIImg2ImgNode", "NAIInpaintNode", "NAIFaceDetailerNode"):
            positive.append(text(inputs.get("prompt")))
            negative.append(text(inputs.get("negative_prompt")))
        else:
            continue
        settings.append({"node": key, "type": kind, **{
            k: inputs[k] for k in SETTINGS if type(inputs.get(k)) in (str, int, float, bool)}})
        cp = inputs.get("characterPrompts")
        if link(cp) and nodes[str(cp[0])]["class_type"] == "CharacterPromptSelect":
            ci = nodes[str(cp[0])].get("inputs", {})
            for i in range(1, 9):
                if i > 1 and not ci.get(f"character{i}_enable", False):
                    continue
                value = text(ci.get(f"character{i}"))
                if value:
                    characters.append({"prompt": value, "slot": i})
                negative.append(text(ci.get(f"character{i}_uc")))
    positive = unique(positive)
    if not positive and not characters:
        return None
    return {"format": "comfyui", "positive": positive, "characters": characters,
            "negative": unique(negative), "settings": settings,
            "warnings": sorted("unresolved-text-node:" + k for k in unresolved)}


def extract(info, filename=""):
    """Return selected generation fields, or None when no reliable prompt exists."""
    comment = object_value(info.get("Comment"))
    if comment and (isinstance(comment.get("prompt"), str) or "v4_prompt" in comment):
        base, characters = caption_parts(comment.get("v4_prompt"))
        negative, negchars = caption_parts(comment.get("v4_negative_prompt"))
        positive = base or comment.get("prompt") or info.get("Description", "")
        if not positive and not characters:
            return None
        return {"format": "novelai", "positive": unique([positive]), "characters": characters,
                "negative": unique([negative or comment.get("uc", ""), *(c["prompt"] for c in negchars)]),
                "original_prompt": comment.get("prompt", ""),
                "description": info.get("Description", ""),
                "settings": {k: comment[k] for k in SETTINGS if type(comment.get(k)) in (str, int, float, bool)},
                "software": str(info.get("Software", "")), "model": str(info.get("Source", "")),
                "warnings": []}
    parameters = info.get("parameters")
    if isinstance(parameters, str) and parameters.strip():
        body, _, settings = parameters.rpartition("\nSteps:")
        if not body:
            body, settings = parameters, ""
        positive, _, negative = body.partition("\nNegative prompt:")
        return {"format": "a1111", "positive": unique([positive]), "characters": [],
                "negative": unique([negative]), "settings_text": ("Steps:" + settings) if settings else "",
                "warnings": []}
    graph = object_value(info.get("prompt"))
    return comfy_prompts(graph, filename) if graph else None


def positive_weighted_text(text, suppressed):
    """Keep positive intent; retain nonpositive weighted spans separately.

    NovelAI's bare :: ends numerical emphasis, including an unclosed section
    at end of prompt. Negative values suppress/invert concepts, not affirm them.
    https://docs.novelai.net/en/image/strengthening-weakening/
    """
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    output, start, weight = [], 0, 1.0
    for match in re.finditer(r"(" + number + r")?::", text):
        span = text[start:match.start()]
        if weight > 0:
            output.append(span)
        else:
            suppressed.append({"syntax": "novelai", "weight": weight, "text": span})
            output.append(",")
        weight = float(match.group(1)) if match.group(1) is not None else 1.0
        start = match.end()
    if weight > 0:
        output.append(text[start:])
    else:
        suppressed.append({"syntax": "novelai", "weight": weight, "text": text[start:]})
        output.append(",")
    return "".join(output)


def prompt_tags(prompts, category_hints=None, *, suppressed=None):
    """Strip positive emphasis, preserving names and parenthetical series names.

    Unresolved alternatives/schedules stay in the caption, not recommendation
    tags. This is a lexical parser, not a natural-language captioning model.
    """
    hints = category_hints or {}
    suppressed = suppressed if suppressed is not None else []
    tags, categories, rejected = set(), {}, []
    for prompt in prompts:
        text = re.sub(r"<[^>]*>", "", prompt)
        text = text.replace(r"\(", "\x01").replace(r"\)", "\x02")
        text = positive_weighted_text(text, suppressed)
        # Repeat to handle nested emphasis; plain (series_name) remains intact.
        for _ in range(8):
            def emphasis(match):
                if float(match.group(2)) > 0:
                    return match.group(1)
                suppressed.append({"syntax": "a1111", "weight": float(match.group(2)),
                                   "text": match.group(1)})
                return ","
            updated = re.sub(r"\(([^()]*)\s*:([+-]?(?:\d+(?:\.\d*)?|\.\d+))\)", emphasis, text)
            updated = re.sub(r"\(([^()]*,[^()]*)\)", r"\1", updated)
            if updated == text:
                break
            text = updated
        text = text.replace("{", "").replace("}", "").replace("[", "").replace("]", "")
        for value in re.split(r"[,\n]", text):
            value = value.strip().replace("\x01", "(").replace("\x02", ")")
            if value.startswith("#") or value.upper() in ("", "BREAK", "AND"):
                continue
            kind = None
            match = re.match(r"^(artist|character|copyright):\s*(.+)$", value, re.I)
            if match:
                kind, value = match.group(1).lower(), match.group(2)
            # Unweighted wrappers at a tag boundary, but not suffix qualifiers.
            while value.startswith("(") and value.endswith(")"):
                value = value[1:-1].strip()
            if "|" in value or ":" in value or "__" in value or len(value) > 160:
                rejected.append(value)
                continue
            tag = canonical(value)
            if not tag or re.fullmatch(r"-?\d+(?:\.\d+)?", tag):
                continue
            tags.add(tag)
            kinds = {kind} if kind else set(hints.get(tag, []))
            categories.setdefault(tag, set()).update(kinds)
    return sorted(tags), {k: sorted(v) for k, v in categories.items()}, rejected


def make_record(extracted, image, sha, category_hints=None):
    prompts = extracted["positive"] + [c["prompt"] for c in extracted["characters"]]
    suppressed = []
    tags, categories, rejected = prompt_tags(prompts, category_hints, suppressed=suppressed)
    caption = "\n".join(extracted["positive"])
    for i, char in enumerate(extracted["characters"], 1):
        caption += f"\nCharacter {i}: {char['prompt']}"
    row = {"source": "generated", "id": sha, "image_sha256": sha, "image": image,
           "rating": "u", "split": "train", "tags": tags, "caption": caption.strip(),
           "caption_source": "embedded-generation-positive-prompt",
           "visual_verdict": "U", "generation": extracted,
           "metadata_parser": VERSION, "unparsed_prompt_fragments": rejected,
           "suppressed_prompt_fragments": suppressed}
    for kind in ("artist", "character", "copyright", "general", "meta"):
        row["tags_" + kind] = [t for t in tags if kind in categories.get(t, [])]
    return row


def recommendation_view(entry):
    """Reparse older generated records for voting without changing stored audit.

    Image vectors and original captions are unchanged. Only generated records
    with one explicit embedded-prompt source have enough evidence to reparse.
    Tagger predictions and external Danbooru records retain their existing tags.
    """
    from .tag_policy import filter_tags, typed_tags
    audit = entry.get("audit", [])
    if len(audit) != 1:
        return entry
    source = audit[0].get("source_record", {})
    if (source.get("source") != "generated"
            or source.get("caption_source") != "embedded-generation-positive-prompt"
            or source.get("metadata_parser") == VERSION):
        return entry
    parsed = make_record(source["generation"], source["image"], source["image_sha256"],
                         entry["categories"])
    kept, _ = filter_tags(parsed)
    typed = typed_tags(parsed)
    return dict(entry, document=dict(entry["document"], tags=kept),
                categories={tag: sorted(k for k, names in typed.items() if tag in names)
                            or ["unknown"] for tag in kept})
