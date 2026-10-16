"""Context expansion and two output renderings; no model or data dependencies.

Separate from gemma4-tipo-native/1: existing training rows keep their exact
template. Inference and context supervision share the same input assembly.
"""
import re

try:
    from .tipo_prompt import _engine
except ImportError:
    from tipo_prompt import _engine

VERSION = "gemma4-context-expansion/1"
OUTPUT_FORMATS = ("both", "tags", "natural_language")
SENTENCES = {"very_short": "2", "short": "3 to 4", "long": "5 to 6", "very_long": "7 to 9"}

def build_expansion_prompt(prompt, tag_length, ban_tags, has_image, output_format, scene=None, tag_set="danbooru"):
    if tag_set not in ("danbooru", "e621"):
        raise ValueError("Unknown tag_set")
    if output_format not in ("tags", "natural_language"):
        raise ValueError("Choose one rendering per model call")
    count = _engine()._TARGET_TAGS[tag_length]
    instruction = (
        "Write a ready-to-use image-generation prompt from the user's request and any reference. "
        "Understand the scene, subjects, actions and spatial relationships before writing. "
        "Preserve what the user asks to keep, apply requested changes, and omit excluded elements. "
        "Keep explicitly requested artist, character and series names exactly, including parentheses; "
        "do not guess or invent identities from appearance. "
        "Treat supplied reference tags as material to select from, not a list to copy in full. "
        "Default to 2D anime illustration; keep 3D game graphics when requested or referenced. "
        "Do not add photographic realism, quality slogans or year tags. "
    )
    if scene is None:
        instruction += (
            "Expand underspecified parts with compatible actions, setting, lighting and composition. "
            "Keep the result one coherent scene, without conflicting poses or repeated descriptions. "
        )
        if has_image:
            instruction += (
                "Use the image's visible features as creative reference, not as a rigid caption. "
                "The user's changes override the original image. Omit uncertain visual details. "
            )
    elif output_format != "tags" or not scene.strip():
        raise ValueError("A nonempty expanded scene can only be rendered as tags")
    else:
        instruction += (
            "The expanded scene below is the scene to render. Convert it to tags without creating "
            "another scene, adding new props, or reintroducing the original setting. "
            "The user's explicit constraints take precedence if the scene contains a mistake. "
        )
    if ban_tags.strip():
        instruction += "Exclude these elements from the final scene: " + ban_tags.strip() + ". "
    if output_format == "natural_language":
        instruction += (
            f"Output only an English image-generation instruction in {SENTENCES[tag_length]} complete sentences. "
            "Describe the final scene directly, starting with 'Create'. "
            "Use natural prose with clear subjects, actions and relationships, not comma-separated tags. "
            "Do not include analysis, headings, lists, or an explanation of your changes."
        )
    else:
        instruction += (
            f"Output ONLY comma-separated established {tag_set} tags in English. Use canonical tag names, "
            "not sentences or ad-hoc phrases made by copying chunks of the prose. Map actions and objects "
            "to separate standard tags when appropriate. Preserve who does what, "
            "object relations, and the scope of scene details as far as tags allow. "
            "A negated object must not become a positive tag. Do not split a name at its parentheses. "
            f"Aim for at most about {count} useful tags; do not add details merely to reach a count. "
            "No prose, headings or duplicate tags."
        )
        if tag_set == "danbooru":
            instruction += (
                " Formatting examples only (do not copy absent details): 'a woman holds a teacup' "
                "becomes '1girl, holding cup, teacup'; 'blue eyes' stays 'blue eyes'. "
                "If no standard tag expresses a lighting nuance, use the closest grounded standard tags "
                "and retain the nuance in the prose output; do not invent a new tag."
            )
    text = instruction + "\n\nUser request:\n" + prompt.strip()
    if scene is not None:
        text += "\n\nExpanded scene to render:\n" + scene.strip()
    return text


def clean_natural_language(raw):
    """Remove wrapper/reasoning only; never apply comma-based tag filtering."""
    text = _engine()._strip_reasoning(raw).strip()
    text = re.sub(r"^```[^\n]*\n|\n?```\s*$", "", text).strip()
    if not text:
        raise RuntimeError("Gemma returned no natural-language instruction")
    return text


def assemble_instruction(inputs, *, reference=None, vocabulary=None, scene_vocabulary=None):
    """Shared native/training framing; all supplemental packets are input data.

    The scene-to-tags pass never receives pixels, even when the original request
    did. A frozen training row stores the actual scene in ``input.scene``.
    """
    try:
        from .tag_knowledge import append_vocabulary
        from .vrs_prompt_reference import append_reference
    except ImportError:
        from tag_knowledge import append_vocabulary
        from vrs_prompt_reference import append_reference
    scene = inputs.get("scene")
    if scene is not None and inputs.get("image") is not None:
        raise ValueError("Scene rendering must not receive an image")
    bans = inputs.get("ban_tags", "")
    if reference:
        bans = ", ".join(t for t in [bans, *reference["packet"]["controls"]["exclude"]] if t.strip())
    text = build_expansion_prompt(
        inputs["prompt"], inputs.get("tag_length", "short"), bans,
        inputs.get("image") is not None, inputs["output_format"], scene, inputs.get("tag_set", "danbooru"))
    for packet in (vocabulary, scene_vocabulary):
        if packet:
            text = append_vocabulary(text, packet)
    return append_reference(text, reference, render_scene=scene is not None)


def context_instruction(row):
    """Rebuild a frozen training request with the native reference sanitization."""
    try:
        from .vrs_prompt_reference import prepare_reference
    except ImportError:
        from vrs_prompt_reference import prepare_reference
    inputs = row["input"]
    state = inputs.get("vrs_state")
    reference = prepare_reference(state, inputs["prompt"], inputs.get("ban_tags", ""),
                                  inputs.get("tag_set", "danbooru")) if state is not None else None
    return assemble_instruction(inputs, reference=reference, vocabulary=inputs.get("vocabulary"),
                                scene_vocabulary=inputs.get("scene_vocabulary"))


def context_messages(row, *, include_answer=True):
    if row.get("prompt_contract") != VERSION:
        raise ValueError("Unknown context prompt contract")
    if row["input"].get("image"):
        raise ValueError("Image supervision requires a multimodal processor")
    messages = [{"role": "user", "content": context_instruction(row)}]
    if include_answer:
        if row.get("purpose") != "training" or not str(row.get("answer", "")).strip():
            raise ValueError("Only training rows may provide assistant targets")
        messages.append({"role": "assistant", "content": row["answer"]})
    if "messages" in row:
        raise ValueError("Rebuild context messages; stored templates are not supported")
    return messages
