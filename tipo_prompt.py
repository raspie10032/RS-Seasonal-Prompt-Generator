"""Versioned TIPO input shared by native inference and future supervision.

The native runtime supplies Gemma's turn/media tokens. Keep this module free of
model dependencies; a text tokenizer must never silently train an image row.
"""

VERSION = "gemma4-tipo-native/1"


def _engine():
    try:
        from . import tipo_engine
    except ImportError:
        import tipo_engine
    return tipo_engine


def build_user_prompt(prompt, tag_length="long", ban_tags="", has_image=False):
    """Preserve the original native node's instruction text byte for byte."""
    te = _engine()
    count = te._TARGET_TAGS[tag_length]
    instruction = (
        te.SYSTEM_TIPO + " Preserve explicitly requested artist, character and series names. "
        "Do not invent names. Follow the user's changes and exclusions. "
        "Translate Korean descriptions into English tags. "
        f"Aim for approximately {count} tags."
    )
    if has_image:
        instruction += (
            " Use the reference image for visual details and expand into a new image prompt. "
            "The user's requested changes take precedence over details in the image."
        )
    return instruction + te._avoid_hint(ban_tags) + "\n\nUser request:\n" + prompt.strip()


def text_messages(inputs, answer=None):
    """One user turn, as in ComfyUI; labels live only in the assistant turn."""
    if inputs.get("image"):
        raise ValueError("Image supervision requires a multimodal processor; refusing text-only encoding")
    messages = [{"role": "user", "content": build_user_prompt(
        inputs["prompt"], inputs.get("tag_length", "long"), inputs.get("ban_tags", ""), False)}]
    if answer is not None:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Missing assistant answer")
        messages.append({"role": "assistant", "content": answer})
    return messages


def row_messages(row):
    """Versioned rows are rebuilt, so stale or hand-edited templates fail closed."""
    if row.get("prompt_contract") is None:
        return row["messages"]  # Historical pilot retains its historical framing.
    try:
        from .prompt_expansion import VERSION as CONTEXT_VERSION, context_messages
    except ImportError:
        from prompt_expansion import VERSION as CONTEXT_VERSION, context_messages
    if row["prompt_contract"] == CONTEXT_VERSION:
        return context_messages(row)
    if row["prompt_contract"] != VERSION:
        raise ValueError("Unknown TIPO prompt contract")
    if row.get("purpose") == "evaluation":
        raise ValueError("Evaluation-only rows cannot supply training targets")
    messages = text_messages(row["input"], row["answer"])
    if "messages" in row and row["messages"] != messages:
        raise ValueError("Stored messages differ from the TIPO prompt contract")
    return messages


def instruction_messages(row):
    try:
        from .prompt_expansion import VERSION as CONTEXT_VERSION, context_messages
    except ImportError:
        from prompt_expansion import VERSION as CONTEXT_VERSION, context_messages
    if row.get("prompt_contract") == CONTEXT_VERSION:
        return context_messages(row, include_answer=False)
    messages = row_messages(row)
    return messages[:-1] if messages[-1]["role"] == "assistant" else messages
