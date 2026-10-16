"""Native context expansion with independent tags and prose STRING ports."""
import json

try:
    from .gemma_tipo_native_node import GemmaTipoNative, validate_input, validate_repetition_penalty, generate_text, clean_tags, apply_tipo_lora, REQUIREMENTS, resolve_clip
    from .prompt_expansion import VERSION, OUTPUT_FORMATS, assemble_instruction, clean_natural_language
    from .tag_knowledge import load_knowledge, append_vocabulary
    from .vrs_prompt_reference import load_reference, prepare_reference, append_reference, external_fingerprint
except ImportError:
    from gemma_tipo_native_node import GemmaTipoNative, validate_input, validate_repetition_penalty, generate_text, clean_tags, apply_tipo_lora, REQUIREMENTS, resolve_clip
    from prompt_expansion import VERSION, OUTPUT_FORMATS, assemble_instruction, clean_natural_language
    from tag_knowledge import load_knowledge, append_vocabulary
    from vrs_prompt_reference import load_reference, prepare_reference, append_reference, external_fingerprint


class GemmaPromptExpandNative:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = GemmaTipoNative.INPUT_TYPES()
        # Append the new widget after this node's existing optional widgets so
        # serialized workflows keep their positional values.
        penalty_input = inputs["optional"].pop("repetition_penalty")
        inputs["required"]["output_format"] = (list(OUTPUT_FORMATS), {
            "default": "both", "tooltip": "both: expand a scene in prose, then render that same scene as tags (two sequential calls)."})
        inputs["required"]["tag_length"][1]["tooltip"] = "Detail target for tags and prose; not a guaranteed length."
        inputs["required"]["max_tokens"][1].update(default=512, tooltip="Output token limit per call; both uses two calls.")
        inputs["required"]["ban_tags"][1]["tooltip"] = (
            "English elements to exclude. Exact tag filtering applies only to tags; prose follows the model instruction.")
        inputs["optional"]["tag_set"] = (["danbooru", "e621"], {"default": "danbooru"})
        inputs["optional"]["tag_knowledge_path"] = ("STRING", {"default": "", "tooltip":
            "Optional local rs-tag-knowledge/1 JSONL or legacy Danbooru Korean CSV. No automatic download. Leave blank to use model knowledge only."})
        inputs["optional"]["vrs_state_path"] = ("STRING", {"default": "", "tooltip":
            "Optional local VRS generation-state /1 or /2 JSON, exported for this exact prompt. Danbooru only. Saved references, not automatic image search."})
        inputs["optional"]["vrs_state"] = ("VRS_STATE", {"tooltip": "Connect VRS Native Image Recall. Uses this request and, if supplied, the same IMAGE."})
        inputs["optional"]["tipo_lora_scope"] = (["both", "tags_only"], {"default": "both", "tooltip":
            "both applies the selected adapter to both calls. tags_only leaves prose on the supplied base CLIP and applies the adapter only to tags; use an unpatched base CLIP."})
        penalty_input[1]["tooltip"] = (
            "Tag call only; prose keeps 1.05. Current ComfyUI ignores penalties at temperature=0; use a positive temperature. Larger values can also omit useful tags or names. Existing workflows retain 1.05.")
        inputs["optional"]["repetition_penalty"] = penalty_input
        return inputs

    @classmethod
    def IS_CHANGED(cls, vrs_state_path="", tag_knowledge_path="", **kwargs):
        return external_fingerprint(vrs_state_path, tag_knowledge_path)

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("tags", "natural_language", "raw_output")
    FUNCTION = "run"
    CATEGORY = "prompt"
    DESCRIPTION = (
        "Expand text/image context into Danbooru tags and/or an English image-generation instruction. "
        "both creates prose first, then tags from that scene using the same Gemma 4 CLIP. "
        "Accept a connected VRS native retrieval state or a saved state, plus a local tag dictionary. "
        "The VRS Recall node performs image lookup; no training is required. " + REQUIREMENTS
    )

    def run(self, clip=None, prompt="", tag_length="long", max_tokens=512, temperature=0.5, ban_tags="", seed=0,
            output_format="both", image=None, tag_set="danbooru", tag_knowledge_path="", vrs_state_path="", vrs_state=None, tipo_lora="none",
            model_source="connected_clip", gguf_path="", mmproj_path="", tokenizer_path="", tipo_lora_scope="both",
            repetition_penalty=1.05):
        validate_repetition_penalty(repetition_penalty)
        if output_format not in OUTPUT_FORMATS:
            raise ValueError("Unknown output_format")
        if tipo_lora_scope not in ("both", "tags_only"):
            raise ValueError("Unknown tipo_lora_scope")
        if not validate_input(prompt, tag_length, image):
            return ("", "", "")
        if tag_set not in ("danbooru", "e621"):
            raise ValueError("Unknown tag_set")
        if vrs_state is not None and vrs_state_path.strip():
            raise ValueError("Use either connected vrs_state or vrs_state_path, not both")
        reference = (prepare_reference(vrs_state, prompt, ban_tags, tag_set) if vrs_state is not None
                     else load_reference(vrs_state_path, prompt, ban_tags, tag_set) if vrs_state_path.strip() else None)
        if vrs_state is not None and "native_query" in vrs_state:
            if image is not None:
                try:
                    from .vrs_int.native import pixel_digest
                except ImportError:
                    from vrs_int.native import pixel_digest
                if pixel_digest(image) != vrs_state["native_query"]["pixel_sha256"]:
                    raise ValueError("VRS state and Gemma IMAGE differ; connect the same image to both nodes")
            reference["query_image_binding"] = ("Native RGB pixel hash verified against Gemma IMAGE" if image is not None
                                                 else "Native retrieval state; no separate Gemma IMAGE supplied")
            reference["native_query"] = dict(vrs_state["native_query"])
        effective_bans = ban_tags
        if reference:
            effective_bans = ", ".join(t for t in [ban_tags, *reference["packet"]["controls"]["exclude"]] if t.strip())
        knowledge = load_knowledge(tag_knowledge_path) if tag_knowledge_path.strip() else None
        if knowledge and tag_set not in knowledge.provenance["sources"]:
            raise ValueError("Tag knowledge has no entries for the selected tag_set")
        vocab = knowledge.vocabulary(prompt, tag_set) if knowledge else None
        clip = resolve_clip(clip, model_source, image, gguf_path, mmproj_path, tokenizer_path)
        # Keep the base handle: ComfyUI switches shared weights through each
        # CLIP's patcher. Do not attach a tag-only adapter to the prose call.
        needs_adapter = tipo_lora_scope == "both" or output_format in ("both", "tags")
        adapted_clip = apply_tipo_lora(clip, tipo_lora) if needs_adapter else clip
        prose_clip = adapted_clip if tipo_lora_scope == "both" else clip
        scene_vocab = None
        def instruction(kind, scene=None):
            nonlocal scene_vocab
            if knowledge and scene:
                # Generated prose may contain ordinary words that also alias
                # artists. Only the user's request can introduce named identities.
                scene_vocab = knowledge.vocabulary(scene, tag_set, include_identities=False)
            return assemble_instruction(
                {"prompt": prompt, "tag_length": tag_length, "ban_tags": ban_tags,
                 "image": image if scene is None else None, "output_format": kind,
                 "scene": scene, "tag_set": tag_set},
                reference=reference, vocabulary=vocab, scene_vocabulary=scene_vocab)
        tags = natural = ""
        raw_tags = raw_natural = None
        if output_format in ("both", "natural_language"):
            raw_natural = generate_text(prose_clip, instruction("natural_language"),
                image, max_tokens, temperature, seed)
            natural = clean_natural_language(raw_natural)
        if output_format in ("both", "tags"):
            # The second pass renders the chosen scene, rather than encoding the
            # image again and independently inventing a second scene.
            tag_image = image if output_format == "tags" else None
            raw_tags = generate_text(adapted_clip, instruction("tags", scene=natural if output_format == "both" else None),
                tag_image, max_tokens, temperature, seed, repetition_penalty=repetition_penalty)
            tags = clean_tags(raw_tags, effective_bans, prompt)
        raw = json.dumps({"prompt_contract": VERSION, "output_format": output_format,
                          "tag_set": tag_set, "natural_language": raw_natural, "tags": raw_tags,
                          "generation": {"temperature": temperature, "seed": seed, "max_tokens": max_tokens,
                                         "repetition_penalty": {"natural_language": 1.05 if raw_natural is not None else None,
                                                                "tags": repetition_penalty if raw_tags is not None else None}},
                          "tipo_lora": {"selected": tipo_lora, "scope": tipo_lora_scope,
                                        "natural_language": (tipo_lora if tipo_lora_scope == "both" else "none") if raw_natural is not None else None,
                                        "tags": tipo_lora if raw_tags is not None else None},
                          "knowledge": knowledge.provenance if knowledge else None,
                          "request_vocabulary": vocab,
                          "scene_vocabulary": scene_vocab,
                          "vrs_reference": reference,
                          "tag_membership": knowledge.check_tags(tags, tag_set) if knowledge and raw_tags else None}, ensure_ascii=False)
        return (tags, natural, raw)
