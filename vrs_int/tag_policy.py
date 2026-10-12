"""Conservative, auditable tag filtering; never use legacy caption drop lists.

Artist/character/copyright labels are core data, including singleton labels.
Category hints come only from explicit source fields, never from name heuristics.
"""
import re

def canonical(tag):
    return " ".join(tag.lower().replace("_", " ").split())

IDENTITY = ("artist", "character", "copyright")
FIELDS = {"artist": "tags_artist", "character": "tags_character",
          "copyright": "tags_copyright", "general": "tags_general", "meta": "tags_meta"}
POLICY_VERSION = "rs-core-identities-v1"
# Query-time policy is separate from the immutable ingestion contract. Existing
# snapshots and resumable builds need no migration or image re-encoding.
RECOMMENDATION_POLICY_VERSION = "creative-reference-quality-year-v1"
RECOMMENDATION_QUALITY = {
    "amazing quality", "best quality", "great quality", "good quality",
    "high quality", "normal quality", "low quality", "worst quality", "bad quality",
    "masterpiece", "aesthetic", "very aesthetic", "highly aesthetic",
    "highres", "absurdres", "incredibly absurdres", "ultra highres", "lowres",
    "4k", "8k", "16k", "uhd", "newest",
}
RECOMMENDATION_SCORE = re.compile(r"score \d+(?: up)?")
RECOMMENDATION_YEAR = re.compile(r"(?:year\s*:?\s*)?[12]\d{3}")
NOISE = {
    "highres", "absurdres", "incredibly absurdres", "lowres", "jpeg artifacts",
    "compression artifacts", "bad quality", "worst quality", "low quality",
    "normal quality", "best quality", "masterpiece", "score 9", "score 8 up",
    "score 7 up", "watermark", "signature", "artist name", "twitter username",
    "patreon username", "weibo username", "pixiv username", "web address",
    "commentary", "commentary request", "translation request", "translated",
    "partially translated", "check translation", "annotation request",
    "character request", "copyright request", "artist request", "source request",
    "fan request", "check artist", "check character", "check copyright",
    "check commentary", "commission", "skeb commission", "pixiv commission",
    "ych commission", "vgen commission", "paid reward available", "variant set",
    "large variant set", "revision", "has bad revision", "duplicate",
    "pixel-perfect duplicate", "md5 mismatch", "resolution mismatch",
    "source smaller", "source larger", "self-upload", "commissioner upload",
    "second-party source", "third-party source", "non-web source", "protected link",
    "inactive account", "alt text", "resized", "upscaled", "sample", "thumbnail",
    "lossless-lossy", "corrupted file", "corrupted twitter file",
    "making-of available", "psd available", "merchandise available",
    "voice drama available", "self-datamine", "imageboard first-party source",
}
NOISE_PATTERNS = [re.compile(p) for p in (
    r".+ commentary", r"bad (?:.+ )?(?:id|link|source)",
    r"(?:patreon|twitter|weibo|pixiv|fanbox) (?:logo|url|id)",
    r"score \d+(?: up)?", r"rating:[gsqe]", r"(?:width|height|score|id):\d+",
)]


def typed_tags(row):
    result = {}
    for category, field in FIELDS.items():
        value = row.get(field, [])
        values = value.split() if isinstance(value, str) else value
        result[category] = {canonical(t) for t in values if t.strip()}
    return result


def raw_tags(row):
    value = row.get("tags", [])
    if isinstance(value, str):
        # The two local source datasets use different serializations.
        value = value.split() if any(k in row for k in FIELDS.values()) else value.split(",")
    result = {canonical(t) for t in value if t.strip()}
    for values in typed_tags(row).values():
        result.update(values)
    return result


def filter_tags(row, category_hints=None):
    hints = category_hints or {}
    typed = typed_tags(row)
    protected = set().union(*(typed[c] for c in IDENTITY))
    kept, removed = [], {}
    for tag in sorted(raw_tags(row)):
        core = tag in protected or set(hints.get(tag, ())).intersection(IDENTITY)
        if not core and (tag in NOISE or any(p.fullmatch(tag) for p in NOISE_PATTERNS)):
            removed[tag] = "non-scene metadata, quality, watermark or signature"
        else:
            kept.append(tag)
    return kept, removed


def filter_recommendation_tags(tags, categories):
    """Filter a ranked list before truncation, preserving explicit identities.

    Never mutate source captions, stored tags, vectors, or category dictionaries.
    Exact matches avoid deleting scene phrases such as 'new year', 'old man',
    '1980s (style)', or lighting/detail descriptors. Unknown names remain unknown.
    """
    kept, removed = [], {}
    for tag in tags:
        if set(categories.get(tag, ())).intersection(IDENTITY):
            kept.append(tag)
            continue
        normalized = canonical(tag)
        if normalized in RECOMMENDATION_QUALITY or RECOMMENDATION_SCORE.fullmatch(normalized):
            removed[tag] = "quality or recency boilerplate"
        elif RECOMMENDATION_YEAR.fullmatch(normalized):
            removed[tag] = "year metadata"
        else:
            kept.append(tag)
    return kept, removed
