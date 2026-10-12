"""Stable, separate identities for collected posts and personal generations."""
import re


def observation_key(row):
    source = row.get("source", "danbooru")
    value = row.get("id")
    if source == "danbooru":
        if type(value) is not int or value <= 0:
            raise ValueError("positive integer Danbooru id required")
    elif source == "generated":
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("generated id must be the image SHA-256")
        if row.get("image_sha256") != value:
            raise ValueError("generated id and image_sha256 must match")
    else:
        raise ValueError("source must be danbooru or generated")
    ratings = ("g", "s", "q", "e", "u") if source == "generated" else ("g", "s", "q", "e")
    if row.get("rating") not in ratings:
        raise ValueError("explicit g/s/q/e rating required; generated images also allow u (unknown)")
    return f"{source}:{value}"
