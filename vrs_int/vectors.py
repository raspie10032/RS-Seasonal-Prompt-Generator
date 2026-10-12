"""Explicit vector representations; only f32le is lossless."""
import base64


def encode_vector(vector, codec="f32le"):
    import numpy as np
    values = np.asarray(vector, dtype="<f4").reshape(-1)
    if not values.size or not np.isfinite(values).all() or np.linalg.norm(values) <= 0:
        raise ValueError("Invalid vector")
    if codec == "f32le":
        result = {"codec": codec, "data": base64.b64encode(values.tobytes()).decode("ascii")}
    elif codec == "i8":
        scale = float(np.max(np.abs(values)) / 127)
        codes = np.clip(np.rint(values / scale), -127, 127).astype("i1")
        result = {"codec": codec, "scale": scale,
                  "data": base64.b64encode(codes.tobytes()).decode("ascii")}
    else:
        raise ValueError("Unsupported vector codec")
    return result


def decode_vector(value, dimension):
    import numpy as np
    raw = base64.b64decode(value["data"], validate=True)
    codec = value["codec"]
    if codec == "f32le" and len(raw) == dimension * 4:
        result = np.frombuffer(raw, dtype="<f4")
    elif codec == "i8" and len(raw) == dimension:
        scale = value.get("scale", 0)
        if not isinstance(scale, (float, int)) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid vector scale")
        result = np.frombuffer(raw, dtype="i1").astype("<f4") * scale
    else:
        raise ValueError("Invalid vector codec or byte length")
    if not np.isfinite(result).all() or np.linalg.norm(result) <= 0:
        raise ValueError("Invalid vector values")
    return result
