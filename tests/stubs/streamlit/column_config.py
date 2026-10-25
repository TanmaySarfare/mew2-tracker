PRESETS = {"plain", "localized", "percent", "dollar", "euro", "yen", "accounting", "compact", "scientific",
           "engineering", "bytes"}


def NumberColumn(label=None, *, width=None, help=None, disabled=None, required=None, pinned=None, alignment=None,
                 default=None, format=None, min_value=None, max_value=None, step=None):
    if format is not None and format not in PRESETS:
        format % 1.5  # printf-style must be valid (sprintf-js has no thousands flag either)
    if step is not None and not isinstance(step, (int, float)):
        raise TypeError("step must be a number")
    return {"type": "number", "format": format, "step": step}


def TextColumn(label=None, *, width=None, help=None, disabled=None, required=None, pinned=None, alignment=None,
               default=None, max_chars=None, validate=None):
    return {"type": "text"}
