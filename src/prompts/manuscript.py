"""Bind the manuscript's named input slots to each runtime stage."""

import json
from pathlib import Path
import re


TEMPLATES = json.loads(
    Path(__file__).with_name("templates.json").read_text(encoding="utf-8")
)
TOKENS = re.compile(r"\{[A-Z][A-Z_0-9]*\}|[{}]")


def bind_template(template_id, **fields):
    """Produce a str.format template while preserving literal JSON in the paper."""
    text = TEMPLATES[template_id]["text"]
    required = set(re.findall(r"\{([A-Z][A-Z_0-9]*)\}", text))
    if fields.keys() != required:
        raise ValueError(
            f"{template_id}: inconsistent input slots {fields.keys() ^ required}"
        )

    def replace(match):
        token = match.group()
        if token in ("{", "}"):
            return token * 2
        return fields[token[1:-1]]

    return TOKENS.sub(replace, text)
