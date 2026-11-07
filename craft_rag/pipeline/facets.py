"""Controlled-vocabulary facets for retrieval filters (finding H3).

The prepared data stores free text in ``craft_cluster`` and ``why_theme`` (64 and 22 distinct
strings respectively), so exact string matching starves filters. These helpers map the raw
values onto small, fixed vocabularies. Filters are normalised on BOTH sides (the chunk and the
requested value), so legacy values such as ``"Bamboo (Battal chisel handles ...)"`` still work.
"""
from __future__ import annotations

import re
from typing import Any

CRAFT_VALUES = ("wood", "bamboo", "both")
THEME_IDS = tuple(str(number) for number in range(1, 13))
OTHER = "other"

# Filter keys that use a controlled vocabulary, mapped to their facet name.
FACET_KEYS = {
    "craft": "craft",
    "craft_cluster": "craft",
    "why_theme": "theme",
    "theme": "theme",
}


def normalize_craft(raw: Any) -> str:
    """Map free text such as ``"Wood (walnut ...)"`` to ``wood`` / ``bamboo`` / ``both`` / ``other``."""
    text = str(raw or "").strip().casefold()
    if not text:
        return ""
    if text == "both" or text.startswith("both"):
        return "both"
    has_wood, has_bamboo = "wood" in text, "bamboo" in text
    first = re.match(r"[a-z]+", text)
    lead = first.group(0) if first else ""
    if lead in ("wood", "bamboo"):
        # "Bamboo/Wood" and "Wood/Bamboo" name both crafts; a parenthetical remark does not.
        head = re.split(r"[(\u2014-]", text, maxsplit=1)[0]
        if "wood" in head and "bamboo" in head:
            return "both"
        return lead
    if has_wood and has_bamboo:
        return "both"
    if has_wood:
        return "wood"
    if has_bamboo:
        return "bamboo"
    return OTHER


def normalize_theme(raw: Any) -> str:
    """Return the theme number ``"1"``..``"12"`` from values like ``"7. Posture, Grip ..."`` or ``"12"``."""
    text = str(raw or "").strip()
    if not text:
        return ""
    match = re.match(r"(\d{1,2})\b", text)
    if match and match.group(1) in THEME_IDS:
        return match.group(1)
    return OTHER


def _raw_value(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value in (None, ""):
        value = (row.get("metadata") or {}).get(key)
    return value


def facet_value(row: dict[str, Any], key: str) -> str:
    """Normalised value of filter ``key`` for one chunk row (``""`` when the chunk has none)."""
    facet = FACET_KEYS.get(key)
    if facet == "craft":
        raw = _raw_value(row, "craft_cluster") or _raw_value(row, "craft")
        return normalize_craft(raw)
    if facet == "theme":
        return normalize_theme(_raw_value(row, "why_theme") or _raw_value(row, "theme"))
    return str(_raw_value(row, key) or "").strip().casefold()


def normalize_filter_value(key: str, value: Any) -> str:
    """Normalise a requested filter value the same way chunk values are normalised."""
    facet = FACET_KEYS.get(key)
    if facet == "craft":
        return normalize_craft(value)
    if facet == "theme":
        return normalize_theme(value)
    return str(value or "").strip().casefold()


def facet_options(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Controlled dropdown values actually present in ``rows`` (fixed order, no free text)."""
    crafts = {facet_value(row, "craft") for row in rows}
    themes = {facet_value(row, "why_theme") for row in rows}
    return {
        "craft_clusters": [value for value in CRAFT_VALUES if value in crafts],
        "why_themes": [value for value in THEME_IDS if value in themes],
    }
