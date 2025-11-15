import logging
import re
import wptools

logger = logging.getLogger(__name__)

# simple helpers
_RE_HTML = re.compile(r"<.*?>")
_RE_TEMPLATES = re.compile(re.escape("{{") + r".*?" + re.escape("}}"))
_RE_LIST_ITEM = re.compile(r"^[*-]\s*", re.M)
_RE_FILE_PREFIX = re.compile(r"^(?:file:|image:)", re.IGNORECASE)
_RE_MULTI_SP = re.compile(r"\s{2,}")

# fields to drop by exact key or by substring
_DROP_KEYS = {
    "image",
    "image_size",
    "imagesize",
    "image_upright",
    "ground_image",
    "logosize",
    "logo",
    "flag",
    "caption",
    "map",
    "map_caption",
    "mapframe-zoom",
    "mapframe",
    "maplocator",
    "map_type",
    "map_size",
    "website",
    "official website",
    "official_website",
    "prev",
    "next",
    "last",
    "first",
    "series",
    "season",
    "competition",
    "timezone",
    "tz",
}
_DROP_SUBSTRINGS = ("size", "upright", "px", "bgcolor", "width", "height")


def _strip_wikicode(val: str) -> str:
    if not val:
        return ""
    v = str(val)
    v = _RE_HTML.sub("", v)
    v = _RE_TEMPLATES.sub("", v)

    start = 0
    while True:
        i = v.find("[[", start)
        if i == -1:
            break
        j = v.find("]]", i + 2)
        if j == -1:
            break
        inner = v[i + 2 : j]
        replacement = inner.rsplit("|", 1)[-1].strip()
        v = v[:i] + replacement + v[j + 2 :]
        start = i + len(replacement)

    v = v.replace("[[", "").replace("]]", "")
    v = re.sub(r"(?i)(file:|image:)", "", v)
    v = re.sub(r"\|\s*\d+px", "", v)
    v = v.replace("|", ",")
    v = _RE_LIST_ITEM.sub("", v)
    v = re.sub(r",\s*,+", ",", v)
    v = v.replace("\n", " ").strip()
    v = re.sub(r"\s*,\s*", ", ", v)
    v = _RE_MULTI_SP.sub(" ", v)
    v = v.strip(" ,;:-")
    return v


def _normalize_key(key: str) -> str:
    """Normalize key format without changing semantic meaning."""
    k = key.lower().strip()
    k = k.replace(" ", "_")
    k = re.sub(r"[^\w\s_-]", "", k)
    k = re.sub(r"_+", "_", k)
    k = k.strip("_")
    return k


def _normalize_value(key: str, value: str) -> str:
    """Normalize values based on detected patterns."""
    k = key.lower()
    v = value.strip()

    if re.match(r"^\d+px$", v):
        return ""

    if (
        v.startswith("{{flatlist")
        or v.startswith("{{Flatlist")
        or v.startswith("{{hlist")
    ):
        return ""

    if any(
        word in k
        for word in [
            "date",
            "time",
            "birth",
            "death",
            "formed",
            "dissipated",
            "established",
            "founded",
            "dissolved",
            "opened",
            "closed",
        ]
    ):
        v = re.sub(r"\s*\(UTC[^)]*\)", "", v)
        if re.search(r"\d{4}", v):
            v = re.sub(
                r",?\s*\d{1,2}:\d{2}\s*(?:am|pm|a\.m\.|p\.m\.)?", "", v, flags=re.I
            )
        v = v.strip(", ")

    if "timezone" in k or k == "tz":
        return ""

    v = re.sub(r"\s*,\s*,+", ",", v)

    if v.lower().startswith("see ") or v.lower().startswith("main article:"):
        return ""

    return v


def clean_infobox(raw: dict[str, str], keep_only: dict[str, str] = None) -> dict:
    """
    Clean and normalize a raw infobox dict from wptools.

    Args:
        raw: raw dict from wptools page.data['infobox']
        keep_only: optional whitelist of keys to keep (case-insensitive)

    Returns:
        Cleaned dictionary with normalized keys and values
    """
    if not raw:
        return {}

    whitelist = None
    if keep_only:
        whitelist = {k.lower() for k in keep_only}

    cleaned = {}
    for k_raw, v_raw in raw.items():
        if not k_raw:
            continue
        k = str(k_raw).strip().lower()

        if whitelist is not None and k not in whitelist:
            continue

        if k in _DROP_KEYS or any(sub in k for sub in _DROP_SUBSTRINGS):
            continue

        v = str(v_raw).strip()
        if not v:
            continue
        if v.lower() in {"yes", "no"}:
            continue

        v = _strip_wikicode(v)
        if not v:
            continue

        v = _normalize_value(k, v)
        if not v:
            continue

        k_normalized = _normalize_key(k)

        if any(
            substr in k_normalized
            for substr in ("mapframe", "image", "size", "upright")
        ):
            continue

        v = re.sub(r"\s+", " ", v).strip()
        v = v.strip(" ,;:-")

        if v:
            cleaned[k_normalized] = v

    return cleaned


class InfoboxExtractor:

    def extract_infobox(self, topic: str) -> dict[str, str]:
        """
        Extract cleaned infobox data for a given topic.

        Args:
            topic: Wikipedia page title

        Returns:
            Cleaned infobox dictionary
        """
        try:
            # Suppress wptools verbose output
            page = wptools.page(topic, silent=True).get_parse(show=False)
            infobox = page.data.get("infobox", {})
        except Exception as e:
            logger.warning(f"Failed to fetch infobox for {topic}: {e}")
            return {}

        return clean_infobox(infobox)
