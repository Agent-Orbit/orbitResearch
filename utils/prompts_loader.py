import json
from pathlib import Path


_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "prompts.json"
_cache: dict | None = None


def _load() -> dict:
    """
    Load prompts.json once and cache it in memory.
    Call reload() if you want to pick up edits without restarting.
    """

    global _cache

    if _cache is None:

        if not _PROMPTS_PATH.exists():

            raise FileNotFoundError(f"prompts.json not found at {_PROMPTS_PATH}")

        with open(_PROMPTS_PATH, "r", encoding="utf-8") as f:

            try:
                data = json.load(f)

            except json.JSONDecodeError as e:

                raise ValueError(f"prompts.json is not valid JSON: {e}") from e

        _cache = {k: v for k, v in data.items() if not k.startswith("_")}

    return _cache


def reload() -> None:
    """Force a fresh read from disk - useful during development."""

    global _cache
    _cache = None
    _load()


def get(key: str, **kwargs) -> str:
    """
    Fetch a prompt template by key and interpolate variables.

    Usage:
        get("retry", query="LangChain agents", attempt=2, revised_query="LangChain ReAct")

    Raises:
        KeyError   - if the prompt key doesn't exist in prompts.json
        ValueError - if a required variable is missing from kwargs
    """

    prompts = _load()

    if key not in prompts:

        available = ", ".join(prompts.keys())
        raise KeyError(f"Prompt '{key}' not found. Available keys: {available}")

    entry    = prompts[key]
    template = entry["template"]
    required = entry.get("variables", [])

    missing = [v for v in required if v not in kwargs]

    if missing:

        raise ValueError(
            f"Prompt '{key}' requires variables {required}. "
            f"Missing: {missing}"
        )

    return template.format(**kwargs) if kwargs else template


def get_raw(key: str) -> dict:
    """Return the full entry dict (template + description + variables) for a key."""

    prompts = _load()
    
    if key not in prompts:

        raise KeyError(f"Prompt '{key}' not found.")

    return prompts[key]


def list_keys() -> list[str]:
    """Return all available prompt keys."""

    return list(_load().keys())