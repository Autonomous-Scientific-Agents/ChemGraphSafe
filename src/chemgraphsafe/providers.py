"""Hosted Argo names adapted from ChemGraph's models/endpoints/argo.py."""

ARGO_ENDPOINT = "https://apps.inside.anl.gov/argoapi/v1/chat/completions"

# Explicit wire names, not a punctuation-stripping heuristic. Claude uses a different protocol.
ARGO_MODELS = {
    "gpt-4o": "gpt4o",
    "gpt-4.1": "gpt41",
    "gpt-4.1-mini": "gpt41mini",
    "gpt-4.1-nano": "gpt41nano",
    "gpt-5": "gpt5",
    "gpt-5-mini": "gpt5mini",
    "gpt-5-nano": "gpt5nano",
    "gpt-5.1": "gpt51",
    "gpt-5.2": "gpt52",
    "gpt-5.4": "gpt54",
    "gpt-5.4-mini": "gpt54mini",
    "gpt-5.4-nano": "gpt54nano",
    "gpt-5.5": "gpt55",
    "gpt-5.6-luna": "gpt56luna",
    "gpt-5.6-sol": "gpt56sol",
    "gpt-5.6-terra": "gpt56terra",
    "o1": "gpto1",
    "o3-mini": "gpto3mini",
    "o3": "gpto3",
    "o4-mini": "gpto4mini",
    "gemini-2.5-pro": "gemini25pro",
    "gemini-2.5-flash": "gemini25flash",
    "gemini-3.1-flash-lite": "gemini31flashlite",
    "gemini-3.5-flash": "gemini35flash",
}

ARGO_WITHOUT_TEMPERATURE = {
    "argo:gpt-5",
    "argo:gpt-5-mini",
    "argo:gpt-5-nano",
    "argo:gpt-5.5",
    "argo:gpt-5.6-luna",
    "argo:gpt-5.6-sol",
    "argo:gpt-5.6-terra",
}


def argo_model(name: str) -> str:
    if not name.startswith("argo:") or name.removeprefix("argo:") not in ARGO_MODELS:
        raise ValueError("Argo requires a supported argo:<model-id> (non-Claude) model")
    return ARGO_MODELS[name.removeprefix("argo:")]


def model_identity(name: str) -> str:
    """Compare known aliases conservatively; unknown checkpoint aliases remain distinct."""
    name = name.removeprefix("codex:").removeprefix("argo:")
    return {wire: public for public, wire in ARGO_MODELS.items()}.get(name, name)
