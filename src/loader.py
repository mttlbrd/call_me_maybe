import json
from pathlib import Path
from .models import FunctionDefinition, PromptInput


def load_json(path: Path) -> list[dict]:
    """Utility function to load a JSON file and return
    its contents as a list of dictionaries."""

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_function(path: Path) -> list[FunctionDefinition]:
    """Loads function definitions from a JSON file and returns
    a list of FunctionDefinition instances."""

    data = load_json(path)

    return [
        FunctionDefinition.model_validate(item)
        for item in data
    ]


def load_prompt(path: Path) -> list[PromptInput]:
    """Loads prompt inputs from a JSON file and returns
    a list of PromptInput instances."""

    data = load_json(path)

    return [
        PromptInput.model_validate(item)
        for item in data
    ]
