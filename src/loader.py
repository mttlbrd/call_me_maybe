import json
from pathlib import Path
from .models import FunctionDefinition, PromptInput


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_function(path: Path) -> list[FunctionDefinition]:
    data = load_json(path)

    return [
        FunctionDefinition.model_validate(item)
        for item in data
    ]


def load_prompt(path: Path) -> list[PromptInput]:
    data = load_json(path)

    return [
        PromptInput.model_validate(item)
        for item in data
    ]