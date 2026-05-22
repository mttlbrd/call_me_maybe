from pydantic import BaseModel


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class PromptInput(BaseModel):
    prompt: str


class FunctionResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, str | int | float | bool]