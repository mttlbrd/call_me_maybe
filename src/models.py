from pydantic import BaseModel


class FunctionDefinition(BaseModel):
    """Represents a function definition with its name, description,
    parameters, and return type."""

    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class PromptInput(BaseModel):
    """Represents a single prompt input with its text content."""

    prompt: str


class FunctionResult(BaseModel):
    """Represents the result of a function call extracted from a prompt,
    including the original prompt, the function name, and the parameters."""

    prompt: str
    name: str
    parameters: dict[str, str | int | float | bool]
