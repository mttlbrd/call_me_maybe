from .models import FunctionDefinition, PromptInput


def format_parameter(name: str, info: dict[str, str]) -> str:
    """Formats a single parameter for display."""
    return f"- {name}: {info['type']}"


def format_parameters(function: FunctionDefinition) -> str:
    """Formats the parameters of a function for display."""
    return "\n".join(
        format_parameter(name, info)
        for name, info in function.parameters.items()
    )


def format_function(function: FunctionDefinition) -> str:
    """Formats a function definition for display."""
    parameters = format_parameters(function)
    return (
        f"{function.name}\n"
        f"Description: {function.description}\n"
        f"Parameters:\n"
        f"{parameters}"
    )


def format_functions(functions: list[FunctionDefinition]) -> str:
    """Formats a list of function definitions for display."""
    return "\n\n".join(
        format_function(function)
        for function in functions
    )


def build_prompt(
    functions: list[FunctionDefinition],
    prompt_input: PromptInput,
) -> str:
    """Builds a prompt for the language model
    based on the provided functions and prompt input."""
    functions_text = format_functions(functions)

    return (
        "You are a function-calling assistant.\n\n"
        "Available functions:\n\n"
        f"{functions_text}\n\n"
        "User request:\n"
        f"{prompt_input.prompt}\n\n"
        "Return only valid JSON in this format:\n"
        '{ "name": "function_name", "parameters": { } }'
    )