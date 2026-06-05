import json


def build_minimal_prompt(user_prompt: str) -> str:
    """Builds a minimal prompt without function definitions,
    used at the start of generation."""

    prompt = (
        "You are an expert system that extracts function calls from text.\n"
        f"User query: {user_prompt}\n"
        "Generate a JSON object with exactly 'name' and 'parameters' keys.\n"
        "Output:"
    )
    return prompt


def build_full_prompt(functions: list, user_prompt: str) -> str:
    """Builds a full prompt including all available function definitions,
    used after the model starts generating and we know it needs
    to select a function. This provides the model with the necessary
    context to make an informed choice."""

    functions_data = [f.model_dump() for f in functions]
    functions_json = json.dumps(functions_data, indent=2)
    prompt = (
        "You are an expert system that extracts function calls from text.\n"
        "Available functions:\n"
        f"{functions_json}\n\n"
        f"User query: {user_prompt}\n"
        "Generate a JSON object with exactly 'name' and 'parameters' keys.\n"
        "Output:"
    )
    return prompt


def build_pruned_prompt(selected_function, user_prompt: str) -> str:
    """Builds a pruned prompt that only includes the selected
    function definition, used after the model has selected a function name.
    This helps guide the model to focus on generating the correct parameters"""

    function_data = [selected_function.model_dump()]
    functions_json = json.dumps(function_data, indent=2)
    prompt = (
        "You are an expert system that extracts function calls from text.\n"
        "Available function:\n"
        f"{functions_json}\n\n"
        f"User query: {user_prompt}\n"
        "Generate a JSON object with exactly 'name' and 'parameters' keys.\n"
        "Output:"
    )
    return prompt
