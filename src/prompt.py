import json

def build_minimal_prompt(user_prompt: str) -> str:
    prompt = (
        "You are an expert system that extracts function calls from text.\n"
        f"User query: {user_prompt}\n"
        "Generate a JSON object with exactly 'name' and 'parameters' keys.\n"
        "Output:"
    )
    return prompt

def build_full_prompt(functions: list, user_prompt: str) -> str:
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
    function_data = [selected_function.model_dump()]
    functions_json = json.dumps(function_data, indent=2)
    prompt = (
        "You are an expert system that extracts function calls from text.\n"
        "Available functions:\n"
        f"{functions_json}\n\n"
        f"User query: {user_prompt}\n"
        "Generate a JSON object with exactly 'name' and 'parameters' keys.\n"
        "Do not generate any extra keys or information beyond the parameters of the selected function.\n"
        "Output:"
    )
    return prompt