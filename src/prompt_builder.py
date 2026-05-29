from .models import FunctionDefinition, PromptInput


def format_function(function: FunctionDefinition) -> str:
    parameters = "\n".join(
        f"- {name}: {info['type']}"
        for name, info in function.parameters.items()
    )

    return (
        f"{function.name}\n"
        f"Description: {function.description}\n"
        f"Parameters:\n"
        f"{parameters}"
    )


def build_prompt(
    functions: list[FunctionDefinition],
    prompt_input: PromptInput,
) -> str:
    functions_text = "\n\n".join(
        format_function(function)
        for function in functions
    )

    return (
        "You are a strict function-calling assistant.\n"
        "Your task is to choose the best function for the user request "
        "and generate its parameters.\n\n"
        "Rules:\n"
        "- Return only one JSON object.\n"
        "- Do not explain.\n"
        "- Do not write Answer:.\n"
        "- Do not repeat the JSON.\n"
        "- The JSON must contain only these keys: name, parameters.\n"
        "- The function name must be one of the available functions.\n"
        "- The parameters must match the selected function schema.\n"
        "- Use concrete values, not descriptions.\n"
        "- If the user says asterisks, use \"*\".\n"
        "- If the user says numbers or digits for a regex, use \"\\\\d+\".\n"
        "- If the user says vowels for a regex, use \"[aeiouAEIOU]\".\n\n"
        "JSON format:\n"
        "{\"name\":\"function_name\",\"parameters\":{}}\n\n"
        "Available functions:\n\n"
        f"{functions_text}\n\n"
        "User request:\n"
        f"{prompt_input.prompt}\n\n"
        "JSON:\n"
    )