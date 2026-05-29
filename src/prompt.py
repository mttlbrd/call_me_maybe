import json
from typing import List
from .models import FunctionDefinition

def build_prompt(functions: List[FunctionDefinition], user_prompt: str) -> str:
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

def build_pruned_prompt(selected_function: FunctionDefinition, user_prompt: str) -> str:
    function_data = [selected_function.model_dump()]
    functions_json = json.dumps(function_data, indent=2)
    
    prompt = (
        "You are an expert system that extracts function calls from text.\n"
        "Available functions:\n"
        f"{functions_json}\n\n"
        f"User query: {user_prompt}\n"
        "Generate a JSON object with exactly 'name' and 'parameters' keys.\n"
        "Output:"
    )
    
    return prompt