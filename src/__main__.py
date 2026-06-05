import json
from time import perf_counter
from pathlib import Path

from .cli import input_parser
from .loader import load_function, load_prompt
from .llm_client import (create_model,
                         get_next_token_logits, decode_tokens,
                         load_vocabulary, rebuild_input_ids)
from .prompt import (build_minimal_prompt,
                     build_full_prompt, build_pruned_prompt)
from .decoder import (State, advance_state,
                      get_best_valid_token, count_unquoted_braces)


def process_single_prompt(model, id_to_token, functions,
                          available_func_names, prompt_data) -> str:
    """Processes a single prompt through the function calling generation loop
    and returns the full generated JSON string."""

    current_base_prompt = build_minimal_prompt(prompt_data.prompt)
    input_ids = rebuild_input_ids(model, current_base_prompt, "")

    state = State.START
    generated_text = ""
    full_generated_json = ""
    expected_keys: list[str] = []

    while True:
        logits = get_next_token_logits(model, input_ids)

        best_token_id = get_best_valid_token(
            logits, id_to_token, state, generated_text,
            available_func_names, expected_keys
        )

        token_str = decode_tokens(model, [best_token_id])
        input_ids.append(best_token_id)
        full_generated_json += token_str

        print(token_str, end="", flush=True)

        old_state = state
        state, generated_text, selected_func = advance_state(
            token_str, state, generated_text, available_func_names
        )

        if old_state == State.START and state == State.FUNCTION_NAME:
            current_base_prompt = build_full_prompt(functions,
                                                    prompt_data.prompt)
            input_ids = rebuild_input_ids(model, current_base_prompt,
                                          full_generated_json)

        elif old_state == State.FUNCTION_NAME and state == State.TRANSITION:
            selected_func_obj = next(f for f in functions
                                     if f.name == selected_func)
            expected_keys = list(selected_func_obj.parameters.keys())

            current_base_prompt = build_pruned_prompt(selected_func_obj,
                                                      prompt_data.prompt)
            input_ids = rebuild_input_ids(model, current_base_prompt,
                                          full_generated_json)

        open_b, close_b = count_unquoted_braces(full_generated_json)
        if open_b == close_b and open_b > 0:
            break

        if len(full_generated_json) > 500:
            break

    return full_generated_json


def save_results(results, output_path: Path):
    """Saves the generated results to the specified
    output path in JSON format."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main():
    """"Main function that orchestrates the loading of data,
    processing of prompts, and saving of results."""

    args = input_parser()
    functions_path = Path(args.functions_definition)
    inputs_path = Path(args.input)
    output_path = Path(args.output)

    functions = load_function(functions_path)
    prompts = load_prompt(inputs_path)
    available_func_names = [f.name for f in functions]

    model = create_model()
    id_to_token = load_vocabulary(model)
    results = []
    total_start = perf_counter()

    for prompt_data in prompts:
        prompt_start = perf_counter()
        print(f"\n\033[1;30m--- '{prompt_data.prompt}' ---\033[0m")

        full_generated_json = process_single_prompt(
            model, id_to_token, functions, available_func_names, prompt_data
        )

        try:
            parsed_result = json.loads(full_generated_json)
            ordered_result = {"prompt": prompt_data.prompt, **parsed_result}
            results.append(ordered_result)
        except json.JSONDecodeError:
            pass

        prompt_elapsed = perf_counter() - prompt_start
        total_elapsed = perf_counter() - total_start
        print(f"\n\033[90m[tempo prompt: {prompt_elapsed:.2f}s"
              f" | tempo totale: {total_elapsed:.2f}s]\033[0m")

    save_results(results, output_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting gracefully.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Exiting gracefully :')")
