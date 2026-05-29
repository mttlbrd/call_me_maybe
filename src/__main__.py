import json
from pathlib import Path
from .cli import input_parser
from .loader import load_function, load_prompt
from .llm_client import create_model, encode_prompt, token_ids_to_list, get_next_token_logits, decode_tokens
from .prompt import build_prompt, build_pruned_prompt
from .decoder import ParserState, advance_state, get_best_valid_token

def load_vocabulary(model):
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_dict = json.load(f)
    return {v: k for k, v in vocab_dict.items()}

def main():
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
    
    for prompt_data in prompts:
        print(f"\n--- Elaborazione: '{prompt_data.prompt}' ---")
        current_prompt_text = build_prompt(functions, prompt_data.prompt)
        input_ids_tensor = encode_prompt(model, current_prompt_text)
        input_ids = token_ids_to_list(input_ids_tensor)
        
        state = ParserState.START
        generated_text = ""
        full_generated_json = ""
        is_pruned = False
        
        while True:
            logits = get_next_token_logits(model, input_ids)
            best_token_id = get_best_valid_token(
                logits, id_to_token, state, generated_text, available_func_names
            )
            token_str = decode_tokens(model, [best_token_id])
            input_ids.append(best_token_id)
            full_generated_json += token_str
            
            print(token_str, end="", flush=True)
            
            state, generated_text, selected_func_name = advance_state(
                token_str, state, generated_text, available_func_names
            )
            
            if selected_func_name and not is_pruned:
                selected_func_obj = next(f for f in functions if f.name == selected_func_name)
                new_prompt_text = build_pruned_prompt(selected_func_obj, prompt_data.prompt)
                input_ids_tensor = encode_prompt(model, new_prompt_text + full_generated_json)
                input_ids = token_ids_to_list(input_ids_tensor)
                is_pruned = True
                
            if full_generated_json.count('{') == full_generated_json.count('}') and full_generated_json.count('{') > 0:
                break
                
            if len(full_generated_json) > 500:
                break
                
        print("\n--- Fine ---")
        
        try:
            parsed_result = json.loads(full_generated_json)
            parsed_result["prompt"] = prompt_data.prompt
            results.append(parsed_result)
        except json.JSONDecodeError:
            pass
            
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()