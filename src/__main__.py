from pathlib import Path

from .cli import input_parser
from .prompt_builder import build_prompt
from .loader import load_function, load_prompt
from .llm_client import create_model, encode_prompt


def run() -> None:
    args = input_parser()
    functions = load_function(Path(args.functions_definition))
    prompts = load_prompt(Path(args.input))

    print(f"Loaded functions: {len(functions)}")
    print(f"Loaded prompts: {len(prompts)}")

    print(f"Output path: {args.output}")

    first_prompt = build_prompt(functions, prompts[0])
    print(first_prompt)

    print("Loading model...")
    model = create_model()
    print("Model loaded")

    token_ids = encode_prompt(model, first_prompt)
    print(token_ids)


def main() -> None:
    try:
        run()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()