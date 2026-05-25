from json import JSONDecodeError
from pathlib import Path
from .cli import input_parser
from pydantic import ValidationError
from .loader import load_function, load_prompt
from llm_sdk import Small_LLM_Model
from .models import PromptInput


def test_model(prompts: list[PromptInput]) -> None:
    model = Small_LLM_Model()

    first_prompt = prompts[0].prompt

    print(f"Prompt: {first_prompt}")

    input_ids = model.encode(first_prompt)

    print(f"Input ids shape: {input_ids.shape}")
    print(f"Input ids: {input_ids}")

    logits = model.get_logits_from_input_ids(
        input_ids[0].tolist()
    )

    print(f"Vocabulary size: {len(logits)}")

    print("First 10 logits:")
    print(logits[:10])



def main():
    args = input_parser()

    try:
        functions = load_function(Path(args.functions_definition))
        prompts = load_prompt(Path(args.input))
        output = Path(args.output)

        test_model(prompts)

    except ValidationError:
        print("Impossibile validare il .json")
    except (FileNotFoundError, PermissionError):
        print("file inesistente o senza permessi")
    except JSONDecodeError:
        print("File .json non valido")

if __name__ == "__main__":
    main()