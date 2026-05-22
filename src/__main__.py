from json import JSONDecodeError
from pathlib import Path
from .cli import input_parser
from pydantic import ValidationError
from .loader import load_function, load_prompt



def main():
    args = input_parser()
    try:
        functions = load_function(Path(args.functions_definition))
        prompts = load_prompt(Path(args.input))
        output = Path(args.output)
    except ValidationError:
        print("Impossibile validare il .json")
    except (FileNotFoundError, PermissionError):
        print("file inesistene o senza permessi")
    except JSONDecodeError:
        print("File .json non valido")


if __name__ == "__main__":
    main()