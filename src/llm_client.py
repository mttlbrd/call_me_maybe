import json
from llm_sdk import Small_LLM_Model


def create_model() -> Small_LLM_Model:
    """Create and return an instance of the small LLM model."""

    return Small_LLM_Model()


def encode_prompt(model: Small_LLM_Model, prompt: str) -> list[int]:
    """Encode a prompt and return the input IDs as a plain Python list."""

    token_ids = model.encode(prompt)
    return token_ids[0].tolist()


def get_next_token_logits(
    model: Small_LLM_Model,
    input_ids: list[int],
) -> list[float]:
    """Get the raw logits for the next token from the model."""

    return model.get_logits_from_input_ids(input_ids)


def decode_tokens(
    model: Small_LLM_Model,
    token_ids: list[int],
) -> str:
    """Decode token IDs back into a string using the model's tokenizer."""

    return model.decode(token_ids)


def load_vocabulary(model):
    """Loads the model's vocabulary and returns
    a mapping from token IDs to token strings."""

    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_dict = json.load(f)
    return {v: k for k, v in vocab_dict.items()}


def rebuild_input_ids(model, prompt_text: str, generated_json: str
                      ) -> list[int]:
    """Rebuilds the input IDs for the model based on the current prompt text
    and the JSON generated so far.
    This is necessary to ensure that the model's attention mechanism
    has access to the full context, including the generated text,
    which can help it generate valid JSON structures."""

    return encode_prompt(model, prompt_text + generated_json)
