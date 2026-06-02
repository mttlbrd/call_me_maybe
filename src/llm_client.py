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
