from llm_sdk import Small_LLM_Model


def create_model() -> Small_LLM_Model:
    """Creates and returns an instance of the Small_LLM_Model."""
    return Small_LLM_Model()


def encode_prompt(model: Small_LLM_Model, prompt: str):
    """Encodes a prompt into token ids using the model's tokenizer."""
    return model.encode(prompt)


def token_ids_to_list(token_ids) -> list[int]:
    """Converts a tensor of token ids to a list of integers."""
    return token_ids[0].tolist()


def get_next_token_logits(
    model: Small_LLM_Model,
    input_ids: list[int],
) -> list[float]:
    """Given a list of input token ids, returns the raw logits for the next token."""
    return model.get_logits_from_input_ids(input_ids)
