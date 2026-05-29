from llm_sdk import Small_LLM_Model


def create_model() -> Small_LLM_Model:
    return Small_LLM_Model()


def encode_prompt(model: Small_LLM_Model, prompt: str):
    return model.encode(prompt)


def token_ids_to_list(token_ids) -> list[int]:
    return token_ids[0].tolist()


def get_next_token_logits(
    model: Small_LLM_Model,
    input_ids: list[int],
) -> list[float]:
    return model.get_logits_from_input_ids(input_ids)


def decode_tokens(
    model: Small_LLM_Model,
    token_ids: list[int],
) -> str:
    return model.decode(token_ids)