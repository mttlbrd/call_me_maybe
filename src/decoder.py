import re
from enum import Enum, auto


class State(Enum):
    """Represents the current state of the parser
    as it processes the generated JSON."""

    START = auto()
    FUNCTION_NAME = auto()
    TRANSITION = auto()
    PARAMETERS = auto()


def count_unquoted_braces(text: str) -> tuple[int, int]:
    """Counts the number of unquoted opening and closing
    braces in the text."""

    in_str = False
    escaped = False
    open_b = 0
    close_b = 0
    for char in text:
        if escaped:
            escaped = False
        elif char == '\\':
            escaped = True
        elif char == '"':
            in_str = not in_str
        elif not in_str:
            if char == '{':
                open_b += 1
            elif char == '}':
                close_b += 1
    return open_b, close_b


def is_valid_token(token_str: str,
                   state: State,
                   generated_text: str,
                   available_functions: list[str],
                   expected_keys: list[str]
                   ) -> bool:
    """Determines if a token is valid given the current parser state
    and the text generated so far."""

    temp_text = generated_text + token_str

    if state == State.START:
        if '{"name":"'.startswith(temp_text):
            return True
        if temp_text.startswith('{"name":"'):
            excess = temp_text[len('{"name":"'):]
            if not excess:
                return True
            for func in available_functions:
                if func.startswith(excess) or excess.startswith(func):
                    return True
            return False
        return False

    if state == State.FUNCTION_NAME:
        for func in available_functions:
            if func.startswith(temp_text):
                return True
            if temp_text.startswith(func):
                excess = temp_text[len(func):]
                expected_transition = '","parameters":{'
                if (expected_transition.startswith(excess)
                   or excess.startswith(expected_transition)):
                    return True
        return False

    if state == State.TRANSITION:
        expected = '","parameters":{'
        if expected.startswith(temp_text):
            return True
        if temp_text.startswith(expected):
            return True
        return False

    if state == State.PARAMETERS:
        found_keys = re.findall(r'"([^"]+)"\s*:', temp_text)

        for key in found_keys:
            if key not in expected_keys:
                return False

        open_b, close_b = count_unquoted_braces(temp_text)

        if close_b > open_b:
            unique_found_keys = set(found_keys)
            if len(unique_found_keys) != len(expected_keys):
                return False

        return True

    return False


def advance_state(token_str: str,
                  state: State,
                  generated_text: str,
                  available_functions: list[str]
                  ) -> tuple[State, str, str | None]:
    """Advances the parser state based on the newly generated token
    and the text generated so far. Also returns the selected function name
    if we just transitioned out of the FUNCTION_NAME state."""

    new_text = generated_text + token_str
    selected_func = None

    if state == State.START:
        if new_text.startswith('{"name":"'):
            state = State.FUNCTION_NAME
            new_text = new_text[len('{"name":"'):]
        else:
            return state, new_text, selected_func

    if state == State.FUNCTION_NAME:
        for func in available_functions:
            if new_text.startswith(func):
                selected_func = func
                state = State.TRANSITION
                new_text = new_text[len(func):]
                break

    if state == State.TRANSITION:
        expected = '","parameters":{'
        if new_text.startswith(expected):
            state = State.PARAMETERS
            new_text = new_text[len(expected):]

    return state, new_text, selected_func


def get_best_valid_token(logits: list[float],
                         id_to_token: dict,
                         state: State,
                         generated_text: str,
                         available_functions: list[str],
                         expected_keys: list[str],
                         top_k: int = 50
                         ) -> int:
    """Given the logits for the next token,
    returns the ID of the best valid token
    based on the current parser state and the text generated so far."""

    candidate_indices = sorted(
        range(len(logits)),
        key=lambda idx: logits[idx],
        reverse=True,
    )[:top_k]

    def select_best_valid(indices: list[int]) -> int | None:
        best_idx = None
        max_logit = float('-inf')

        for idx in indices:
            token_str = id_to_token.get(idx, "")
            if not is_valid_token(token_str,
                                  state, generated_text,
                                  available_functions,
                                  expected_keys):
                continue

            val = logits[idx]
            if val > max_logit:
                max_logit = val
                best_idx = idx

        return best_idx

    best_idx = select_best_valid(candidate_indices)
    if best_idx is not None:
        return best_idx

    fallback_idx = select_best_valid(list(range(len(logits))))
    if fallback_idx is not None:
        return fallback_idx

    return candidate_indices[0] if candidate_indices else 0
