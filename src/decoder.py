from enum import Enum, auto

class ParserState(Enum):
    START = auto()
    FUNCTION_NAME = auto()
    TRANSITION = auto()
    PARAMETERS = auto()

def is_valid_token(token_str: str, state: ParserState, generated_text: str, available_functions: list[str]) -> bool:
    temp_text = generated_text + token_str

    if state == ParserState.START:
        if '{"name":"'.startswith(temp_text):
            return True
        if temp_text.startswith('{"name":"'):
            excess = temp_text[len('{"name":"'):]
            if not excess:
                return True
            # Se il token eccede '{"name":"', controlliamo immediatamente che 
            # l'eccesso sia l'inizio valido di una funzione
            for func in available_functions:
                if func.startswith(excess) or excess.startswith(func):
                    return True
            return False
        return False
    
    elif state == ParserState.FUNCTION_NAME:
        for func in available_functions:
            if func.startswith(temp_text):
                return True
            if temp_text.startswith(func):
                excess = temp_text[len(func):]
                expected_transition = '","parameters":{'
                # Controlliamo che l'eventuale stringa successiva al nome funzione sia la sintassi corretta
                if expected_transition.startswith(excess) or excess.startswith(expected_transition):
                    return True
        return False
        
    elif state == ParserState.TRANSITION:
        expected = '","parameters":{'
        if expected.startswith(temp_text):
            return True
        if temp_text.startswith(expected):
            return True
        return False
        
    elif state == ParserState.PARAMETERS:
        return True

    return False

def advance_state(token_str: str, state: ParserState, generated_text: str, available_functions: list[str]) -> tuple[ParserState, str, str | None]:
    new_text = generated_text + token_str
    selected_func = None
    
    # Usiamo gli 'if' in sequenza (non 'elif') per permettere a un singolo token lungo 
    # di far avanzare la macchina di più stati in un colpo solo
    if state == ParserState.START:
        if new_text.startswith('{"name":"'):
            state = ParserState.FUNCTION_NAME
            new_text = new_text[len('{"name":"'):]
        else:
            return state, new_text, selected_func
            
    if state == ParserState.FUNCTION_NAME:
        for func in available_functions:
            if new_text.startswith(func):
                selected_func = func
                state = ParserState.TRANSITION
                new_text = new_text[len(func):]
                break
                
    if state == ParserState.TRANSITION:
        expected = '","parameters":{'
        if new_text.startswith(expected):
            state = ParserState.PARAMETERS
            new_text = new_text[len(expected):]
            
    return state, new_text, selected_func

def get_best_valid_token(logits: list[float], id_to_token: dict, state: ParserState, generated_text: str, available_functions: list[str]) -> int:
    for idx in range(len(logits)):
        token_str = id_to_token.get(idx, "")
        if not is_valid_token(token_str, state, generated_text, available_functions):
            logits[idx] = float('-inf')
            
    best_idx = 0
    max_logit = float('-inf')
    
    for idx, val in enumerate(logits):
        if val > max_logit:
            max_logit = val
            best_idx = idx
            
    return best_idx