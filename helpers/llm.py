import re

def estimate_token_usage(message:str) -> int:
    char_estimate = len(message) / 4.0
    word_estimate = len(re.findall(r"\w+", message, flags=re.UNICODE)) * 1.5
    
    return int((char_estimate + word_estimate)/2.0)