from unittest.mock import MagicMock


def mk_callback_context(initial_state=None) -> MagicMock:
    """
    CallbackContext нам нужен только с полем .state (dict).
    """
    ctx = MagicMock()
    ctx.state = initial_state or {"callbacks": {}}
    return ctx


def mk_llm_response(interaction_id, total, prompt, candidates) -> MagicMock:
    """
    LlmResponse с полями:
      - .interaction_id
      - .usage_metadata.total_token_count
      - .usage_metadata.prompt_token_count
      - .usage_metadata.candidates_token_count
    """
    resp = MagicMock()
    resp.interaction_id = interaction_id

    usage = MagicMock()
    usage.total_token_count = total
    usage.prompt_token_count = prompt
    usage.candidates_token_count = candidates

    resp.usage_metadata = usage
    return resp
