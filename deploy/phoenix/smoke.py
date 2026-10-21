"""Real-exporter OTLP check against the disposable Phoenix test container."""

import json
import time
from urllib.error import HTTPError
from urllib.request import urlopen


def check():
    expected = {
        "contractor.planner.model": ("Go", 2, 3, 1),
        "contractor.worker.model": ("Runtime", 13, 17, 3),
    }
    for _ in range(30):
        try:
            with urlopen(
                "http://127.0.0.1:6006/v1/projects/default/spans?limit=10", timeout=5
            ) as response:
                spans = {span["name"]: span for span in json.load(response)["data"]}
        except HTTPError as error:
            if error.code != 404:
                raise
            spans = {}
        if set(expected) <= set(spans):
            for name, (source, inputs, outputs, cached) in expected.items():
                span = spans[name]
                values = span["attributes"]
                checks = {
                    "llm.token_count.prompt": inputs,
                    "llm.token_count.completion": outputs,
                    "llm.token_count.total": inputs + outputs,
                    "llm.token_count.prompt_details.cache_read": cached,
                    "llm.input_messages.0.message.role": "system",
                    "llm.input_messages.0.message.content": f"{source} system instruction",
                    "llm.input_messages.1.message.role": "user",
                    "llm.input_messages.1.message.content": f"synthetic {source} input",
                    "llm.output_messages.0.message.role": "assistant",
                    "llm.output_messages.0.message.content": f"synthetic {source} output",
                }
                if span["span_kind"] != "LLM":
                    raise RuntimeError(f"{name} was not classified as an LLM span")
                for key, value in checks.items():
                    if values.get(key) != value:
                        raise RuntimeError(
                            f"{name}: native Phoenix projection mismatch for {key}: {values.get(key)!r}"
                        )
                if any(key.startswith("langfuse.") for key in values):
                    raise RuntimeError("Collector received vendor-specific attributes")
            print(
                "Real Go/Runtime exporters: native LLM prompts, responses and token projections verified"
            )
            return
        time.sleep(1)
    raise RuntimeError("The ingested spans did not appear within the probe budget")


if __name__ == "__main__":
    check()
