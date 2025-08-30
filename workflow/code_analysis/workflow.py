from __future__ import annotations
from tools.cdxgen import cdxgen_tool, CdxgenOutput
from helpers.llm import estimate_token_usage
from workflow.base import settings

def analyze_dependencies(project_dir: str):
    cdxgen_output: CdxgenOutput = CdxgenOutput(**cdxgen_tool(project_dir))
    tokens_used = estimate_token_usage(cdxgen_output.model_dump())

    if tokens_used / settings.context_length < 0.4:
        analyze_short()


def analyze_short(project_dir: str, cdxgen_output:CdxgenOutput):
    ...

def analyze_long(project_dir: str, cdxgen_output:CdxgenOutput):
    ...