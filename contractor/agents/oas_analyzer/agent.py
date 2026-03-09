from __future__ import annotations

import logging

from google.adk.agents import LlmAgent, SequentialAgent
from langfuse import Langfuse

from contractor.agents.oas_analyzer.sub_agents.analytic_agents import analytic_agent
from contractor.agents.oas_analyzer.sub_agents.report_agent import report_agent
from contractor.agents.oas_analyzer.sub_agents.review_agent import review_agent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

langfuse: Langfuse = None

# try:
#    langfuse = config.setup_langfuse()
# except Exception as exc:
#    logger.warning(f"Failed to setup langfuse: {exc}")
#    langfuse = None


report_generator = SequentialAgent(
    name="report_generator",
    description="report generator agent to generate a report of the vulnerabilities found",
    sub_agents=[review_agent, analytic_agent, report_agent],
)

root_agent = report_generator
