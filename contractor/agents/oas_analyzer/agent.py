from __future__ import annotations

import logging

from google.adk.agents import SequentialAgent

from contractor.agents.oas_analyzer.sub_agents.analytic_agents import analytic_agent
from contractor.agents.oas_analyzer.sub_agents.report_agent import report_agent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

report_generator = SequentialAgent(
    name="report_generator",
    description="report generator agent to generate a report of the vulnerabilities found",
    sub_agents=[analytic_agent, report_agent],
)

root_agent = report_generator
