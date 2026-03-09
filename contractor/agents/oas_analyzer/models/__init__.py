from typing import Literal

from pydantic import BaseModel, Field


class EndpointVulnerability(BaseModel):
    """
    Model to describe a vulnerability in an HTTP-endpoint.
    """

    tag: str
    path: str = Field(description="The path of the endpoint that is vulnerable.")
    method: str = Field(
        description="The HTTP method of the endpoint that is vulnerable."
    )
    parameters: list[str] = Field(
        default_factory=list,
        description="The parameters of the endpoint that are vulnerable.",
    )
    vulnerability: str = Field(description="The type of vulnerability.")
    description: str = Field(description="A description of the vulnerability.")
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="The severity of the vulnerability."
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="The confidence level of the vulnerability."
    )


class ServiceBasicInfo(BaseModel):
    """
    Model to describe a service.
    """

    name: str = Field(description="The name of the analyzed service.")
    description: str = Field(description="Description of the service.")
    summary: str = Field(
        description="Detailed summary of the service business functions."
    )
    diagram: str = Field(description="A diagram of the service.")
    criticality: Literal["low", "medium", "high"] = Field(
        description="The criticality of the service."
    )
    criticality_reason: str = Field(
        description="The reason for the criticality of the service."
    )
