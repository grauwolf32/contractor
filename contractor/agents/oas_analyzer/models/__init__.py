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


class EndpointVulnerabilities(BaseModel):
    vulnerabilities: list[EndpointVulnerability] = Field(
        default_factory=list, description="A list of vulnerabilities in the endpoints."
    )


class Report(BaseModel):
    service_info: ServiceInfo = Field(
        default_factory=ServiceInfo, description="The service information."
    )
    vulneabilities: list[EndpointVulnerability] = Field(
        default_factory=list, description="The vulnerabilities found in the service."
    )
    reccomendation: str = Field(
        default="", description="The reccomendation for mitigating the vulnerabilities."
    )


class ServiceBasicInfo(BaseModel):
    """
    Model to describe a service.
    """

    name: str = Field(description="The name of the service.")
    description: str = Field(description="A description of the service.")
    summary: str = Field(description="A summary of the service.")
    diagram: str = Field(description="A diagram of the service.")
    criticality: Literal["low", "medium", "high"] = Field(
        description="The criticality of the service."
    )
    criticality_reason: str = Field(
        description="The reason for the criticality of the service."
    )


class ServiceInfo(ServiceBasicInfo):
    """
    Model to describe a service (internal usage)
    """

    servers: dict[str, Any] = Field(
        default_factory=dict,
        description="The servers of the service, as described in the OpenAPI specification.",
    )
    security: dict[str, Any] = Field(
        default_factory=dict,
        description="The security of the service, as described in the OpenAPI specification.",
    )
    language: str = Field(
        default="", description="The programming language of the service."
    )
    framework: str = Field(default="", description="The framework of the service.")
