from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ProgrammingLanguage(str, Enum):
    CPP = "C++"
    C = "C"
    GO = "GOLANG"
    RUST = "RUST"
    PYTHON = "PYTHON"
    JAVA = "JAVA"
    JAVASCRIPT = "JAVASCRIPT"
    TYPESCRIPT = "TYPESCRIPT"
    PHP = "PHP"
    RUBY = "RUBY"
    SCALA = "SCALA"
    KOTLIN = "KOTLIN"


class ProjectBasicInformation(BaseModel):
    project_dir: str = Field(
        title="Project directory", description="Project folder for analysis"
    )
    language: ProgrammingLanguage = Field(
        title="Programming language",
        description="The selected programming language for implementing the project",
    )
    framework: str = Field(
        title="Framework",
        description="The framework selected for the project's implementation",
    )


class DependencyTag(str, Enum):
    # Web / UI
    WEB_FRAMEWORK = "WebFramework"
    UI = "UI"
    TEMPLATING = "Templating"
    I18N = "Internationalization"

    # Security
    SECURITY = "Security"
    CRYPTOGRAPHY = "Cryptography"
    AUTHN = "Authentication"
    AUTHZ = "Authorization"
    SECRET_MANAGEMENT = "SecretManagement"

    # Parsers
    FORMAT_PARSING = "Parsing"

    # Interactions
    EXTERNAL_API = "ExternalAPI"
    DATABASE = "Database"
    CACHE = "Cache"
    MESSAGE_BROKER = "MessageBroker"
    FILESYSTEM = "Filesystem"
    OPERATING_SYSTEM = "OperatingSystem"

    # Protocols / APIs
    HTTP = "HTTP"
    GRPC = "gRPC"
    GRAPHQL = "GraphQL"
    SOAP = "SOAP"
    WEBSOCKET = "WebSocket"

    # Storage / Cloud
    S3 = "S3"

    # ML
    ML = "ML"

    # Observability / Config
    LOGGING = "Logging"
    METRICS = "Metrics"
    TRACING = "Tracing"
    CONFIG = "Config"

    # DevOps
    DEVOPS = "DevOps"


class DependencyInformation(BaseModel):
    package: str = Field(
        description="The package or library name, as specified in the dependency file (e.g., requirements.txt, pyproject.toml, package.json)."
    )
    version: Optional[str] = Field(
        description="The package or library version, if explicitly specified (e.g., '1.2.3'). Can be omitted if the version is not pinned or is resolved automatically."
    )
    description: Optional[str] = Field(
        description="Brief description of what this dependency is used for. Can be omitted if not sure"
    )
    tags: list[DependencyTag] = Field(
        default_factory=list,
        description=(
            "A list of categories (DependencyTag) describing the role of this dependency. "
            "Multiple tags may apply simultaneously to reflect different aspects, "
            "such as web framework, security, database interaction, cloud storage, etc."
        )
    )

class ProjectDependencies(BaseModel):
    dependencies: list[DependencyInformation] = Field(
        default_factory=list, description="list of project dependencies"
    )
