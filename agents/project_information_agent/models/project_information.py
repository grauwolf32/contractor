from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class DependencyInformation(BaseModel):
    name: str = Field(
        description="The package or library name, as specified in the dependency file (e.g., requirements.txt, pyproject.toml, package.json)."
    )
    version: str = Field(
        description="The package or library version, if explicitly specified (e.g., '1.2.3')."
    )
    description: str = Field(
        description="Brief description of what this dependency is used for."
    )
    tags: list[str] = Field(
        default_factory=list,
        description=(
            "A list of categories describing the role of this dependency. "
            "Multiple tags may apply simultaneously to reflect different aspects, "
            "such as web framework, security, database interaction, cloud storage, etc."
        ),
    )


class ProjectInformation(BaseModel):
    project_dir: str = Field(
        title="Project directory", description="Project folder for analysis"
    )
    language: Literal[
        "cpp",
        "c",
        "go",
        "rust",
        "python",
        "java",
        "js",
        "ts",
        "php",
        "ruby",
        "scala",
        "kotlin",
    ] = Field(
        title="Programming language",
        description="The selected programming language for implementing the project",
    )
    framework: str = Field(
        title="Framework",
        description="The framework selected for the project's implementation",
    )

    dependencies: list[DependencyInformation] = Field(
        default_factory=list, description="list of project dependencies"
    )
