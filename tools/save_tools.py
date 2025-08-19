import os
import pathlib
from agents.project_information_agent.models import ProjectInformation

def save_project_information(pd: ProjectInformation) -> dict[str, str]:
    """Save dependencies in JSON format with respect to provided JSON Schema
    Args:
        dependencies (ProjectInformation) :Project dependencies
    """

    try:
        pd = ProjectInformation.model_validate(pd)
    except Exception as e:
        return {
            "error" : str(e)
        }

    for i in range(len(pd.dependencies)):
        pd.dependencies[i].tags = list(set(pd.dependencies[i].tags))

    TEMP_DIR = str(pathlib.Path(__file__).parent.parent / "data" / "outputs" )

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR, exist_ok=True)

    with open(os.path.join(TEMP_DIR, "project_dependencies.json"), "w") as f:
        f.write(pd.model_dump_json())

    return {"success" : "true"}