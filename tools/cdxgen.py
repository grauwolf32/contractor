import json
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Any


def ensure_cdxgen() -> str | None:
    if shutil.which("cdxgen") is None:
        if shutil.which("node") is None:
            return "NodeJS needs to be installed to use this tool"

        if shutil.which("npm") is None:
            return "npm needs to be installed to use this tool"

        return_code = subprocess.run(
            ["npm", "install", "-g", "@cyclonedx/cdxgen"]
        ).returncode

        if return_code != 0:
            return "could not install cyclonedx"

        return


def cdxgen_run(project_absolute_path: str) -> dict[str, Any]:
    tmpfd, tmp = tempfile.mkstemp(suffix=".json")
    subprocess.run(
        ["cdxgen", "--profile", "research", project_absolute_path, "-o", tmp],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    sbom: dict[str, Any] = dict()
    with os.fdopen(tmpfd) as f:
        sbom = json.load(f)

    os.remove(tmp)
    return sbom


def filter_cdxgen_output(sbom: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "services": sbom.get("services", []),
        "dependencies": sbom.get("dependencies", []),
        "components": [],
    }

    components = sbom.get("components")
    for component in components:
        filtered_component = {
            k: component.get(k, "")
            for k in {"name", "version", "purl", "description", "type"}
        }
        filtered_component["tags"] = component.get("tags", [])

        output["components"].append(filtered_component)

    return output


def cdxgen_tool(project_absolute_path: str) -> dict:
    """Build list of project dependencies
    Args:
        project_absolute_path: str
            Absolute path to the project

    Returns:
        dict: A dictionary containing the weather information with a 'status' key ('success' or 'error') and a 'report' key with the weather details if successful, or an 'error_message' if an error occurred.
    """

    if err := ensure_cdxgen() is not None:
        return {"error": err}

    if not os.path.exists(project_absolute_path):
        return {"error": f"path {project_absolute_path} does not exists"}

    sbom: dict[str, Any] = cdxgen_run(project_absolute_path)
    return filter_cdxgen_output(sbom)


def cdxgen_mock_tool(project_absolute_path: str) -> dict:
    """Build list of project dependencies
    Args:
        project_absolute_path: str
            Absolute path to the project

    Returns:
        dict: A dictionary containing the weather information with a 'status' key ('success' or 'error') and a 'report' key with the weather details if successful, or an 'error_message' if an error occurred.
    """

    TEST_DATA_DIR = str(
        pathlib.Path(__file__).parent.parent / "data" / "tests" / "cdxgen"
    )
    mock_file: str = os.path.join(TEST_DATA_DIR, "bom.json")
    mock_data: dict[str, Any] = dict()

    with open(mock_file) as f:
        mock_data = json.load(f)

    return mock_data
