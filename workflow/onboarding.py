import os
import pathlib
import shutil

class ProjectNotFound(Exception):
    ...

def onboard_contractor(project_dir: str):
    if not os.path.exists(project_dir):
        raise ProjectNotFound
    contractor_path = pathlib.Path(project_dir) / ".contractor"

    if not os.path.exists(contractor_path):
        os.makedirs(contractor_path)
    
    return
