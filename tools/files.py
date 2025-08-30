import os
import pathlib

def get_project_tree(project_dir: str, depth: int|None = None, include_files: bool = False) -> dict:
    if not os.path.exists(project_dir):
        return {
            "error": f"path {project_dir} does not exist"
        }
    if not os.path.isdir(project_dir):
        return {
            "error": f"{project_dir} is not a directory"
        }
    
    base_path = pathlib.Path(project_dir)

    def build_tree(path: pathlib.Path, current_depth: int) -> dict:
        if depth is not None and current_depth > depth:
            return {}

        tree = {"type": "directory", "name": path.name, "children": []}

        try:
            for entry in path.iterdir():
                if entry.is_dir():
                    tree["children"].append(build_tree(entry, current_depth + 1))
                elif include_files:
                    tree["children"].append({
                        "type": "file",
                        "name": entry.name
                    })
        except PermissionError:
            tree["children"].append({
                "type": "error",
                "name": f"Permission denied: {path}"
            })

        return tree

    return build_tree(base_path, 1)

def project_tree_to_str(tree: dict, indent: str = "", is_last: bool = True) -> str:
    lines = []

    if "error" in tree:
        return tree["error"]

    prefix = "└── " if is_last else "├── "
    lines.append(indent + prefix + tree["name"])

    if "children" in tree:
        child_indent = indent + ("    " if is_last else "│   ")
        for i, child in enumerate(tree["children"]):
            lines.append(project_tree_to_str(child, child_indent, i == len(tree["children"]) - 1))

    return "\n".join(lines)

def get_project_tree_tool(project_dir:str, depth: int|None=None) -> dict[str, str]:
    project_tree = get_project_tree(project_dir, depth, include_files=True)
    return {
        "result" : project_tree_to_str(project_tree)
    }

def get_project_dirs_tool(project_dir:str, depth: int|None=None) -> dict[str, str]:
    """Get project dirs structure as a tree"""
    project_tree = get_project_tree(project_dir, depth, include_files=False)
    return {
        "result" : project_tree_to_str(project_tree)
    }
    
def get_project_files_tool(tree: dict) -> list[str]:
    if "error" in tree:
        return []

    if tree.get("type") != "directory":
        return []

    files = []
    for child in tree.get("children", []):
        if child.get("type") == "file":
            files.append(child["name"])
    return files

def get_project_extensions_tool(project_dir:str)->dict[str, str]:
    project_tree = get_project_tree(project_dir, depth=None, include_files=True)
    project_files = get_project_files_tool(project_tree)