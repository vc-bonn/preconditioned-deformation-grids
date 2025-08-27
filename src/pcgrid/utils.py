import ast


def validate_dict(d: dict, required_keys: set):
    missing = required_keys - d.keys()
    if missing:
        raise ValueError(f"Missing required keys: {missing}")


def list_class_names(file_path):
    with open(file_path, "r") as file:
        node = ast.parse(file.read(), filename=file_path)
    return [n.name for n in ast.walk(node) if isinstance(n, ast.ClassDef)]
