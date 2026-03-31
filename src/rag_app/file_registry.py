import hashlib
import json
import os

REGISTRY_FILE = "file_registry.json"


def get_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def load_registry():
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}


def save_registry(registry):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def file_exists(file_path):
    registry = load_registry()
    file_hash = get_file_hash(file_path)
    return file_hash in registry


def register_file(file_path, file_name):
    registry = load_registry()
    file_hash = get_file_hash(file_path)

    registry[file_hash] = {
        "file_name": file_name
    }

    save_registry(registry)


def remove_file(file_name):
    registry = load_registry()
    removed = False

    for file_hash, entry in list(registry.items()):
        if entry.get("file_name") == file_name:
            del registry[file_hash]
            removed = True

    if removed:
        save_registry(registry)

    return removed


def clear_registry():
    save_registry({})