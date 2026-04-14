import yaml
from pathlib import Path

class IncludeLoader(yaml.SafeLoader):
    def __init__(self, stream):
        self._root = Path(stream.name).parent if hasattr(stream, 'name') else Path.cwd()
        super().__init__(stream)

def include_constructor(loader, node):
    filename = Path(loader._root) / node.value
    with open(filename, 'r') as f:
        return yaml.load(f, IncludeLoader)

# Đăng ký tag
IncludeLoader.add_constructor('!include', include_constructor)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.load(f, IncludeLoader)