"""Select one pinned offline wheel compatible with the actual Python runtime."""

from pathlib import Path

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename


def compatible_rdkit_wheel(directory, version, tags=None):
    supported = set(sys_tags() if tags is None else tags)
    compatible = []
    for path in Path(directory).glob("rdkit*.whl"):
        name, wheel_version, _, wheel_tags = parse_wheel_filename(path.name)
        if name == "rdkit" and str(wheel_version) == version and wheel_tags & supported:
            compatible.append(path)
    if len(compatible) != 1:
        raise ValueError(
            f"Expected one compatible RDKit{version} wheel, found{len(compatible)}"
        )
    return compatible[0]
