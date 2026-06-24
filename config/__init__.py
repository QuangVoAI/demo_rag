"""Bridge Django's ``config`` package to the assistant runtime config.

The room-assistant modules historically import values with ``from config
import ...`` while Django also owns a top-level ``config`` package. Re-exporting
the runtime config here keeps those imports stable in both Django and tests.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_runtime_config_path = Path(__file__).resolve().parent.parent / "python" / "config.py"
_spec = importlib.util.spec_from_file_location("_nhatrovn_runtime_config", _runtime_config_path)

if _spec and _spec.loader:
    _runtime_config = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_runtime_config)
    for _name in dir(_runtime_config):
        if not _name.startswith("_"):
            globals()[_name] = getattr(_runtime_config, _name)

    def validate_runtime_config(strict: bool | None = None) -> dict[str, list[str]]:
        for _name, _value in globals().items():
            if _name.isupper():
                setattr(_runtime_config, _name, _value)
        return _runtime_config.validate_runtime_config(strict=strict)
