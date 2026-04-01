from .reference_source_loader import SonicReferenceData, load_reference_source
from .sonic_reference_exporter import SonicReferenceExporter
from .sonic_zmq_manager_client import SonicZmqManagerControl, SonicZmqManagerControlConfig

__all__ = [
    "SonicReferenceExporter",
    "SonicReferenceData",
    "load_reference_source",
    "SonicZmqManagerControl",
    "SonicZmqManagerControlConfig",
]

# Optional import: DDS bridge depends on IsaacLab/Unitree runtime stack.
# Keep reference export and other lightweight utilities usable even when that stack
# (e.g. pxr/Isaac Sim) is not available in the current Python environment.
try:  # pragma: no cover - optional runtime dependency
    from .unitree_dds_bridge import SonicDdsBridge, SonicDdsBridgeConfig
except Exception:
    pass
else:
    __all__.extend([
        "SonicDdsBridge",
        "SonicDdsBridgeConfig",
    ])
