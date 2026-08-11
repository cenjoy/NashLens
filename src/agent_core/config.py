"""Minimal standalone shim of the original project's `agent_core.config`.

The full NashLens experiment code (`poker_experiment.py`, `chart_generator.py`)
only references `config.WORKSPACE_ROOT` — the directory under which it writes
result JSON and figures. In this open-source release we default it to the
`results/` and `figures/` folders next to the source. Override with the
NASHLENS_WORKSPACE environment variable if you want outputs elsewhere.
"""
import os

# Default workspace = the release root (parent of src/), so the experiment
# writes into ../results and ../figures relative to this file.
_HERE = os.path.dirname(os.path.abspath(__file__))
_RELEASE_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

WORKSPACE_ROOT = os.getenv(
    "NASHLENS_WORKSPACE", os.path.join(_RELEASE_ROOT, "results")
)
os.makedirs(WORKSPACE_ROOT, exist_ok=True)
