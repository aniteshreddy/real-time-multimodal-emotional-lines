"""Shared utilities: seeding, device selection, parameter accounting."""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from . import config


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(module: torch.nn.Module) -> int:
    """Total learned-parameter count of a torch module (all parameters,
    trainable or frozen: the assignment counts total, not active)."""
    return sum(p.numel() for p in module.parameters())


@dataclass
class BudgetRow:
    component: str
    params: int
    source: str  # "pretrained hf", "pretrained mediapipe", "trained here", "handcrafted (0)"


def parameter_table(rows: Iterable[BudgetRow]) -> list[dict]:
    """Aggregate rows, check the 6B hard budget, and return a JSON-ready table."""
    rows = list(rows)
    total = sum(r.params for r in rows)
    if total > config.PARAM_BUDGET:
        raise RuntimeError(
            f"Parameter budget exceeded: {total:,} > {config.PARAM_BUDGET:,}. "
            "The assignment counts every encoder, head, and generator."
        )
    table = [{"component": r.component, "params": r.params, "source": r.source} for r in rows]
    table.append({"component": "TOTAL", "params": total, "source": f"budget {config.PARAM_BUDGET:,}"})
    return table


def hardware_report() -> dict:
    """Observed hardware facts for the report required by the assignment."""
    report = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(get_device()),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report["gpu"] = {
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1e9, 1),
            "multi_processor_count": props.multi_processor_count,
        }
    import platform  # local import to surface only for report purposes in run dir machine info

    report["machine"] = platform.processor() or platform.machine()
    ram_bytes = None
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    ram_bytes = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    report["ram_total_bytes"] = ram_bytes
    return report


@dataclass
class LatencyLog:
    events: dict = field(default_factory=dict)

    def measure(self, name: str) -> "_LatencyContext":
        return _LatencyContext(self, name)


class _LatencyContext:
    def __init__(self, log: LatencyLog, name: str):
        self.log = log
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.log.events[self.name] = (time.perf_counter() - self.t0) * 1000.0
        return False


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
