"""
Device Manager.

Probes and tracks compute hardware (CPU, GPU, RAM) for device-aware
model placement and migration decisions.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GpuInfo:
    """Information about a single GPU."""

    index: int
    name: str
    vendor: str  # "nvidia", "amd", "apple"
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    compute_capability: str | None = None
    driver_version: str | None = None


@dataclass
class CpuInfo:
    """Information about the CPU."""

    architecture: str  # "x86_64", "aarch64"
    cores_physical: int
    cores_logical: int
    brand: str


@dataclass
class SystemInfo:
    """Full hardware snapshot."""

    cpu: CpuInfo
    ram_total_mb: int
    ram_available_mb: int
    gpus: list[GpuInfo]
    platform: str  # "linux", "darwin", "windows"


@dataclass
class DeviceCapability:
    """What a device can do and how much memory it has."""

    device_id: str  # "cpu", "cuda:0", "cuda:1", "mps"
    can_train: bool
    can_infer: bool
    memory_total_mb: int
    memory_free_mb: int
    vendor: str  # "cpu", "nvidia", "amd", "apple"


class DeviceManager:
    """Probes and tracks compute hardware."""

    def __init__(self) -> None:
        self._system_info: SystemInfo | None = None
        self._capabilities: list[DeviceCapability] = []

    def probe(self) -> SystemInfo:
        """Run hardware detection. Call once at sidecar startup."""
        cpu = self._probe_cpu()
        ram_total, ram_avail = self._probe_ram()
        gpus = self._probe_gpus()

        self._system_info = SystemInfo(
            cpu=cpu,
            ram_total_mb=ram_total,
            ram_available_mb=ram_avail,
            gpus=gpus,
            platform=platform.system().lower(),
        )
        self._capabilities = self._build_capabilities()

        logger.info(
            "Device probe complete: %d GPU(s), %d MB RAM total",
            len(gpus),
            ram_total,
        )
        for gpu in gpus:
            logger.info(
                "  GPU %d: %s (%s) — %d MB VRAM",
                gpu.index,
                gpu.name,
                gpu.vendor,
                gpu.vram_total_mb,
            )

        return self._system_info

    def refresh_memory(self) -> dict[str, dict[str, int]]:
        """Re-poll memory usage for all devices. Cheap, can call frequently."""
        result: dict[str, dict[str, int]] = {}

        # RAM
        try:
            import psutil

            mem = psutil.virtual_memory()
            result["cpu"] = {
                "total_mb": mem.total // (1024 * 1024),
                "used_mb": mem.used // (1024 * 1024),
                "free_mb": mem.available // (1024 * 1024),
            }
        except ImportError:
            pass

        # CUDA GPUs
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    total = torch.cuda.get_device_properties(i).total_mem
                    reserved = torch.cuda.memory_reserved(i)
                    allocated = torch.cuda.memory_allocated(i)
                    free = total - reserved
                    result[f"cuda:{i}"] = {
                        "total_mb": total // (1024 * 1024),
                        "used_mb": allocated // (1024 * 1024),
                        "free_mb": free // (1024 * 1024),
                    }
        except ImportError:
            pass

        # Update system info if we have it
        if self._system_info:
            if "cpu" in result:
                self._system_info.ram_available_mb = result["cpu"]["free_mb"]
            for gpu in self._system_info.gpus:
                key = f"cuda:{gpu.index}" if gpu.vendor == "nvidia" else "mps"
                if key in result:
                    gpu.vram_used_mb = result[key]["used_mb"]
                    gpu.vram_free_mb = result[key]["free_mb"]

            # Rebuild capabilities with updated memory
            self._capabilities = self._build_capabilities()

        return result

    def get_system_info(self) -> SystemInfo | None:
        return self._system_info

    def get_capabilities(self) -> list[DeviceCapability]:
        return self._capabilities

    def best_device_for(self, task: str, model_size_mb: int) -> str:
        """
        Recommend a device for a given task and model size.

        Args:
            task: "inference" or "training"
            model_size_mb: Estimated model memory footprint in MB

        Returns:
            Device ID string, e.g. "cuda:0", "mps", "cpu"
        """
        if not self._capabilities:
            return "cpu"

        # Filter to devices that can handle the task
        candidates = []
        for cap in self._capabilities:
            if task == "training" and not cap.can_train:
                continue
            if not cap.can_infer:
                continue
            candidates.append(cap)

        if not candidates:
            return "cpu"

        # Prefer GPU with enough free memory (with 20% headroom)
        required = int(model_size_mb * 1.2)
        gpu_candidates = [
            c for c in candidates if c.vendor != "cpu" and c.memory_free_mb >= required
        ]

        if gpu_candidates:
            # Pick GPU with most free memory
            best = max(gpu_candidates, key=lambda c: c.memory_free_mb)
            return best.device_id

        # Fall back to CPU
        return "cpu"

    def to_dict(self) -> dict[str, Any]:
        """Serialize system info and capabilities for IPC response."""
        info = self._system_info
        if info is None:
            return {"error": "Device probe not yet run"}

        return {
            "cpu": {
                "architecture": info.cpu.architecture,
                "cores_physical": info.cpu.cores_physical,
                "cores_logical": info.cpu.cores_logical,
                "brand": info.cpu.brand,
            },
            "ram_total_mb": info.ram_total_mb,
            "ram_available_mb": info.ram_available_mb,
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "vendor": g.vendor,
                    "vram_total_mb": g.vram_total_mb,
                    "vram_used_mb": g.vram_used_mb,
                    "vram_free_mb": g.vram_free_mb,
                    "compute_capability": g.compute_capability,
                }
                for g in info.gpus
            ],
            "platform": info.platform,
            "capabilities": [
                {
                    "device_id": c.device_id,
                    "can_train": c.can_train,
                    "can_infer": c.can_infer,
                    "memory_total_mb": c.memory_total_mb,
                    "memory_free_mb": c.memory_free_mb,
                    "vendor": c.vendor,
                }
                for c in self._capabilities
            ],
        }

    # ── Private probing methods ──────────────────────────────────

    def _probe_cpu(self) -> CpuInfo:
        cores_physical = 1
        cores_logical = 1
        try:
            import psutil

            cores_physical = psutil.cpu_count(logical=False) or 1
            cores_logical = psutil.cpu_count(logical=True) or 1
        except ImportError:
            import os

            cores_logical = os.cpu_count() or 1
            cores_physical = cores_logical

        return CpuInfo(
            architecture=platform.machine(),
            cores_physical=cores_physical,
            cores_logical=cores_logical,
            brand=platform.processor() or "unknown",
        )

    def _probe_ram(self) -> tuple[int, int]:
        try:
            import psutil

            mem = psutil.virtual_memory()
            return mem.total // (1024 * 1024), mem.available // (1024 * 1024)
        except ImportError:
            return 0, 0

    def _probe_gpus(self) -> list[GpuInfo]:
        gpus: list[GpuInfo] = []

        # NVIDIA / CUDA
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    mem_total = props.total_memory // (1024 * 1024)
                    mem_allocated = torch.cuda.memory_allocated(i) // (1024 * 1024)
                    gpus.append(
                        GpuInfo(
                            index=i,
                            name=props.name,
                            vendor="nvidia",
                            vram_total_mb=mem_total,
                            vram_used_mb=mem_allocated,
                            vram_free_mb=mem_total - mem_allocated,
                            compute_capability=f"{props.major}.{props.minor}",
                        )
                    )
        except ImportError:
            pass

        # Apple MPS
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS shares system RAM, so VRAM numbers aren't separately queryable
                gpus.append(
                    GpuInfo(
                        index=0,
                        name="Apple MPS",
                        vendor="apple",
                        vram_total_mb=0,
                        vram_used_mb=0,
                        vram_free_mb=0,
                    )
                )
        except (ImportError, AttributeError):
            pass

        return gpus

    def _build_capabilities(self) -> list[DeviceCapability]:
        caps: list[DeviceCapability] = []
        info = self._system_info
        if info is None:
            return caps

        # CPU is always available
        caps.append(
            DeviceCapability(
                device_id="cpu",
                can_train=True,
                can_infer=True,
                memory_total_mb=info.ram_total_mb,
                memory_free_mb=info.ram_available_mb,
                vendor="cpu",
            )
        )

        for gpu in info.gpus:
            if gpu.vendor == "nvidia":
                device_id = f"cuda:{gpu.index}"
                can_train = True
            elif gpu.vendor == "apple":
                device_id = "mps"
                can_train = False  # MPS training is still experimental
            else:
                device_id = f"gpu:{gpu.index}"
                can_train = False

            caps.append(
                DeviceCapability(
                    device_id=device_id,
                    can_train=can_train,
                    can_infer=True,
                    memory_total_mb=gpu.vram_total_mb,
                    memory_free_mb=gpu.vram_free_mb,
                    vendor=gpu.vendor,
                )
            )

        return caps
