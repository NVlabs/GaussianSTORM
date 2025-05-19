import logging
import os
import random
import re
import socket
import sys
from typing import Dict, List

import torch
import torch.distributed as dist

_LOCAL_RANK = -1
_LOCAL_WORLD_SIZE = -1

_TORCH_DISTRIBUTED_ENV_VARS = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
)


logger = logging.getLogger("STORM")


def is_enabled() -> bool:
    """Check if distributed mode is enabled."""
    return dist.is_available() and dist.is_initialized()


def get_global_rank() -> int:
    """Get the global rank of the current process."""
    return dist.get_rank() if is_enabled() else 0


def get_world_size() -> int:
    """Get the world size (number of processes)."""
    return dist.get_world_size() if is_enabled() else 1


def is_main_process() -> bool:
    """Check if the current process is the main process."""
    return get_global_rank() == 0


def _collect_env_vars() -> Dict[str, str]:
    """Collect PyTorch distributed environment variables."""
    env_vars = {
        env_var: os.environ[env_var]
        for env_var in _TORCH_DISTRIBUTED_ENV_VARS
        if env_var in os.environ
    }
    if "WORLD_SIZE" in env_vars and "LOCAL_WORLD_SIZE" not in env_vars:
        env_vars["LOCAL_WORLD_SIZE"] = env_vars["WORLD_SIZE"]
    return env_vars


def _is_slurm_job_process() -> bool:
    """Check if the process is running as part of a Slurm job."""
    return "SLURM_JOB_ID" in os.environ and not os.isatty(sys.stdout.fileno())


def _parse_slurm_node_list(s: str) -> List[str]:
    """Parse a Slurm node list into a list of hostnames."""
    nodes = []
    # Extract "hostname", "hostname[1-2,3,4-5]," substrings
    pattern = re.compile(r"(([^\[]+)(?:\[([^\]]+)\])?),?")
    for match in pattern.finditer(s):
        prefix, suffixes = s[match.start(2) : match.end(2)], s[match.start(3) : match.end(3)]
        for suffix in suffixes.split(","):
            span = suffix.split("-")
            if len(span) == 1:
                nodes.append(prefix + suffix)
            else:
                width = len(span[0])
                start, end = int(span[0]), int(span[1]) + 1
                nodes.extend([prefix + f"{i:0{width}}" for i in range(start, end)])
    return nodes


def _check_env_variable(key: str, new_value: str):
    """Ensure that environment variables are consistent."""
    if key in os.environ and os.environ[key] != new_value:
        raise RuntimeError(f"Environment variable conflict: {key} is already set")


def _restrict_print_to_main_process() -> None:
    """
    This function disables printing when not in the main process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_main_process() or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def _get_master_port(seed: int = 0) -> int:
    """Get a master port, either from the environment or randomly."""
    MIN_MASTER_PORT, MAX_MASTER_PORT = 20_000, 60_000

    master_port_str = os.environ.get("MASTER_PORT")
    if master_port_str is None:
        rng = random.Random(seed)
        return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)

    return int(master_port_str)


def _get_available_port() -> int:
    """Find an available port on the current machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # A "" host address means INADDR_ANY i.e. binding to all interfaces.
        # Note this is not compatible with IPv6.
        s.bind(("", 0))
        return s.getsockname()[1]


class _TorchDistributedEnvironment:
    """Manages PyTorch distributed environment variables."""

    def __init__(self):
        self.master_addr = "127.0.0.1"
        self.master_port = 0
        self.rank = -1
        self.world_size = -1
        self.local_rank = -1
        self.local_world_size = -1

        if _is_slurm_job_process():
            return self._set_from_slurm_env()

        env_vars = _collect_env_vars()
        if not env_vars:
            # Environment is not set
            pass
        elif len(env_vars) == len(_TORCH_DISTRIBUTED_ENV_VARS):
            # Environment is fully set
            return self._set_from_preset_env()
        else:
            # Environment is partially set
            collected_env_vars = ", ".join(env_vars.keys())
            raise RuntimeError(
                f"Partially set environment: {collected_env_vars}."
                f"Unset environment variables: {[env_var for env_var in _TORCH_DISTRIBUTED_ENV_VARS if env_var not in env_vars]}"
            )

        if torch.cuda.device_count() > 0:
            return self._set_from_local()

        raise RuntimeError("Can't initialize PyTorch distributed environment")

    def _set_from_slurm_env(self):
        """Slurm job created with sbatch, submitit, etc..."""
        logger.info("Initializing from Slurm environment")
        job_id = int(os.environ["SLURM_JOB_ID"])
        node_count = int(os.environ["SLURM_JOB_NUM_NODES"])
        nodes = _parse_slurm_node_list(os.environ["SLURM_JOB_NODELIST"])
        assert len(nodes) == node_count, f"SLURM_JOB_NODELIST mismatch: {nodes} vs {node_count}"

        self.master_addr = nodes[0]
        self.master_port = _get_master_port(seed=job_id)
        self.rank = int(os.environ["SLURM_PROCID"])
        self.world_size = int(os.environ["SLURM_NTASKS"])
        logger.info(
            f"Master address: {self.master_addr}, Master port: {self.master_port}, Rank: {self.rank}, World size: {self.world_size}"
        )
        assert self.rank < self.world_size
        self.local_rank = int(os.environ["SLURM_LOCALID"])
        self.local_world_size = self.world_size // node_count
        assert self.local_rank < self.local_world_size

    def _set_from_preset_env(self):
        logger.info("Initialization from preset environment")
        self.master_addr = os.environ["MASTER_ADDR"]
        self.master_port = os.environ["MASTER_PORT"]
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        assert self.rank < self.world_size
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.local_world_size = (
            int(os.environ["LOCAL_WORLD_SIZE"])
            if "LOCAL_WORLD_SIZE" in os.environ
            else self.world_size
        )
        assert self.local_rank < self.local_world_size

    def _set_from_local(self):
        """Single node and GPU job (i.e. local script run)"""
        logger.info("Initializing from local environment")
        self.master_addr = "127.0.0.1"
        self.master_port = _get_available_port()
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.local_world_size = 1

    def export(self, *, overwrite: bool) -> "_TorchDistributedEnvironment":
        """Export the environment variables for distributed initialization."""
        env_vars = {
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
            "RANK": str(self.rank),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_RANK": str(self.local_rank),
            "LOCAL_WORLD_SIZE": str(self.local_world_size),
        }
        if not overwrite:
            for k, v in env_vars.items():
                _check_env_variable(k, v)

        os.environ.update(env_vars)
        return self


def enable(
    *,
    set_cuda_current_device: bool = True,
    overwrite: bool = False,
    allow_nccl_timeout: bool = False,
):
    """Enable distributed mode

    Args:
        set_cuda_current_device: If True, call torch.cuda.set_device() to set the
            current PyTorch CUDA device to the one matching the local rank.
        overwrite: If True, overwrites already set variables. Else fails.
    """

    global _LOCAL_RANK, _LOCAL_WORLD_SIZE
    if _LOCAL_RANK >= 0 or _LOCAL_WORLD_SIZE >= 0:
        raise RuntimeError("Distributed mode already enabled")
    torch_env = _TorchDistributedEnvironment()
    torch_env.export(overwrite=overwrite)

    if set_cuda_current_device:
        torch.cuda.set_device(torch_env.local_rank)

    if allow_nccl_timeout:
        # This allows to use torch distributed timeout in a NCCL backend
        key, value = "NCCL_ASYNC_ERROR_HANDLING", "1"
        if not overwrite:
            _check_env_variable(key, value)
        os.environ[key] = value

    dist.init_process_group(backend="nccl")
    dist.barrier()

    # Finalize setup
    _LOCAL_RANK = torch_env.local_rank
    _LOCAL_WORLD_SIZE = torch_env.local_world_size
    _restrict_print_to_main_process()
