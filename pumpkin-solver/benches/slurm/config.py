from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import tomllib
except ModuleNotFoundError:
    import pip._vendor.tomli as tomllib

PathStr = str


@dataclass
class SlurmConfig:
    experiment_name: str
    partition: str
    account: str


@dataclass
class ResourceLimits:
    # "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
    time_limit_minutes: int
    cpus: int
    mem_per_cpu: int
    batch_size: int
    array_size: int


@dataclass
class Program:
    binary: PathStr
    arguments: List[str]
    exec_cmd_after: Optional[List[str]] = None


@dataclass
class Input:
    path: PathStr


@dataclass
class ExperimentConfig:
    experiment_dir: PathStr
    slurm: SlurmConfig
    resource_limits: ResourceLimits
    program: Dict[str, Program]
    input: Dict[str, Input]

    @classmethod
    def from_toml_file(cls, file_name: str):
        with open(file_name, "rb") as f:
            data = tomllib.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data):
        return cls(
            experiment_dir=data["experiment_dir"],
            slurm=SlurmConfig(**data["slurm"]),
            resource_limits=ResourceLimits(**data["resource_limits"]),
            program={k: Program(**v) for k, v in data["program"].items()},
            input={k: Input(**v) for k, v in data["input"].items()},
        )
