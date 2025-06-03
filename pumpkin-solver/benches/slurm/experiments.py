import dataclasses
import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, TypeVar, Optional

from config import ExperimentConfig

# Directory structure:
# experiments/{exp_id}/{input_name}/{file_idx}/{program_name}
FilePath = str


@dataclass
class Run:
    dir: FilePath
    input: FilePath
    program: FilePath
    program_name: str
    arguments: List[str]


@dataclass
class Array:
    job_id: Optional[int]
    batches: Dict[int, Dict[int, Run]]


@dataclass
class Experiment:
    exp_id: int
    config: ExperimentConfig

    # Dict[arr_idx, Dict[batch_idx, Dict[run_idx, Run]]]
    arrays: Dict[int, Array]

    def get_log_file_path(self, experiment_dir: Path):
        return experiment_dir / f"{self.exp_id}.json"

    def to_json_file(self, experiment_dir: Path):
        experiment_log = self.get_log_file_path(experiment_dir)
        with open(experiment_log, "w") as f:
            json.dump(self, f, cls=DataClassEncoder)

    @classmethod
    def from_json_file(cls, file_name: str):
        with open(file_name, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data):
        return cls(
            exp_id=data["exp_id"],
            config=ExperimentConfig.from_dict(data["config"]),
            arrays={
                int(arr_idx): Array(
                    int(arr_data["job_id"]) if arr_data["job_id"] is not None else None,
                    {
                        int(batch_idx): {
                            int(run_idx): Run(**run_data)
                            for run_idx, run_data in batch_data.items()
                        }
                        for batch_idx, batch_data in arr_data["batches"].items()
                    }
                )
                for arr_idx, arr_data in data["arrays"].items()
            },
        )


class DataClassEncoder(json.JSONEncoder):
    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return json.JSONEncoder.default(self, obj)


def get_next_experiment_id(experiment_dir: Path):
    experiment_dir.mkdir(parents=True, exist_ok=True)
    existing_ids = [int(re.search("(\d+).json", str(f)).group(1))
                    for f in experiment_dir.glob("*.json")]
    return max(existing_ids, default=0) + 1


T = TypeVar("T")


def chunk(arr: List[T], chunk_size: int) -> List[List[T]]:
    return [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]


def generate_run_dict(runs: List[Run], config: ExperimentConfig) -> "Experiment.arrays":
    batch_size = config.resource_limits.batch_size
    batches = chunk(runs, batch_size)

    # The last batch could not be full
    last_batch = None
    if len(batches[0]) != len(batches[-1]):
        batches, last_batch = batches[:-1], batches[-1]

    array_size = config.resource_limits.array_size
    arrays = chunk(batches, array_size)

    if last_batch is not None:
        arrays.append([last_batch])

    runs = {}
    for (arr_idx, array) in enumerate(arrays):
        runs[arr_idx] = Array(None, {})

        for (batch_idx, batch) in enumerate(array):
            runs[arr_idx].batches[batch_idx] = {}

            for (run_idx, run) in enumerate(batch):
                runs[arr_idx].batches[batch_idx][run_idx] = run

    return runs


def prepare_new_experiment(config: ExperimentConfig) -> Experiment:
    experiment_dir = Path(config.experiment_dir).resolve()

    exp_id = get_next_experiment_id(experiment_dir)

    base_dir = experiment_dir / str(exp_id)
    base_dir.mkdir(parents=True)

    runs = []
    for (input_name, input_path) in config.input.items():
        input_files = glob.glob(input_path.path, recursive=True)

        for (input_file_idx, input_file_name) in enumerate(input_files):
            for (program_name, program) in config.program.items():
                run_dir = base_dir / input_name / str(input_file_idx) / program_name

                # print(f"Creating run_dir: {run_dir}")
                run_dir.mkdir(parents=True)

                runs.append(Run(
                    str(run_dir.resolve()),
                    str(Path(input_file_name).resolve()),
                    str(Path(program.binary).resolve()),
                    program_name,
                    program.arguments
                ))

    if len(runs) == 0:
        print(f"No program x input combinations found!")
        exit(1)

    experiment = Experiment(exp_id, config, generate_run_dict(runs, config))
    experiment.to_json_file(experiment_dir)

    return experiment
