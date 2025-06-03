import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta, datetime
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict

from config import ExperimentConfig
from experiments import Experiment

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'


@dataclass
class SingularJobId:
    job_id: int
    batch_idx: int
    run_idx: Optional[int]


@dataclass
class MultipleJobId:
    job_id: int
    batch_start_idx: int
    batch_end_idx: int


@dataclass
class JobStatus:
    job_id: Union[SingularJobId, MultipleJobId]
    job_id_raw: str

    alloc_nodes: int
    node_list: str
    alloc_cpus: int

    elapsed_time: timedelta

    state: str
    exit_code: str


def parse_status(status: str) -> Optional[JobStatus]:
    job_id, job_id_raw, alloc_nodes, node_list, alloc_cpus, elapsed, state, exit_code = status.split("|")

    job_step_id = re.search(r"^(\d+)_(\d+)\.(\d+)$", job_id)
    job_array_id = re.search(r"^(\d+)_\[(\d+)-(\d+)]$", job_id)
    job_id = re.search(r"^(\d+)_(\d+)$", job_id)
    if (job_step_id is None) and (job_array_id is None) and (job_id is None):
        return None

    if job_array_id is not None:
        job_id = MultipleJobId(
            int(job_array_id.group(1)),
            int(job_array_id.group(2)),
            int(job_array_id.group(3))
        )
    elif job_step_id is not None:
        job_id = SingularJobId(
            int(job_step_id.group(1)),
            int(job_step_id.group(2)),
            int(job_step_id.group(3))
        )
    elif job_id is not None and int(alloc_cpus) == 1:
        job_id = SingularJobId(
            int(job_id.group(1)),
            int(job_id.group(2)),
            int(0)
        )
    else:
        print(f"Unexpected case {job_id}!")
        return None

    elapsed_time = datetime.strptime(elapsed, "%H:%M:%S")
    elapsed_time = timedelta(hours=elapsed_time.hour, minutes=elapsed_time.minute, seconds=elapsed_time.second)

    return JobStatus(
        job_id,
        job_id_raw,

        int(alloc_nodes),
        node_list,
        int(alloc_cpus),

        elapsed_time,

        state,
        exit_code
    )


def check_status(args):
    config = ExperimentConfig.from_toml_file(args.config)

    experiment_dir = Path(config.experiment_dir)

    experiment_logs = {e.exp_id: e for e in
                       map(lambda e: Experiment.from_json_file(e), experiment_dir.glob("*.json"))}

    if len(experiment_logs.values()) == 0:
        print("No experiments found")
        return

    job_id_exp_id = {
        array.job_id: exp.exp_id
        for exp in experiment_logs.values()
        for array in exp.arrays.values()
        if array.job_id is not None
    }

    job_ids_str = ",".join([str(job_id) for job_id in job_id_exp_id.keys()])

    cmd = ["sacct",
           "--jobs", job_ids_str,
           "--format=JobID,JobIDRaw,AllocNodes,NodeList,AllocCPUS,Elapsed,State,ExitCode",
           "--parsable2", "--noheader"
           ]

    print(f"{' '.join(cmd)}\n")

    job_statuses = subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()

    exp_statuses: Dict[int, List[JobStatus]] = defaultdict(list)
    for status in map(parse_status, job_statuses):
        if status is not None:
            job_id = status.job_id.job_id
            exp_id = job_id_exp_id[job_id]
            exp_statuses[exp_id].append(status)

    for (exp_id, statuses) in exp_statuses.items():
        exp = experiment_logs[exp_id]

        total_runs = len([
            run
            for array in exp.arrays.values()
            for batch in array.batches.values()
            for run in batch.values()
        ])

        completed_statuses = len([status for status in statuses if status.state == "COMPLETED"])

        job_failed_states = ["FAILED", "OUT_OF_MEMORY", "TIMEOUT"]
        slurm_failed_states = ["BOOT_FAIL", "CANCELLED", "DEADLINE", "NODE_FAIL", "PREEMPTED", "SUSPENDED", "REVOKED"]

        failed_statuses = [
            status
            for status in statuses
            if (status.state in slurm_failed_states) or (status.state in job_failed_states)
        ]

        oom = len([s for s in failed_statuses if s.state == "OUT_OF_MEMORY"])
        timeout = len([s for s in failed_statuses if s.state == "TIMEOUT"])
        failed = len([s for s in failed_statuses if s.state == "FAILED"])
        slurm = len([s for s in failed_statuses if s.state in slurm_failed_states])

        failed_status_msg = f"{oom} OOM, {timeout} timeout, {failed} errors, {slurm} slurm errors"

        running_statuses = len([status for status in statuses if status.state == "RUNNING"])

        pending_statuses_single = len([
            status
            for status in statuses
            if status.state == "PENDING" and isinstance(status.job_id, SingularJobId)
        ])
        pending_statuses_batch = sum([
            status.job_id.batch_end_idx - status.job_id.batch_start_idx + 1
            for status in statuses
            if status.state == "PENDING" and isinstance(status.job_id, MultipleJobId)
        ])

        message = f"""{HEADER}{BOLD}Experiment {exp_id}:{ENDC}
{OKBLUE}{BOLD}- Runs scheduled:{ENDC} {total_runs}
{OKBLUE}{BOLD}- Runs success/fail:{ENDC} {OKGREEN}{completed_statuses}{ENDC} / {FAIL}{failed_status_msg}{ENDC}
{OKBLUE}{BOLD}- Runs running:{ENDC} {WARNING}{running_statuses}{ENDC}
{OKBLUE}{BOLD}- Runs pending:{ENDC} 
    * {pending_statuses_single} single runs
    * ~{pending_statuses_batch * config.resource_limits.batch_size} runs in batches ({pending_statuses_batch} batches of size {exp.config.resource_limits.batch_size})
{ENDC}"""
        print(message)
