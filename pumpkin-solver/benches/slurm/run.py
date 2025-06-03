import os
import resource
import signal
import subprocess
from multiprocessing.pool import ThreadPool
from pathlib import Path

import psutil

from config import ExperimentConfig
from experiments import prepare_new_experiment, Experiment

LOCAL_CPUS = 20


def schedule_local_batches(experiment: Experiment, config: ExperimentConfig):
    thread_pool = ThreadPool(processes=LOCAL_CPUS)

    experiment_dir = Path(config.experiment_dir).resolve()
    experiment_log = experiment.get_log_file_path(experiment_dir)
    experiment.to_json_file(experiment_dir)

    timeout = config.resource_limits.time_limit_minutes * 60
    max_memory = config.resource_limits.mem_per_cpu * 1024 * 1024

    def schedule_local_run(array_idx: int, batch_idx: int, run_idx: int):
        run = experiment.arrays[array_idx].batches[batch_idx][run_idx]
        print(f"Scheduling {array_idx}/{batch_idx}/{run_idx}: {run.program_name} x {run.input}")

        cmd = [
            str((Path(__file__).parent / "slurm_wrapper.py").resolve()),
            str(experiment_log),
            str(array_idx),
            str(batch_idx),
            str(run_idx)
        ]

        p = subprocess.Popen(cmd,
                             stdin=None,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             )

        psutil.Process(p.pid).rlimit(resource.RLIMIT_AS, (max_memory, max_memory))

        stdout, stderr = None, None
        try:
            stdout, stderr = p.communicate(timeout=timeout, input=None)
            if p.returncode != 0:
                print(f"==> {array_idx}/{batch_idx}/{run_idx} Error! {stderr.decode()} / {stdout.decode()}")
                stdout = None
            else:
                print(f"==> {array_idx}/{batch_idx}/{run_idx} OK! {stdout.decode()}")
                stderr = None
        except subprocess.TimeoutExpired as e:
            print(f"==> {array_idx}/{batch_idx}/{run_idx} Timeout")
            p.kill()
        except Exception as e:
            print(f"==> {array_idx}/{batch_idx}/{run_idx} Error {e}")
            p.kill()
        finally:
            try:
                p.kill()
            except ProcessLookupError:
                # Process already terminated
                pass

        return stdout, stderr

    for (array_idx, array) in experiment.arrays.items():
        for (batch_idx, batch) in array.batches.items():
            for run_idx in batch.keys():
                # schedule_local_run(array_idx, batch_idx, run_idx)
                thread_pool.apply_async(schedule_local_run, (array_idx, batch_idx, run_idx))

    thread_pool.close()
    thread_pool.join()


def schedule_slurm_batches(experiment: Experiment, config: ExperimentConfig):
    slurm_config, resource_config = config.slurm, config.resource_limits

    experiment_dir = Path(config.experiment_dir).resolve()
    experiment_log = experiment.get_log_file_path(experiment_dir)

    stdout_file = experiment_dir / "log_%A_%a.$i.out"
    stderr_file = experiment_dir / "log_%A_%a.$i.err"

    def to_time_string(limit_minutes: int) -> str:
        time_limit_hours = int(limit_minutes / 60)
        time_limit_minutes = limit_minutes % 60
        return f"{time_limit_hours:02d}:{time_limit_minutes:02d}:00"

    for (array_idx, array) in experiment.arrays.items():
        batches = array.batches
        batch_size = len(batches[0].keys())

        common_arguments = f"""#!/bin/bash
#SBATCH --job-name='slurm-experiment'
#SBATCH --array={0}-{len(batches.keys()) - 1}
#SBATCH --ntasks={batch_size}
#SBATCH --partition={slurm_config.partition}
#SBATCH --cpus-per-task={resource_config.cpus}
#SBATCH --mem-per-cpu={resource_config.mem_per_cpu}
#SBATCH --account={slurm_config.account}
"""

        batch_script_1 = f"""{common_arguments}
#SBATCH --time={to_time_string(resource_config.time_limit_minutes)}
#SBATCH --output={stdout_file}
#SBATCH --error={stderr_file}
set -x

i=0
slurm_wrapper.py {experiment_log} {array_idx} $SLURM_ARRAY_TASK_ID 0
"""

        batch_script_more = f"""{common_arguments}
#SBATCH --time={to_time_string(resource_config.time_limit_minutes + 5)}
set -x

i=0
for i in $(seq 0 {batch_size - 1}); do
    srun -c1 -n1 -N1 --exact --mem-per-cpu={resource_config.mem_per_cpu} --time={to_time_string(resource_config.time_limit_minutes)} --output={stdout_file} --error={stderr_file} slurm_wrapper.py {experiment_log} {array_idx} $SLURM_ARRAY_TASK_ID $i &
done
wait"""
        batch_script = batch_script_1 if batch_size == 1 else batch_script_more

        p = subprocess.Popen(["sbatch", "--parsable"],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE
                             )

        print(f"Scheduling batch for experiment {experiment.exp_id}")
        print(batch_script)

        stdout, stderr = p.communicate(input=batch_script.encode())

        job_id = int(stdout.decode().rstrip().split(";")[0])
        array.job_id = job_id

        print(f"Scheduled job!\n{stderr.decode()}")

    print("Writing experiment log again")
    experiment.to_json_file(experiment_dir)


def run(args):
    config = ExperimentConfig.from_toml_file(args.config)
    experiment = prepare_new_experiment(config)

    if args.local:
        schedule_local_batches(experiment, config)
    else:
        schedule_slurm_batches(experiment, config)
