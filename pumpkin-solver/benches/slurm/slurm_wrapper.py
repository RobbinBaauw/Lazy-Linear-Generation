#!/usr/bin/env python3

import subprocess
import sys
import time
from pathlib import Path

from experiments import Experiment

if __name__ == "__main__":
    _, experiment_log, array_idx, batch_idx, run_idx = sys.argv
    experiment = Experiment.from_json_file(experiment_log)
    run = experiment.arrays[int(array_idx)].batches[int(batch_idx)][int(run_idx)]

    p = subprocess.Popen([run.program, run.input, *run.arguments],
                         cwd=run.dir,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         )

    start_time = time.time()
    stdout, stderr = p.communicate()
    end_time = time.time()

    experiment_dir = Path(run.dir)

    (experiment_dir / "stdout").open('wb+').write(stdout)
    (experiment_dir / "stderr").open('wb+').write(stderr)

    metrics = f"time:{end_time - start_time}\nexit_code:{p.returncode}"
    (experiment_dir / "metrics").open('w+').write(metrics)

    program_config = experiment.config.program[run.program_name]
    if program_config.exec_cmd_after is not None:
        p = subprocess.Popen(program_config.exec_cmd_after,
                             cwd=run.dir,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE
                             )

        stdout, stderr = p.communicate()
        (experiment_dir / "stdout").open('ab').write(stdout)
        (experiment_dir / "stderr").open('ab').write(stderr)
