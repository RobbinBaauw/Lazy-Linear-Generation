from collections import defaultdict
from dataclasses import dataclass, field
from textwrap import wrap
from typing import BinaryIO, Callable, Tuple, Dict, List, Any, Optional, Set

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, ticker

from parse_output_files import RunResult, generate_results, RunData

Program = str
InstanceImprovements = Dict[str, Dict[Program, float]]
ProgramData = Dict[Program, np.array]


def get_improvement_in_metric(
        results_parsed_pkl: BinaryIO,
        metric_getter: Callable[[RunResult], float],
        baseline_program: str,
        compared_programs: Set[str],
        instance_filter: Callable[[str], bool] = lambda r: True,
        result_filter: Callable[[RunResult], bool] = lambda r: True,
) -> Tuple[InstanceImprovements, ProgramData]:
    per_instance: InstanceImprovements = {}
    per_program: ProgramData = defaultdict(list)

    skipped, total = 0, 0
    for (instance, program_results) in generate_results(results_parsed_pkl):
        if not instance_filter(instance):
            continue

        program_metrics = {}
        for ((_, program_name), run_result) in program_results.items():
            if program_name not in compared_programs:
                continue

            if (run_result is None) or (run_result.run_data is None):
                continue

            if run_result.failed():
                continue

            if result_filter(run_result):
                program_metrics[program_name] = metric_getter(run_result)

        total += 1
        if len(set(compared_programs) - set(program_metrics.keys())) > 0:
            skipped += 1
            continue

        baseline_metric = program_metrics[baseline_program]

        instance_improvement = {
            program: metric / baseline_metric
            for (program, metric) in program_metrics.items()
        }
        per_instance[instance] = instance_improvement

        for (program, metric) in program_metrics.items():
            per_program[program].append(instance_improvement[program])

    print(f"Skipped/total: {skipped}/{total}")

    per_program = {program: np.array(values)
                   for (program, values) in per_program.items()}

    return per_instance, per_program


@dataclass
class BoxplotConfig:
    data: ProgramData
    name: str
    show_outliers: bool
    show_quartiles: bool = True
    y_lim: Optional[Tuple[Optional[float], Optional[float]]] = None
    major_ticks: List[float] = field(default_factory=lambda: [1.0])
    quartile_bbox: Optional[Dict] = None
    args: Optional[Any] = None


def improvements_boxplots(data: List[BoxplotConfig],
                          skip_programs: List[str] = [],
                          titles: Dict[str, str] = {},
                          show_title: bool = True,
                          palette = list(sns.color_palette())):
    programs = list(filter(lambda p: p not in skip_programs, titles.keys()))

    fig, ax = plt.subplots(len(data), len(programs), sharey='row', figsize=(8 * len(programs), 3))
    if len(programs) == 1:
        if len(data) == 1:
            ax = np.array([ax])
        else:
            ax = np.array(list(map(lambda x: [x], ax)))

    if len(data) == 1:
        ax = np.array([ax])

    for j, program in enumerate(programs):
        if show_title:
            ax[0, j].set_title(titles.get(program, program), loc='center')

        for i, plot_conf in enumerate(data):
            ax_curr = ax[i, j]

            program_data = plot_conf.data[program]
            quartiles = list(np.percentile(program_data, [25, 50, 75]))

            bp = sns.boxplot(program_data,
                             ax=ax_curr,
                             showfliers=plot_conf.show_outliers,
                             color=palette[j % len(palette)],
                             **({} if plot_conf.args is None else plot_conf.args))

            ax_curr.set_yscale("log", base=2)
            ax_curr.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax_curr.yaxis.set_major_locator(ticker.LogLocator(base=2.0, subs=plot_conf.major_ticks))
            ax_curr.yaxis.set_minor_locator(ticker.LogLocator(base=2.0, subs=[0.5]))
            ax_curr.set_ylim(plot_conf.y_lim)
            ax_curr.tick_params(axis='both', which='major', labelsize=18)

            if plot_conf.show_quartiles:
                def plot_quartile(j: int):
                    bbox = dict(facecolor='#00a6d6', alpha=1, linewidth=0)
                    bbox |= {} if plot_conf.quartile_bbox is None else plot_conf.quartile_bbox
                    bp.text(-0.25 + 0.25 * j, quartiles[j], f"Q{j + 1}: {quartiles[j]:.3f}",
                            horizontalalignment='center', verticalalignment='center',
                            color='w', weight='bold', bbox=bbox, size=20)

                plot_quartile(0)
                plot_quartile(1)
                plot_quartile(2)

    for i, plot_conf in enumerate(data):
        ax[i, 0].set_ylabel(plot_conf.name, labelpad=10, fontsize=18)

    fig.tight_layout()


@dataclass
class ScatterplotConfig:
    data: ProgramData
    name: str
    args: Optional[Any] = None


def improvements_scatterplots(data: List[ScatterplotConfig], skip_programs: List[str] = [],
                              titles: Dict[str, str] = {}):
    programs = list(filter(lambda p: p not in skip_programs, data[0].data.keys()))

    fig, ax = plt.subplots(1, len(programs), figsize=(8, 6))

    for i, program in enumerate(programs):
        ax_curr = ax[i]
        ax_curr.set_title(titles.get(program, program), loc='center')

        quartiles_0 = list(np.percentile(data[0].data[program], [5, 95]))
        idx_0 = np.where((data[0].data[program] >= quartiles_0[0]) & (data[0].data[program] <= quartiles_0[1]))

        quartiles_1 = list(np.percentile(data[1].data[program], [5, 95]))
        idx_1 = np.where((data[1].data[program] >= quartiles_1[0]) & (data[1].data[program] <= quartiles_1[1]))

        idx = np.intersect1d(idx_0, idx_1)

        sns.scatterplot(x=data[0].data[program][idx], y=data[1].data[program][idx], ax=ax_curr, *data.args)
        ax_curr.set_xlabel("fraction conflicts")
        ax_curr.set_ylabel("fraction walltime")

    fig.tight_layout()
    plt.show()
