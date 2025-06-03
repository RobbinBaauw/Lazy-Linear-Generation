import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

from parse_analysis_log import parse_analysis_log_experiments_if_not_exists
from parse_output_files import parse_experiment_to_file, generate_results, ConstraintType, all_results, Program, parse_instance_run_data_generator
from plot_output import get_improvement_in_metric, improvements_boxplots, BoxplotConfig

DATASETS_DIR = Path(__file__).parent / "datasets"
RESULTS_DIR = Path(__file__).parent / "results"

VARIANTS_RESULTS_DIR = RESULTS_DIR / "variants"
ANALYSIS_LOG_RESULTS_DIR = RESULTS_DIR / "analysis_log"
LINEAR_RESULTS_DIR = RESULTS_DIR / "linear"

VARIANTS_INPUT_PKL = Path('experiment_1_2_7_8_variants.pkl')
EXPLANATION_STATS_PKL = Path('experiment_2_6_explanation_stats.pkl')
ANALYSIS_LOG_NO_LEARNED = Path('analysis_log_parsed.pkl')

EXPERIMENT_2_INPUT_PKL = Path('experiment_2_matching_propagations.pkl')
EXPERIMENT_2_AGG_PKL = Path('experiment_2_matching_propagations.agg.pkl')

EXPERIMENT_3_INPUT_PKL = Path('experiment_3_learning_propagation_over_time.pkl')
EXPERIMENT_3A_AGG_PKL = Path('experiment_3a_fallback_reasons_over_time.agg.pkl')
EXPERIMENT_3B_AGG_PKL = Path('experiment_3b_learning_propagation_over_time.agg.pkl')
EXPERIMENT_3C_AGG_PKL = Path('experiment_3c_slack_in_explanations.agg.pkl')

EXPERIMENT_4_PKL = Path('experiment_4_pumpkin_vs_linear.pkl')


def parse_variants_if_not_exists():
    if not VARIANTS_INPUT_PKL.exists():
        with VARIANTS_INPUT_PKL.open('wb') as f:
            parse_experiment_to_file(VARIANTS_RESULTS_DIR, f)


def parse_analysis_log():
    parse_analysis_log_experiments_if_not_exists(ANALYSIS_LOG_RESULTS_DIR, ANALYSIS_LOG_NO_LEARNED, EXPERIMENT_2_INPUT_PKL, EXPERIMENT_3_INPUT_PKL,
                                                 EXPLANATION_STATS_PKL)


def experiment_0_overall_stats():
    parse_variants_if_not_exists()

    compaed_programs = {"resolution", "llg"}

    total_instances = 0
    for input_dir in VARIANTS_RESULTS_DIR.iterdir():
        if not input_dir.is_dir():
            continue
        total_instances += len(list(input_dir.iterdir()))

    @dataclass
    class ProgramStats:
        timeouts: List[str] = field(default_factory=list)
        oom: List[str] = field(default_factory=list)
        error: List[str] = field(default_factory=list)

        success: List[str] = field(default_factory=list)
        learned_inequality: List[str] = field(default_factory=list)

    program_stats = {Program.LLG: ProgramStats(), Program.RESOLUTION: ProgramStats()}

    with VARIANTS_INPUT_PKL.open('rb') as f:
        for (instance, program_results) in generate_results(f):
            for ((program, program_name), run_result) in program_results.items():
                if program_name not in compaed_programs:
                    continue

                if run_result.timed_out():
                    program_stats[program].timeouts.append(run_result.fzn_file_name)
                    continue

                if run_result.exit_code == -9:
                    program_stats[program].oom.append(run_result.fzn_file_name)
                    continue

                if run_result.failed():
                    program_stats[program].error.append(run_result.fzn_file_name)
                    continue

                program_stats[program].success.append(run_result.fzn_file_name)

                if program == Program.LLG:
                    if run_result.run_data.stats.llg_learned_constraints > 0:
                        program_stats[program].learned_inequality.append(run_result.fzn_file_name)

    def show_stats(program):
        stats = program_stats[program]

        all_instances = stats.success + stats.timeouts + stats.oom + stats.error

        # Emperically: instances that didn't finish without creating any metrics, were all OOMs for other programs
        # So add EXPERIMENT_INSTANCES-len(all_instances) to OOM
        extra_oom = total_instances - len(all_instances)

        print("==========")
        print(f"Program: {program}")
        print(f"Instances total: {total_instances}")
        print(f"Instances OOM: {len(stats.oom) + extra_oom}")
        print(f"Instances timeout: {len(stats.timeouts)}")
        print(f"Instances error: {len(stats.error)}")
        print(f"Instances success: {len(stats.success)}")
        print(f"Instances learned inequalities: {len(stats.learned_inequality)}")
        print("==========")

        return all_instances

    instances_llg = set(show_stats(Program.LLG))
    instances_resolution = set(show_stats(Program.RESOLUTION))

    print(instances_resolution - instances_llg)
    print(instances_llg - instances_resolution)

    print("...")


def experiment_0_analyze_top_instances():
    parse_variants_if_not_exists()

    def compute_instance_improvements():
        with VARIANTS_INPUT_PKL.open('rb') as f:
            program_titles = {
                "llg": "LLG",
                "resolution": "Resolution",
            }

            compared_programs = set(program_titles.keys())

            instance_improvements, _ = get_improvement_in_metric(f,
                                                                 lambda res: max(1, res.run_data.stats.num_conflicts),
                                                                 "resolution",
                                                                 compared_programs,
                                                                 lambda instance: True,
                                                                 lambda res: res.run_data.stats.num_conflicts > 0)

            return instance_improvements

    def find_output_name(mzn_path: Path, dzn_path: Optional[Path], extension: str, divider="."):
        if dzn_path is None:
            output_name = f"{mzn_path.stem}{divider}{extension}"
        else:
            dzn_name = dzn_path.stem

            dzn_parent_path = dzn_path.parent
            while mzn_path.parent != dzn_parent_path:
                dzn_name = f"{dzn_parent_path.stem}{divider}{dzn_name}"
                dzn_parent_path = dzn_parent_path.parent

            output_name = f"{mzn_path.stem}{divider}{dzn_name}{divider}{extension}"

        return output_name

    def compute_instance_models():
        instance_models = {}
        with VARIANTS_INPUT_PKL.open('rb') as f:
            results = all_results(f)
            for (instance, programs) in results.items():
                resolution_data = programs.get((Program.RESOLUTION, "resolution"))
                if resolution_data is None:
                    continue

                paper_set_i = resolution_data.fzn_file_path.parts.index("paper-set")
                remaining_parts = Path(*resolution_data.fzn_file_path.parts[paper_set_i + 1:-1])
                base_path = DATASETS_DIR / "paper-set-inputs"
                folder_path = base_path.joinpath(remaining_parts)
                mzns = list(folder_path.glob("*.mzn"))
                dzns = list(folder_path.rglob("*.dzn"))
                if len(mzns) > 0 and len(dzns) == 0:
                    dzns = [None]

                for (mzn, dzn) in product(mzns, dzns):
                    instance_name = find_output_name(mzn, dzn, "fzn")
                    instance_models[instance_name.removesuffix(".fzn")] = mzn.name

        return instance_models

    instance_models = compute_instance_models()
    print("Instance to model:", instance_models)

    all_instance_improvements = compute_instance_improvements()
    print("Instance improvements:", all_instance_improvements)

    with open("paper-set-counts.pkl", "rb") as var_f:
        problem_counts = pickle.load(var_f)
    problem_counts = {p.stem: {c: v for (c, v) in counts.items() if c != ''} for (p, counts) in problem_counts.items()}
    constraint_types = list(list(problem_counts.values())[0].keys())

    differences_per_constraint = {}

    total_model_count = defaultdict(lambda: 0)
    for model in instance_models.values():
        total_model_count[model] += 1

    def instance_stats(filter_improvement, name):
        if len(differences_per_constraint) == 0 and name != "baseline":
            print("First call must be baseline")
            exit(1)

        instance_improvements = {instance: v
                                 for (instance, v) in all_instance_improvements.items()
                                 if filter_improvement(v["llg"])
                                 }

        curr_instances = list(instance_improvements.keys())
        curr_models = set(map(lambda k: instance_models[k], filter(lambda k: k in instance_models, curr_instances)))

        curr_instances_count = {instance: problem_counts[instance] for instance in curr_instances}
        curr_instance_constraint_count = {instance: sum(counts.values()) for (instance, counts) in curr_instances_count.items()}
        median_constraints = np.median(list(curr_instance_constraint_count.values()))
        std_constraints = np.std(list(curr_instance_constraint_count.values()))

        curr_constraint_count = {constr: sum(c[constr] for c in curr_instances_count.values()) for constr in constraint_types}
        curr_constraint_total = sum(curr_constraint_count.values())
        curr_constraint_percs = {constr: 100 * count / curr_constraint_total for (constr, count) in curr_constraint_count.items()}

        print(f"==> Found {len(curr_instances)} instances from {len(curr_models)} models, constraints {median_constraints} (std {std_constraints})")

        model_count = defaultdict(lambda: 0)
        for instance in curr_instances:
            model = instance_models[instance]
            model_count[model] += 1

        for (model, count) in model_count.items():
            print(f"==> {model}: {count} / {total_model_count[model]}")

        print("==> Distribution:", curr_constraint_percs)
        print("==> Instances:", curr_instances)
        print("")

        for c in constraint_types:
            perc_curr = curr_constraint_percs[c]
            if name == "baseline":
                differences_per_constraint[c] = [(perc_curr, "baseline")]
            else:
                perc_baseline = differences_per_constraint[c][0][0]
                diff = perc_curr - perc_baseline
                differences_per_constraint[c].append((diff, name))

    print("All instances")
    instance_stats(lambda v: True, "baseline")

    print(">110% instances")
    instance_stats(lambda v: v > 1.1, ">1.1")

    print(">100% instances")
    instance_stats(lambda v: v > 1, ">1")

    print(">90% instances")
    instance_stats(lambda v: v > 0.9, ">0.9")

    print("<50% instances")
    instance_stats(lambda v: v < 0.5, "<0.5")

    print("<30% instances")
    instance_stats(lambda v: v < 0.3, "<0.3")

    print("<25% instances")
    instance_stats(lambda v: v < 0.25, "<0.25")

    print("<20% instances")
    instance_stats(lambda v: v < 0.2, "<0.2")

    print("<10% instances")
    instance_stats(lambda v: v < 0.1, "<0.1")

    for (c, values) in differences_per_constraint.items():
        print(c)

        for (perc, name) in values:
            if name == "baseline":
                print(f"baseline: {perc:.3f}%")
                continue

            if perc > 0:
                print(f"{name}: +{perc:.3f}%")
            else:
                print(f"{name}: {perc:.3f}%")

        print("")


def experiment_1_reduction_of_conflicts():
    parse_variants_if_not_exists()

    with VARIANTS_INPUT_PKL.open('rb') as f:
        program_titles = {
            "llg": "LLG",
            "resolution": "Resolution",
        }

        _, conflict_improvements = get_improvement_in_metric(f,
                                                             lambda res: max(1, res.run_data.stats.num_conflicts),
                                                             "resolution",
                                                             set(program_titles.keys()),
                                                             lambda instance: True,
                                                             lambda res: (res.run_data.stats.num_conflicts > 0) and
                                                                         (
                                                                                 res.program == Program.RESOLUTION or res.run_data.stats.llg_learned_constraints > 0)
                                                             )

        improvements_boxplots([
            BoxplotConfig(conflict_improvements,
                          r"\textbf{Ratio of \# conflicts}",
                          show_outliers=False,
                          quartile_bbox=dict(facecolor="#4987b3"),
                          args=dict(whis=(10, 90))
                          ),
        ],
            skip_programs=['resolution'],
            titles=program_titles,
            show_title=False,
        )

        plt.tight_layout()

        plt.savefig("experiment_1_overall_conflict_improvements_only_learned.pgf")
        plt.show()


def experiment_2_strength_inequalities():
    if not EXPERIMENT_2_AGG_PKL.exists():
        parse_analysis_log()

        with ANALYSIS_LOG_NO_LEARNED.open('rb') as analysis_log_pkl, EXPERIMENT_2_INPUT_PKL.open('rb') as exp_2_pkl:
            analysis_log_parsed = all_results(analysis_log_pkl)

            instance_percs_weights_ineq, instance_percs_weights_nogood = {}, {}

            failed_instances = 0
            for i, (instance, constraint_type, propagation_matches) in enumerate(generate_results(exp_2_pkl)):
                print(f"Checking {i} (failed {failed_instances}): {instance}")

                if analysis_log_parsed[instance][(Program.LLG, "llg-analysis-log")].failed_print_reason():
                    failed_instances += 1
                    continue

                if constraint_type == ConstraintType.INEQUALITY:
                    instance_percs_weights_ineq[instance] = (
                        np.array(list(map(lambda t: t[1] / t[0], propagation_matches))),
                        np.array(list(map(lambda t: 1 / len(propagation_matches), propagation_matches)))
                    )
                else:
                    instance_percs_weights_nogood[instance] = (
                        np.array(list(map(lambda t: t[1] / t[0], propagation_matches))),
                        np.array(list(map(lambda t: 1 / len(propagation_matches), propagation_matches)))
                    )

        with EXPERIMENT_2_AGG_PKL.open("wb") as agg_pkl:
            pickle.dump((instance_percs_weights_ineq, instance_percs_weights_nogood), agg_pkl)
    else:
        with EXPERIMENT_2_AGG_PKL.open("rb") as agg_pkl:
            (instance_percs_weights_ineq, instance_percs_weights_nogood) = pickle.load(agg_pkl)

    ineqs_matched = np.average(np.concatenate(list(map(lambda v: v[0], instance_percs_weights_ineq.values()))),
                               weights=np.concatenate(list(map(lambda v: v[1], instance_percs_weights_ineq.values()))))

    nogoods_matched = np.average(np.concatenate(list(map(lambda v: v[0], instance_percs_weights_nogood.values()))),
                                 weights=np.concatenate(list(map(lambda v: v[1], instance_percs_weights_nogood.values()))))

    print(f"Ineqs matched: {100 * ineqs_matched}%")
    print(f"Nogoods matched: {100 * nogoods_matched}%")

    values = [nogoods_matched, ineqs_matched]

    # Plotting
    plt.figure(figsize=(10, 6))

    plt.bar([r"\textbf{Nogood matched}", r"\textbf{Inequality matched}"],
            values,
            color=plt.cm.tab10(np.linspace(0, 1, len(values))))

    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"${100 * v:.3f}\%$", ha='center', fontsize=14, fontweight='bold')

    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1))

    plt.ylabel(r"\textbf{\% of matched propagations by alternative}", labelpad=15)

    plt.tight_layout()

    plt.savefig("experiment_2_matching_propagations.pgf")
    plt.show()


def compute_instance_improvements():
    parse_variants_if_not_exists()
    with VARIANTS_INPUT_PKL.open('rb') as f:
        program_titles = {
            "llg": "LLG",
            "resolution": "Resolution",
        }

        instance_improvements, _ = get_improvement_in_metric(f,
                                                             lambda res: max(1, res.run_data.stats.num_conflicts),
                                                             "resolution",
                                                             set(program_titles.keys()),
                                                             lambda instance: True,
                                                             lambda res: (res.run_data.stats.num_conflicts > 0))

    return instance_improvements


def experiment_3a_fallback_reasons_over_time():
    bins = np.linspace(0, 100, 51)

    if not EXPERIMENT_3A_AGG_PKL.exists():
        parse_analysis_log()

        with ANALYSIS_LOG_NO_LEARNED.open('rb') as analysis_log_pkl, EXPERIMENT_3_INPUT_PKL.open('rb') as exp_3_pkl:
            analysis_log_parsed = all_results(analysis_log_pkl)

            print("Aggregating all instances....")

            instance_fallbacks, all_reasons = {}, set()

            total_instances, failed_instances = 0, 0
            for (instance, program_name, total_conflicts, success_at_conflict, fallback_at_conflict, _) in generate_results(exp_3_pkl):
                print(f"==> {total_instances} (failed {failed_instances}): {instance} ({total_conflicts} conflicts)")

                if analysis_log_parsed[instance][(Program.LLG, program_name)].failed_print_reason():
                    failed_instances += 1
                    continue

                if total_conflicts <= 50:
                    # Does not fit properly in the bins
                    continue

                total_instances += 1

                fallbacks = defaultdict(list)
                for (fallback_conflict, fallback_reason) in fallback_at_conflict:
                    all_reasons.add(fallback_reason)
                    fallbacks[fallback_reason].append(100 * (fallback_conflict - 1) / (total_conflicts - 2))

                successes = []
                for learned_at_conflict in success_at_conflict:
                    successes.append(100 * (learned_at_conflict - 1) / (total_conflicts - 2))

                instance_fallbacks[instance] = (dict(fallbacks), successes)

        with EXPERIMENT_3A_AGG_PKL.open("wb") as agg_pkl:
            pickle.dump((instance_fallbacks, all_reasons), agg_pkl)
    else:
        with EXPERIMENT_3A_AGG_PKL.open("rb") as agg_pkl:
            (instance_fallbacks, all_reasons) = pickle.load(agg_pkl)

    def plot_and_output_for_instances(name, instance_filter):
        curr_instance_fallbacks = {k: v for (k, v) in instance_fallbacks.items() if instance_filter(k)}

        # Plotting
        plt.figure(figsize=(8, 5))

        hists = {
            instance: {
                          reason.name: np.histogram(occurences, bins)[0]
                          for (reason, occurences) in fallbacks[0].items()
                      } | {"success": np.histogram(fallbacks[1], bins)[0]}
            for (instance, fallbacks) in curr_instance_fallbacks.items()
        }

        totals = {
            instance: np.sum(list(hists.values()), axis=0)
            for (instance, hists) in hists.items()
        }

        hists_percentages = {
            instance: {
                reason: occurences / totals[instance] * 100
                for (reason, occurences) in curr_hists.items()
            }
            for (instance, curr_hists) in hists.items()
        }

        percs = {
            reason: np.sum(np.array(list(map(lambda p: p.get(reason, np.zeros(len(bins) - 1)), hists_percentages.values()))), axis=0) / len(
                hists_percentages)
            for reason in [*map(lambda r: r.name, all_reasons), "success"]
        }

        # Plotting
        reason_labels = {
            "success": "Successful linear analysis",
            "NOGOOD_EXPLANATION": "Nogood explanation encountered",
            "NOGOOD_CONFLICT": "Nogood conflict encountered",
            "NOT_CONFLICTING": "Not conflicting",
            # "OVERFLOW": "Overflow",
            # "DECISION_REACHED": "Decision reached",
            # "NOTHING_LEARNED": "Nothing learned",
        }

        percs_labeled = {
            reason_label: percs[reason]
            for (reason, reason_label) in reason_labels.items()
        }

        line_styles = ["-", ":", "--", "-."]
        for i, (label, values) in enumerate(percs_labeled.items()):
            plt.plot(bins[:-1], values, label=label, linestyle=line_styles[i % len(line_styles)])

        plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(100))
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(100))
        plt.gca().tick_params(axis='both', which='major', labelsize=19)

        plt.xlabel(r"\textbf{\% of full search}", labelpad=10, fontsize=19)
        plt.ylabel(r"\textbf{\% of conflict analyses}", labelpad=10, fontsize=19)
        plt.legend(fontsize=19)

        plt.grid(color='#d6d6d6', linewidth=0.5)

        plt.tight_layout()

        plt.savefig(f"experiment_3a_fallback_reasons_over_time_{name}.pgf", bbox_inches='tight', pad_inches=0.0)
        plt.show()

    plot_and_output_for_instances("all", lambda instance: True)

    instance_improvements = compute_instance_improvements()
    top_50 = {i for (i, improvements) in instance_improvements.items() if improvements['llg'] < 0.5}
    top_25 = {i for (i, improvements) in instance_improvements.items() if improvements['llg'] < 0.25}

    plot_and_output_for_instances("top50", lambda instance: instance in top_50)
    plot_and_output_for_instances("top25", lambda instance: instance in top_25)


def experiment_3b_propagation_over_time():
    bins = np.linspace(0, 100, 51)

    if not EXPERIMENT_3B_AGG_PKL.exists():
        parse_analysis_log()

        with ANALYSIS_LOG_NO_LEARNED.open('rb') as analysis_log_pkl, EXPERIMENT_3_INPUT_PKL.open('rb') as exp_3_pkl:
            analysis_log_parsed = all_results(analysis_log_pkl)

            print("Aggregating all instances....")

            instance_count = {}

            total_instances, failed_instances = 0, 0
            for (instance, program_name, total_conflicts, _, _, propagated_at_conflict) in generate_results(exp_3_pkl):
                print(f"==> {total_instances} (failed {failed_instances}): {instance} ({total_conflicts} conflicts)")

                if analysis_log_parsed[instance][(Program.LLG, program_name)].failed_print_reason():
                    failed_instances += 1
                    continue

                if total_conflicts <= 50:
                    continue

                if len(propagated_at_conflict) == 0:
                    continue

                total_instances += 1

                bin_counts = np.zeros(len(bins) - 1)
                for constr in propagated_at_conflict:
                    if len(constr) == 0:
                        continue

                    group_density, _ = np.histogram(100 * (np.array(list(constr.keys())) - 1) / (total_conflicts - 1),
                                                    bins=bins,
                                                    weights=list(constr.values()))
                    bin_counts += group_density

                instance_count[instance] = bin_counts

        with EXPERIMENT_3B_AGG_PKL.open("wb") as agg_pkl:
            pickle.dump(instance_count, agg_pkl)
    else:
        with EXPERIMENT_3B_AGG_PKL.open("rb") as agg_pkl:
            instance_count = pickle.load(agg_pkl)

    # Plotting
    plt.figure(figsize=(8, 5))

    instance_count = list(instance_count.items())
    instance_count = sorted(instance_count, key=lambda v: v[1][0])

    # Option 0: no groups
    # groups = [instance_count]
    # group_labels = ["all"]

    # Option 1: group by #conflicts
    # def chunk_into_n(lst, n):
    #     size = ceil(len(lst) / n)
    #     return list(
    #         map(lambda x: lst[x * size:x * size + size],
    #             list(range(n)))
    #     )
    # groups = chunk_into_n(instance_count, 6)
    # group_labels = [
    #     str(max([conflicts for (_instance, (conflicts, _density)) in group]))
    #     for group in groups
    # ]

    # Option 2: group all, 50%, 25%
    instance_improvements = compute_instance_improvements()
    top_50 = {i for (i, improvements) in instance_improvements.items() if improvements['llg'] < 0.5}
    top_25 = {i for (i, improvements) in instance_improvements.items() if improvements['llg'] < 0.25}

    all_instances = list(filter(lambda ic: True, instance_count))
    top_50_instances = list(filter(lambda ic: ic[0] in top_50, instance_count))
    top_25_instances = list(filter(lambda ic: ic[0] in top_25, instance_count))

    groups = [all_instances, top_50_instances, top_25_instances]
    group_labels = ["All", "$\ge 50\%$ conflict reduction", "$\ge 75\%$ conflict reduction"]

    density_per_group = []
    for group in groups:
        group_count, group_instances = np.zeros(len(bins) - 1), 0

        for instance, instance_count in group:
            group_count += instance_count
            group_instances += 1

        group_density = group_count / group_count.sum()
        group_density /= 2  # Half because of density and bins of 2%
        density_per_group.append(group_density)

    line_styles = ["-", ":", "--", "-."]
    for i, (density, label) in enumerate(zip(density_per_group, group_labels)):
        label = label if len(group_labels) > 1 else None
        plt.plot(bins[:-1], density, label=label, linestyle=line_styles[i % len(line_styles)])

    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1))
    plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(100))
    plt.gca().tick_params(axis='both', which='major', labelsize=19)

    plt.xlabel(r"\textbf{\% of full search}", labelpad=10, fontsize=19)
    plt.ylabel(r"\textbf{\% of learned linear propagations}", labelpad=10, fontsize=19)
    plt.legend(fontsize=19)

    plt.grid(color='#d6d6d6', linewidth=0.5)

    plt.tight_layout()

    plt.savefig("experiment_3b_propagation_over_time.pgf", bbox_inches='tight', pad_inches=0.0)
    plt.show()


def experiment_3c_slack_in_explanations():
    if not EXPERIMENT_3C_AGG_PKL.exists():
        if not EXPLANATION_STATS_PKL.exists():
            parse_analysis_log()

        with ANALYSIS_LOG_NO_LEARNED.open('rb') as analysis_log_pkl, EXPLANATION_STATS_PKL.open('rb') as es_f:
            analysis_log_parsed = all_results(analysis_log_pkl)

            instance_used_conflict_slack_frequency, instance_used_explanation_slack_frequency = {}, {}
            instance_skip_conflict_slack_frequency, instance_skip_explanation_slack_frequency = {}, {}

            failed_instances = 0
            for i, (instance, success_constraints, fallback_constraints) in enumerate(generate_results(es_f)):
                print(f"Checking {i} (failed {failed_instances}): {instance}")

                if analysis_log_parsed[instance][(Program.LLG, "llg-analysis-log")].failed_print_reason():
                    failed_instances += 1
                    continue

                used_conflict_slack_frequency, used_explanation_slack_frequency = defaultdict(lambda: 0), defaultdict(lambda: 0)
                for explanations in success_constraints:
                    if len(explanations) == 0:
                        continue

                    used_conflict_slack_frequency[explanations[0].slack] += 1
                    for explanation in explanations[1:]:
                        used_explanation_slack_frequency[explanation.slack] += 1
                instance_used_conflict_slack_frequency[instance] = dict(used_conflict_slack_frequency)
                instance_used_explanation_slack_frequency[instance] = dict(used_explanation_slack_frequency)

                skip_conflict_slack_frequency, skip_explanation_slack_frequency = defaultdict(lambda: 0), defaultdict(lambda: 0)
                for explanations in fallback_constraints:
                    if len(explanations) == 0:
                        continue

                    skip_conflict_slack_frequency[explanations[0].slack] += 1
                    for explanation in explanations:
                        skip_explanation_slack_frequency[explanation.slack] += 1
                instance_skip_conflict_slack_frequency[instance] = dict(skip_conflict_slack_frequency)
                instance_skip_explanation_slack_frequency[instance] = dict(skip_explanation_slack_frequency)

        with open(EXPERIMENT_3C_AGG_PKL, "wb") as f:
            pickle.dump((instance_used_conflict_slack_frequency, instance_used_explanation_slack_frequency,
                         instance_skip_conflict_slack_frequency, instance_skip_explanation_slack_frequency), f)
    else:
        with open(EXPERIMENT_3C_AGG_PKL, "rb") as f:
            (instance_used_conflict_slack_frequency, instance_used_explanation_slack_frequency,
             instance_skip_conflict_slack_frequency, instance_skip_explanation_slack_frequency) = pickle.load(f)

    def plot_for(instance_used_slack_frequency, instance_skip_slack_frequency, name):
        used_slack_frequency = defaultdict(lambda: 0)
        for slacks in instance_used_slack_frequency.values():
            for (slack, count) in slacks.items():
                used_slack_frequency[slack] += count

        skip_slack_frequency = defaultdict(lambda: 0)
        for slacks in instance_skip_slack_frequency.values():
            for (slack, count) in slacks.items():
                skip_slack_frequency[slack] += count

        def compute_slack_percentiles(frequency_dict):
            values = np.array(sorted(frequency_dict.keys()))
            counts = np.array([frequency_dict[v] for v in values])

            cum_counts = np.cumsum(counts)
            total = cum_counts[-1]

            def percentile(p):
                threshold = p / 100 * total
                return values[np.searchsorted(cum_counts, threshold)]

            return [percentile(p) for p in [10, 90]]

        [perc_used_low, perc_used_high] = compute_slack_percentiles(used_slack_frequency)
        [perc_skip_low, perc_skip_high] = compute_slack_percentiles(skip_slack_frequency)

        perc_low = min(perc_used_low, perc_skip_low)
        perc_high = max(perc_used_high, perc_skip_high)

        bins = np.linspace(start=perc_low, stop=perc_high, num=100)

        used_slack_frequency_filt = {s: c for (s, c) in used_slack_frequency.items() if perc_low <= s <= perc_high}
        skip_slack_frequency_filt = {s: c for (s, c) in skip_slack_frequency.items() if perc_low <= s <= perc_high}

        all_slacks = set(used_slack_frequency_filt.keys()) | set(skip_slack_frequency_filt.keys())

        slack_success_perc = {s: used_slack_frequency_filt.get(s, 0) / (used_slack_frequency_filt.get(s, 0) + skip_slack_frequency_filt.get(s, 0))
                              for s in all_slacks}

        # Plotting
        plt.figure(figsize=(10, 6))

        plt.hist(list(slack_success_perc.keys()),
                 weights=list(slack_success_perc.values()),
                 bins=bins,
                 histtype='bar',
                 label=[r"\textbf{Used explanations}", r"\textbf{Skipped explanations}"])

        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1))

        plt.xlabel(r"\textbf{Slack}", labelpad=10)
        plt.ylabel(r"\textbf{\% of explanations}", labelpad=15)

        plt.legend()

        plt.tight_layout()

        plt.savefig(f"experiment_3c_slack_in_explanations_{name}.pgf")
        plt.show()

    plot_for(instance_used_conflict_slack_frequency, instance_skip_conflict_slack_frequency, "conflicts")
    plot_for(instance_used_explanation_slack_frequency, instance_skip_explanation_slack_frequency, "propagations")


def experiment_4_llg_vs_decomposition():
    # Merge linear and normal datasets
    if not EXPERIMENT_4_PKL.exists():
        instance_results = defaultdict(dict)
        for instance_data in parse_instance_run_data_generator(LINEAR_RESULTS_DIR):
            (input_name, instance, program_name, run_result, run_data) = instance_data
            program_name = "llg" if input_name == "paper-set" else "llg-linear"
            instance_results[instance][(Program.LLG, program_name)] = run_result

        with EXPERIMENT_4_PKL.open('wb') as f:
            for (instance, programs) in instance_results.items():
                pickle.dump((instance, programs), f, pickle.HIGHEST_PROTOCOL)

    with EXPERIMENT_4_PKL.open('rb') as f:
        program_titles = {
            "llg-linear": "LLG linear",
            "llg": "LLG",
        }

        instance_improvements, conflict_improvements = get_improvement_in_metric(f,
                                                                                 lambda res: max(1, res.run_data.stats.num_conflicts),
                                                                                 "llg-linear",
                                                                                 set(program_titles.keys()),
                                                                                 lambda instance: True,
                                                                                 lambda res: res.run_data.stats.num_conflicts > 0)

        improvements_boxplots([
            BoxplotConfig(conflict_improvements,
                          r"\textbf{Ratio of \# conflicts}",
                          show_outliers=False,
                          quartile_bbox=dict(facecolor="#4987b3"),
                          args=dict(whis=(10, 90)),
                          ),
        ],
            titles=program_titles,
            skip_programs=["llg-linear"],
            show_title=False,
        )

        plt.tight_layout()

        plt.savefig("experiment_4_llg_vs_decomposition.pgf")
        plt.show()


def experiment_5_learning_nogoods():
    parse_variants_if_not_exists()

    with VARIANTS_INPUT_PKL.open('rb') as f:
        program_titles = {
            "llg": "Invariant: before return",
            "llg-skip-nogood-learning": "Skip nogood learning",
        }

        _, conflict_improvements = get_improvement_in_metric(f,
                                                             lambda res: max(1, res.run_data.stats.num_conflicts),
                                                             "llg",
                                                             set(program_titles.keys()),
                                                             lambda instance: True,
                                                             lambda res: res.run_data.stats.num_conflicts > 0)

        # Plotting
        improvements_boxplots([
            BoxplotConfig(conflict_improvements, r"\textbf{Ratio \# conflicts}",
                          show_outliers=False,
                          quartile_bbox=dict(facecolor="#4987b3"),
                          args=dict(whis=(10, 90)),
                          ),
        ],
            skip_programs=['llg'],
            titles=program_titles,
            show_title=False
        )

        plt.tight_layout()

        plt.savefig("experiment_5_learning_nogoods.pgf")
        plt.show()


if __name__ == "__main__":
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,

        "font.size": "14",
    })

    experiment_0_overall_stats()
    experiment_0_analyze_top_instances()
    experiment_1_reduction_of_conflicts()
    experiment_2_strength_inequalities()
    experiment_3a_fallback_reasons_over_time()
    experiment_3b_propagation_over_time()
    experiment_3c_slack_in_explanations()
    experiment_4_llg_vs_decomposition()
    experiment_5_learning_nogoods()
