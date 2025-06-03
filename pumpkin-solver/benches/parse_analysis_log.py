import pickle
from contextlib import ExitStack
from typing import List, Tuple, BinaryIO

from parse_output_files import InstanceRunData, LearnedConstraint, \
    FallbackConstraint, parse_instance_run_data_generator, LearnedConstraints, parse_experiment_to_file


def parse_analysis_log_experiments_if_not_exists(analysis_results_dir, analysis_results, exp_2_pkl, exp_3_pkl, expl_stats_pkl):
    # Parse w/o learned constraints
    if not analysis_results.exists():
        with analysis_results.open('wb') as f:
            parse_experiment_to_file(analysis_results_dir, f, skip_learned_constraints=True)

    experiment_files = {
        "2": exp_2_pkl,
        "3": exp_3_pkl,
        "stats": expl_stats_pkl
    }

    # Parse w/ learned constraints
    requires_parsing = dict(filter(lambda f: not f[1].exists(), experiment_files.items()))
    if len(requires_parsing) == 0:
        return

    with ExitStack() as stack:
        files = {f[0]: stack.enter_context(f[1].open('wb')) for f in requires_parsing.items()}

        for instance_data in parse_instance_run_data_generator(analysis_results_dir):
            if "2" in files:
                transform_experiment_2_matching_propagations(instance_data, files["2"])

            if "3" in files:
                transform_experiment_3_learning_propagation_over_time(instance_data, files["3"])

            if "stats" in files:
                transform_explanation_stats(instance_data, files["stats"])


def transform_experiment_2_matching_propagations(instance_data: InstanceRunData, f: BinaryIO):
    (input_name, instance, program_name, run_result, run_data) = instance_data

    learned_constraints: LearnedConstraints = run_data.learned_constraints
    if (learned_constraints is None) or (len(learned_constraints) == 0):
        return

    # { [(total_props_errs, props_errs_matched)] }
    propagation_matches: List[Tuple[int, int]] = []
    constraint_type = None

    for constr in learned_constraints.values():
        if isinstance(constr, FallbackConstraint):
            continue

        # Constant within an instance
        constraint_type = constr.constraint_type

        prop_stats = constr.propagation_error_stats

        total_props_errs, props_errs_matched = (prop_stats.total_propagations + prop_stats.total_errors,
                                                prop_stats.matched_propagations + prop_stats.matched_errors)

        if total_props_errs > 0:
            propagation_matches.append((total_props_errs, props_errs_matched))

    pickle.dump((instance, constraint_type, propagation_matches), f, pickle.HIGHEST_PROTOCOL)


def transform_experiment_3_learning_propagation_over_time(instance_data: InstanceRunData, f: BinaryIO):
    (input_name, instance, program_name, run_result, run_data) = instance_data

    if program_name != "llg-analysis-log":
        return

    learned_constraints: LearnedConstraints = run_data.learned_constraints
    if learned_constraints is None:
        return

    total_conflicts = run_data.stats.num_conflicts

    success_at_conflict = [c.learned_at_conflict
                           for c in learned_constraints.values()
                           if isinstance(c, LearnedConstraint)]

    fallback_at_conflict = [(c.fallback_at_conflict, c.fallback_reason)
                            for c in learned_constraints.values()
                            if isinstance(c, FallbackConstraint)]

    propagated_at_conflict = [c.propagation_error_stats.propagations_at_conflicts
                              for c in learned_constraints.values()
                              if isinstance(c, LearnedConstraint)]

    pickle.dump((instance, program_name, total_conflicts, success_at_conflict, fallback_at_conflict, propagated_at_conflict), f,
                pickle.HIGHEST_PROTOCOL)


def transform_explanation_stats(instance_data: InstanceRunData, f: BinaryIO):
    (input_name, instance, program_name, run_result, run_data) = instance_data

    if program_name != "llg-analysis-log":
        return

    learned_constraints: LearnedConstraints = run_data.learned_constraints
    if learned_constraints is None:
        return

    success_constraints, fallback_constraints = [], []
    for constr in learned_constraints.values():
        if isinstance(constr, FallbackConstraint):
            fallback_constraints.append(constr.used_explanations)
        else:
            success_constraints.append(constr.inequality_stats.used_explanations)

    pickle.dump((instance, success_constraints, fallback_constraints), f, pickle.HIGHEST_PROTOCOL)
