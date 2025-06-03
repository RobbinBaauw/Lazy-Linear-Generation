# Lazy Linear Generation
## Introduction
This codebase is supplementary material for the paper "Conflict Analysis Based on Cutting Planes for Constraint Programming" to be presented at CP2025.
This implementation allows for reproduction of the experimental results. The datasets can be downloaded separately [here](TODO) (TODO).

The main parts of LLG can be found in the following files:

* [`llg_benchmarks.rs`](../src/bin/pumpkin-solver/llg_benchmarks.rs): the entry point of the binary used for experiments, taking the arguments as used by the experiments
* [`llg_conflict_resolver.rs`](../src/engine/conflict_analysis/resolvers/llg_conflict_resolver.rs): contains the conflict analysis algorithm, with a fallback to LCG
* As part of the propagators [`absolute_value.rs`](../src/propagators/arithmetic/absolute_value.rs), [`division.rs`](../src/propagators/arithmetic/division.rs), [`integer_multiplication.rs`](../src/propagators/arithmetic/integer_multiplication.rs), [`linear_less_or_equal.rs`](../src/propagators/arithmetic/linear_less_or_equal.rs), [`linear_not_equal.rs`](../src/propagators/arithmetic/linear_not_equal.rs), [`maximum.rs`](../src/propagators/arithmetic/maximum.rs), [`element.rs`](../src/propagators/element.rs) and [`reified_propagator.rs`](../src/propagators/reified_propagator.rs)
* [`linear_inequality_literal_propagator.rs`](../src/propagators/linear_inequality_literal_propagator.rs): the auxiliary variable propagator

## Running experiments
After building the benchmark binary using `cargo build --bin llg_benchmarks --release`, we can run the following experiment files:

* [`experiments_analysis_log.toml`](experiments_analysis_log.toml): performs experiments that keep track of extensive analysis information such as failure reasons
* [`experiments_variations.toml`](experiments_variations.toml): performs experiments on different variations of LLG
* [`experiments_linear.toml`](experiments_linear.toml): compares the linear decomposition versus the standard Pumpkin decomposition

These experiment files can either be run locally, or on a slurm cluster. The configuration of slurm and output directories for experimetns can also be found in the experiment files.
We can execute the experiments using [`slurm.py`](slurm/slurm.py) by executing the following command:

```bash
slurm.py --config ./experiments_analysis_log.toml # For slurm execution
slurm.py --local --config ./experiments_analysis_log.toml # For local execution
```

When running on slurm, the status can be checked using `slurm.py status`.

## Analyzing the dataset
A prerequisite for some experiments is an analysis of the dataset. This is performed by [`problem-set-analysis.py`](problem-set-analysis.py).
This tool analyzes the dataset in `./datasets/paper-set` (can be changed), and outputs `paper-set-counts.pkl` to be used by other experiments.

## Analyzing experiments
[`experiments.py`](experiments.py) is tasked with analyzing the experiments and outputting plots. 
It contains the following functions:

* `experiment_0_overall_stats`: prints overall statistics about the success rates of LLG vs LCG
* `experiment_0_analyze_top_instances`: analyzes the properties of the best-performing instances
* `experiment_1_reduction_of_conflicts`: plots the reduction of conflicts between LLG and LCG
* `experiment_2_strength_inequalities`: plots the propagation match percentage between LLG and LCG
* `experiment_3a_fallback_reasons_over_time`: plots fallback reasons over time
* `experiment_3b_propagation_over_time`: plots density of propagations from learned constraints
* `experiment_3c_slack_in_explanations`: plots the slack of explanations used and skipped for constructing learned constraints
* `experiment_4_llg_vs_decomposition`: plots the reduction of conflicts between LLG and the linear decomposition
* `experiment_5_learning_nogoods`: plots the increase of conflicts when omitting clause storage 

Note that the default settings assume the results of the experiments can be found in the `results/variants`, `results/analysis_log` and `results/linear` directories respectively.