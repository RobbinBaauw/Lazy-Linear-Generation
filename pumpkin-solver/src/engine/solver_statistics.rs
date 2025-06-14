use crate::basic_types::moving_averages::CumulativeMovingAverage;
use crate::create_statistics_struct;

create_statistics_struct!(
    /// Structure responsible for storing several statistics of the solving process of the
    /// [`ConstrallgisfactionSolver`].
    SolverStatistics {
        /// Core statistics of the solver engine (e.g. the number of decisions)
        engine_statistics: EngineStatistics,
        /// The statistics related to clause learning
        learned_clause_statistics: LearnedClauseStatistics,
        llg_statistics: LLGStatistics,
    }
);

create_statistics_struct!(
    /// Core statistics of the solver engine (e.g. the number of decisions)
    EngineStatistics {
        /// The number of decisions taken by the solver
        num_decisions: u64,
        /// The number of conflicts encountered by the solver
        num_conflicts: u64,
        /// The number of times the solver has restarted
        num_restarts: u64,
        /// The average number of (integer) propagations made by the solver
        num_propagations: u64,
        /// The amount of time which is spent in the solver
        time_spent_in_solver: u64,
});

create_statistics_struct!(
    /// The statistics related to clause learning
    LearnedClauseStatistics {
        /// The average number of elements in the conflict explanation
        average_conflict_size: CumulativeMovingAverage<u64>,
        /// The average number of literals removed by recursive minimisation during conflict analysis
        average_number_of_removed_literals_recursive: CumulativeMovingAverage<u64>,
        /// The average number of literals removed by semantic minimisation during conflict analysis
        average_number_of_removed_literals_semantic: CumulativeMovingAverage<u64>,
        /// The number of learned clauses which have a size of 1
        num_unit_clauses_learned: u64,
        /// The average length of the learned clauses
        average_learned_clause_length: CumulativeMovingAverage<u64>,
        /// The average number of levels which have been backtracked by the solver (e.g. when a learned clause is created)
        average_backtrack_amount: CumulativeMovingAverage<u64>,
        /// The average literal-block distance (LBD) metric for newly added learned nogoods
        average_lbd: CumulativeMovingAverage<u64>,
});

create_statistics_struct!(LLGStatistics {
    llg_learned_constraints: u64,
    llg_learned_constraints_avg_length: CumulativeMovingAverage<u64>,
    llg_constraint_avg_lhs_coeff: CumulativeMovingAverage<u64>,
    llg_fallback_used: u64,

    llg_linked_auxiliaries: u64,
    llg_unlinked_auxilaries: u64,
    llg_unlinked_auxiliaries_constraints: u64,

    llg_skipped_high_slack: u64,
    llg_clauses_to_inequalities: u64,

    llg_time_spent_analysis_ns: u64,
    llg_time_spent_cutting_ns: u64,
    llg_time_spent_checking_backjump_ns: u64,
    llg_time_spent_checking_overflow_ns: u64,
    llg_time_spent_checking_conflicting_ns: u64,
    llg_time_spent_creating_explanations_ns: u64,
});
