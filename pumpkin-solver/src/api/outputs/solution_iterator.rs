//! Contains the structures corresponding to solution iterations.

use std::fmt::Debug;
use std::fmt::Formatter;

use super::SatisfactionResult::Satisfiable;
use super::SatisfactionResult::Unknown;
use super::SatisfactionResult::Unsatisfiable;
use crate::branching::Brancher;
use crate::predicate;
use crate::predicates::Predicate;
use crate::results::ProblemSolution;
use crate::results::Solution;
use crate::termination::TerminationCondition;
use crate::variables::DomainId;
use crate::Solver;

/// A struct which allows the retrieval of multiple solutions to a satisfaction problem.
pub struct SolutionIterator<'solver, 'brancher, 'termination, B: Brancher, T> {
    solver: &'solver mut Solver,
    brancher: &'brancher mut B,
    termination: &'termination mut T,
    is_part_of_output: Option<&'solver dyn Fn(DomainId) -> bool>,
    next_blocking_clause: Option<Vec<Predicate>>,
    has_solution: bool,
}

impl<'solver, 'brancher, 'termination, B: Brancher, T: TerminationCondition>
    SolutionIterator<'solver, 'brancher, 'termination, B, T>
{
    pub(crate) fn new(
        solver: &'solver mut Solver,
        brancher: &'brancher mut B,
        termination: &'termination mut T,
        is_part_of_output: Option<&'solver dyn Fn(DomainId) -> bool>,
    ) -> Self {
        SolutionIterator {
            solver,
            brancher,
            termination,
            is_part_of_output,
            next_blocking_clause: None,
            has_solution: false,
        }
    }

    /// Find a new solution by blocking the previous solution from being found. Also calls the
    /// [`Brancher::on_solution`] method from the [`Brancher`] used to run the initial solve.
    pub fn next_solution(&mut self) -> IteratedSolution {
        if let Some(blocking_clause) = self.next_blocking_clause.take() {
            self.solver
                .get_satisfaction_solver_mut()
                .restore_state_at_root(self.brancher);
            if self.solver.add_clause(blocking_clause).is_err() {
                return IteratedSolution::Finished;
            }
        }
        match self.solver.satisfy(self.brancher, self.termination) {
            Satisfiable(solution) => {
                self.has_solution = true;
                self.next_blocking_clause = Some(self.get_blocking_clause(&solution));
                IteratedSolution::Solution(solution)
            }
            Unsatisfiable => {
                if self.has_solution {
                    IteratedSolution::Finished
                } else {
                    IteratedSolution::Unsatisfiable
                }
            }
            Unknown => IteratedSolution::Unknown,
        }
    }

    /// Creates a clause which prevents the current solution from occurring again by going over the
    /// defined output variables and creating a clause which prevents those values from
    /// being assigned.
    ///
    /// This method is used when attempting to find multiple solutions.
    fn get_blocking_clause(&self, solution: &Solution) -> Vec<Predicate> {
        solution
            .get_domains()
            .filter(|domain| self.is_part_of_output.is_none_or(|check| check(*domain)))
            .map(|variable| predicate!(variable != solution.get_integer_value(variable)))
            .collect::<Vec<_>>()
    }
}

impl<B: Brancher, T> Debug for SolutionIterator<'_, '_, '_, B, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SolutionIterator")
            .field("solver", &self.solver)
            .field("next_blocking_clause", &self.next_blocking_clause)
            .field("has_solution", &self.has_solution)
            .finish_non_exhaustive()
    }
}

/// Enum which specifies the status of the call to [`SolutionIterator::next_solution`].
#[allow(
    clippy::large_enum_variant,
    reason = "these will not be stored in bulk, so this is not an issue"
)]
#[derive(Debug)]
pub enum IteratedSolution {
    /// A new solution was identified.
    Solution(Solution),

    /// No more solutions exist.
    Finished,

    /// The solver was terminated during search.
    Unknown,

    /// There exists no solution
    Unsatisfiable,
}
