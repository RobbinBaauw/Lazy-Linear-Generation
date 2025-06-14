#[cfg(doc)]
use crate::branching::variable_selection::Smallest;
use crate::branching::SelectionContext;
use crate::engine::predicates::predicate::Predicate;
use crate::engine::variables::DomainId;
#[cfg(doc)]
use crate::engine::ConstraintSatisfactionSolver;

/// A trait containing the interface for [`VariableSelector`]s,
/// specifying the appropriate hooks into the solver and the methods required for selecting
/// variables.
pub trait VariableSelector<Var> {
    /// Determines which variable to select next if there are any left to branch on.
    /// Should only return [`None`] when all variables which have been passed to the
    /// [`VariableSelector`] have been assigned. Otherwise it should return the variable to
    /// branch on next.
    fn select_variable(&mut self, context: &mut SelectionContext) -> Option<Var>;

    /// Adds a new auxiliary variable to the variable selector
    fn add_auxiliary_variable(&mut self, _variable: DomainId) {}

    /// A function which is called after a conflict has been found and processed but (currently)
    /// does not provide any additional information.
    fn on_conflict(&mut self) {}

    /// A function which is called whenever a backtrack occurs in the solver.
    fn on_backtrack(&mut self) {}

    /// A function which is called after a [`DomainId`] is unassigned during backtracking (i.e. when
    /// it was fixed but is no longer), specifically, it provides `variable` which is the
    /// [`DomainId`] which has been reset. This method could thus be called multiple times in a
    /// single backtracking operation by the solver.
    fn on_unassign_integer(&mut self, _variable: DomainId, _value: i32) {}

    /// A function which is called when a [`Predicate`] appears in a conflict during conflict
    /// analysis.
    fn on_appearance_in_conflict_predicate(&mut self, _predicate: Predicate) {}

    /// This method returns whether a restart is *currently* pointless for the [`VariableSelector`].
    ///
    /// For example, if a [`VariableSelector`] is using a static strategy (e.g. [`Smallest`]) then a
    /// restart is pointless; however, for a [`VariableSelector`] which
    /// changes throughout the search process restarting is not pointless.
    ///
    /// Note that even if the [`VariableSelector`] has indicated that a restart is pointless, it
    /// could be that the restart is still performed.
    fn is_restart_pointless(&mut self) -> bool {
        true
    }
}
