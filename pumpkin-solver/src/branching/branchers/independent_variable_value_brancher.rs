//! A [`Brancher`] which simply switches uses a single [`VariableSelector`] and a single
//! [`ValueSelector`].

use std::marker::PhantomData;

use crate::basic_types::SolutionReference;
use crate::branching::value_selection::ValueSelector;
use crate::branching::variable_selection::VariableSelector;
use crate::branching::Brancher;
use crate::branching::SelectionContext;
use crate::engine::predicates::predicate::Predicate;
use crate::engine::variables::DomainId;

/// An implementation of a [`Brancher`] which simply uses a single
/// [`VariableSelector`] and a single [`ValueSelector`] independently of one another.
#[derive(Debug)]
pub struct IndependentVariableValueBrancher<Var, VariableSelect, ValueSelect>
where
    VariableSelect: VariableSelector<Var>,
    ValueSelect: ValueSelector<Var>,
{
    /// The [`VariableSelector`] of the [`Brancher`], determines which (unfixed) variable to branch
    /// next on.
    pub(crate) variable_selector: VariableSelect,
    /// The [`ValueSelector`] of the [`Brancher`] determines which value in the domain to branch
    /// next on given a variable.
    pub(crate) value_selector: ValueSelect,
    /// [`PhantomData`] to ensure that the variable type is bound to the
    /// [`IndependentVariableValueBrancher`]
    pub(crate) variable_type: PhantomData<Var>,
    /// Whether to allow auxiliary variables to be added to the variable selector.
    pub(crate) allow_auxiliary_variables: bool,
}

impl<Var, VariableSelect, ValueSelect>
    IndependentVariableValueBrancher<Var, VariableSelect, ValueSelect>
where
    VariableSelect: VariableSelector<Var>,
    ValueSelect: ValueSelector<Var>,
{
    pub fn new(
        var_selector: VariableSelect,
        val_selector: ValueSelect,
        allow_auxiliary_variables: bool,
    ) -> Self {
        IndependentVariableValueBrancher {
            variable_selector: var_selector,
            value_selector: val_selector,
            variable_type: PhantomData,
            allow_auxiliary_variables,
        }
    }
}

impl<Var, VariableSelect, ValueSelect> Brancher
    for IndependentVariableValueBrancher<Var, VariableSelect, ValueSelect>
where
    VariableSelect: VariableSelector<Var>,
    ValueSelect: ValueSelector<Var>,
{
    /// First we select a variable
    ///  - If all variables under consideration are fixed (i.e. `select_variable` return None) then
    ///    we simply return None
    ///  - Otherwise we select a value and return the corresponding literal
    fn next_decision(&mut self, context: &mut SelectionContext) -> Option<Predicate> {
        self.variable_selector
            .select_variable(context)
            .map(|selected_variable| {
                // We have selected a variable, select a value for the PropositionalVariable
                self.value_selector.select_value(context, selected_variable)
            })
    }

    fn on_backtrack(&mut self) {
        self.variable_selector.on_backtrack()
    }

    fn on_conflict(&mut self) {
        self.variable_selector.on_conflict()
    }

    fn on_unassign_integer(&mut self, variable: DomainId, value: i32) {
        self.variable_selector.on_unassign_integer(variable, value);
        self.value_selector.on_unassign_integer(variable, value)
    }

    fn on_appearance_in_conflict_predicate(&mut self, predicate: Predicate) {
        self.variable_selector
            .on_appearance_in_conflict_predicate(predicate)
    }

    fn on_solution(&mut self, solution: SolutionReference) {
        self.value_selector.on_solution(solution);
    }

    fn is_restart_pointless(&mut self) -> bool {
        self.variable_selector.is_restart_pointless() && self.value_selector.is_restart_pointless()
    }

    fn add_auxiliary_variable(&mut self, variable: DomainId) {
        if self.allow_auxiliary_variables {
            self.variable_selector.add_auxiliary_variable(variable)
        }
    }
}
