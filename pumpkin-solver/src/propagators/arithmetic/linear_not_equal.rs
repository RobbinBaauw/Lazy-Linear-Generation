use std::rc::Rc;

use enumset::enum_set;
use itertools::Itertools;

use crate::basic_types::linear_less_or_equal::FilterNonZero;
use crate::basic_types::linear_less_or_equal::LinearLessOrEqual;
use crate::basic_types::PropagationReason;
use crate::basic_types::PropagationStatusCP;
use crate::basic_types::PropositionalConjunction;
use crate::engine::cp::propagation::ReadDomains;
use crate::engine::domain_events::DomainEvents;
use crate::engine::opaque_domain_event::OpaqueDomainEvent;
use crate::engine::propagation::EnqueueDecision;
use crate::engine::propagation::LocalId;
use crate::engine::propagation::PropagationContext;
use crate::engine::propagation::PropagationContextMut;
use crate::engine::propagation::Propagator;
use crate::engine::propagation::PropagatorInitialisationContext;
use crate::engine::variables::IntegerVariable;
use crate::engine::Assignments;
use crate::engine::IntDomainEvent;
use crate::new_explanation;
use crate::predicate;
use crate::pumpkin_assert_extreme;
use crate::pumpkin_assert_moderate;
use crate::pumpkin_assert_simple;
use crate::variables::TransformableVariable;

/// Propagator for the constraint `\sum x_i != rhs`, where `x_i` are
/// integer variables and `rhs` is an integer constant.
#[derive(Clone, Debug)]
pub(crate) struct LinearNotEqualPropagator<Var> {
    /// The terms of the sum
    terms: Rc<[Var]>,
    /// The right-hand side of the sum
    rhs: i32,

    /// The number of fixed terms; note that this constraint can only propagate when there is a
    /// single unfixed variable and can only detect conflicts if all variables are assigned
    number_of_fixed_terms: usize,
    /// The sum of the values of the fixed terms
    fixed_lhs: i32,
    /// Indicates whether the single unfixed variable has been updated; if this is the case then
    /// the propagator is not scheduled again
    unfixed_variable_has_been_updated: bool,
    /// Indicates whether the value of [`LinearNotEqualPropagator::fixed_lhs`] is invalid and
    /// should be recalculated
    should_recalculate_lhs: bool,
}

impl<Var> LinearNotEqualPropagator<Var>
where
    Var: IntegerVariable + 'static,
{
    pub(crate) fn new(terms: Box<[Var]>, rhs: i32) -> Self {
        LinearNotEqualPropagator {
            terms: terms.into(),
            rhs,
            number_of_fixed_terms: 0,
            fixed_lhs: 0,
            unfixed_variable_has_been_updated: false,
            should_recalculate_lhs: false,
        }
    }

    fn create_reason(
        &self,
        assignments: &mut Assignments,
        increase_lower_bound_inequality: bool,
    ) -> LinearLessOrEqual {
        // Transform terms into linleq
        let LinearLessOrEqual { lhs, rhs, .. } = LinearLessOrEqual::new(&self.terms, self.rhs);

        let lb_lhs_init = lhs.lb_initial(assignments) as i32;
        let ub_lhs_init = lhs.ub_initial(assignments) as i32;

        // If lb(Ax) >= b, we know that Ax >= b is always true. Then, we can simply state
        // Ax > b, or -Ax <= -b - 1
        if lb_lhs_init == self.rhs {
            return LinearLessOrEqual::new_expl(
                lhs.iter().map(|var| var.scaled(-1)).collect_vec(),
                -rhs - 1,
                200,
            );
        }

        // If ub(Ax) <= b, we know that Ax <= b is always true. Then, we can simply state
        // Ax < b, or Ax <= b - 1
        if ub_lhs_init == self.rhs {
            return LinearLessOrEqual::new_expl(lhs.clone(), rhs - 1, 201);
        }

        // We have two options: either Ax < b or Ax > b.
        // We use an aux variable p to represent Ax < b <=> p.
        // This allows us to construct two possible inequalities:
        // * Ax <= b - 1 + M(1-p)
        // * Ax >= b + 1 - Mp
        //
        // Rewriting to linear inequalities leads to
        // * Ax + Mp <= b - 1 + M
        // * -Ax - Mp <= -b - 1

        // Construct auxiliary variable
        // Ax < b, or Ax <= b - 1
        let p = assignments.new_linked_aux_variable(LinearLessOrEqual::new(lhs.clone(), rhs - 1));

        // Compute big_m
        let lb_lhs = lhs.lb_initial(assignments) as i32;
        let ub_lhs = lhs.ub_initial(assignments) as i32;

        // Option 1: Ax + Mp <= b - 1 + M
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m_opt_1 = (ub_lhs - rhs + 1).max(0);
        let mut opt_1_lhs = lhs.clone();
        opt_1_lhs.0.push(p.scaled(big_m_opt_1));
        let opt_1 =
            LinearLessOrEqual::new_expl(opt_1_lhs.non_zero_scale(), rhs - 1 + big_m_opt_1, 202);

        // Option 2: -Ax - Mp <= -b - 1
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m_opt_2 = (-lb_lhs + rhs + 1).max(0);
        let mut opt_2_lhs = lhs.clone();
        opt_2_lhs.iter_mut().for_each(|var| *var = var.scaled(-1));
        opt_2_lhs.0.push(p.scaled(-big_m_opt_2));
        let opt_2 = LinearLessOrEqual::new_expl(opt_2_lhs.non_zero_scale(), -rhs - 1, 203);

        // Observation: an equality with positive variables, e.g. x + y <= 1 can only decrease the
        // upper bounds. If we want to increase the lower bounds, we need to make them
        // negative. This is what option 2 does.
        if increase_lower_bound_inequality {
            opt_2
        } else {
            opt_1
        }
    }
}

impl<Var> Propagator for LinearNotEqualPropagator<Var>
where
    Var: IntegerVariable + 'static,
{
    fn priority(&self) -> u32 {
        1
    }

    fn name(&self) -> &str {
        "LinearNe"
    }

    fn notify(
        &mut self,
        context: PropagationContext,
        local_id: LocalId,
        _event: OpaqueDomainEvent,
    ) -> EnqueueDecision {
        // If the updated term is fixed then we update the number of fixed variables
        self.number_of_fixed_terms += 1;
        // We update the value of the left-hand side with the value of the newly fixed variable
        self.fixed_lhs += context.lower_bound(&self.terms[local_id.unpack() as usize]);

        // Either the number of fixed variables is the number of terms - 1 in which case we can
        // propagate if it has not been updated before; if it has been updated then we don't need to
        // remove the value from its domain again.
        let can_propagate = self.number_of_fixed_terms == self.terms.len() - 1
            && !self.unfixed_variable_has_been_updated;
        // Otherwise the number of fixed variables is equal to the number of terms in the following
        // cases:
        // - Either we can report a conflict
        // - Or the sum of the values of the left-hand side is inaccurate and we should recalculate
        let is_conflicting_or_outdated = self.number_of_fixed_terms == self.terms.len()
            && (self.should_recalculate_lhs || self.fixed_lhs == self.rhs);
        if can_propagate || is_conflicting_or_outdated {
            EnqueueDecision::Enqueue
        } else {
            EnqueueDecision::Skip
        }
    }

    fn notify_backtrack(
        &mut self,
        _context: PropagationContext,
        local_id: LocalId,
        event: OpaqueDomainEvent,
    ) -> EnqueueDecision {
        if matches!(
            self.terms[local_id.unpack() as usize].unpack_event(event),
            IntDomainEvent::Assign
        ) {
            pumpkin_assert_simple!(
                self.number_of_fixed_terms >= 1,
                "The number of fixed terms should never be negative"
            );
            // An assign has been undone, we can decrease the
            // number of fixed variables
            self.number_of_fixed_terms -= 1;

            // We don't keep track of the old bound to which this variable was assigned so we simply
            // indicate that our lhs is out-of-date
            self.should_recalculate_lhs = true;
        } else {
            // A removal has been undone
            pumpkin_assert_moderate!(matches!(
                self.terms[local_id.unpack() as usize].unpack_event(event),
                IntDomainEvent::Removal
            ));

            // We set the flag whether the unfixed variable has been updated
            self.unfixed_variable_has_been_updated = false;
        }

        EnqueueDecision::Skip
    }

    fn initialise_at_root(
        &mut self,
        context: &mut PropagatorInitialisationContext,
    ) -> Result<(), PropagationReason> {
        self.terms.iter().enumerate().for_each(|(i, x_i)| {
            let _ = context.register(x_i.clone(), DomainEvents::ASSIGN, LocalId::from(i as u32));
            let _ = context.register_for_backtrack_events(
                x_i.clone(),
                DomainEvents::create_with_int_events(enum_set!(
                    IntDomainEvent::Assign | IntDomainEvent::Removal
                )),
                LocalId::from(i as u32),
            );
        });

        self.recalculate_fixed_variables(context.as_readonly());
        self.check_for_conflict(context.as_readonly())?;
        Ok(())
    }

    fn detect_inconsistency_mut(
        &self,
        context: &mut PropagationContextMut,
    ) -> Option<PropagationReason> {
        let (fixed_lhs, number_of_fixed_terms) =
            self.calculate_fixed_variables(context.as_readonly());
        self.is_conflicting_mut(context, fixed_lhs, number_of_fixed_terms)
            .err()
    }

    fn detect_inconsistency(&self, context: PropagationContext) -> Option<PropagationReason> {
        let (fixed_lhs, number_of_fixed_terms) = self.calculate_fixed_variables(context);
        self.is_conflicting(context, fixed_lhs, number_of_fixed_terms)
            .err()
    }

    fn propagate(&mut self, context: &mut PropagationContextMut) -> PropagationStatusCP {
        // If the left-hand side is out of date then we simply recalculate from scratch; we only do
        // this when we can propagate or check for a conflict
        if self.should_recalculate_lhs && self.number_of_fixed_terms >= self.terms.len() - 1 {
            self.recalculate_fixed_variables(context.as_readonly());
            self.should_recalculate_lhs = false;
        }

        pumpkin_assert_extreme!(self.is_propagator_state_consistent(context.as_readonly()));

        // If there is only 1 unfixed variable, then we can propagate
        if self.number_of_fixed_terms == self.terms.len() - 1 {
            pumpkin_assert_simple!(!self.should_recalculate_lhs);

            // The value which would cause a conflict if the current variable would be set equal to
            // this
            let value_to_remove = self.rhs - self.fixed_lhs;

            // We find the value which is unfixed
            // We could make use of a sparse-set to determine this, if necessary
            let unfixed_x_i = self
                .terms
                .iter()
                .position(|x_i| !context.is_fixed(x_i))
                .unwrap();

            if context.contains(&self.terms[unfixed_x_i], value_to_remove) {
                // We keep track of whether we have removed the value which could cause a conflict
                // from the unfixed variable
                self.unfixed_variable_has_been_updated = true;

                let curr_lb = self.terms[unfixed_x_i].lower_bound(context.assignments);
                let curr_ub = self.terms[unfixed_x_i].upper_bound(context.assignments);

                let increased_lb = if curr_lb == value_to_remove {
                    true
                } else if curr_ub == value_to_remove {
                    false
                } else {
                    // Given that we're creating a hole that doesn't change the bounds,
                    // it doesn't really matter which one we choose
                    false
                };

                let reason_linleq =
                    new_explanation!(self.create_reason(context.assignments, increased_lb));

                context.remove(
                    &self.terms[unfixed_x_i],
                    value_to_remove,
                    (
                        self.terms
                            .iter()
                            .enumerate()
                            .filter(|&(i, _)| i != unfixed_x_i)
                            .map(|(_, x_i)| predicate![x_i == context.lower_bound(x_i)])
                            .collect::<PropositionalConjunction>(),
                        reason_linleq,
                    ),
                )?;
            }
        } else if self.number_of_fixed_terms == self.terms.len() {
            pumpkin_assert_simple!(!self.should_recalculate_lhs);
            // Otherwise we check for a conflict
            self.check_for_conflict_mut(context)?;
        }

        Ok(())
    }

    fn debug_propagate_from_scratch(
        &self,
        context: &mut PropagationContextMut,
    ) -> PropagationStatusCP {
        let num_fixed = self
            .terms
            .iter()
            .filter(|&x_i| context.is_fixed(x_i))
            .count();
        if num_fixed < self.terms.len() - 1 {
            return Ok(());
        }

        let lhs = self
            .terms
            .iter()
            .map(|var| {
                if context.is_fixed(var) {
                    context.lower_bound(var) as i64
                } else {
                    0
                }
            })
            .sum::<i64>();

        if num_fixed == self.terms.len() - 1 {
            let value_to_remove = self.rhs as i64 - lhs;

            let unfixed_x_i = self
                .terms
                .iter()
                .position(|x_i| !context.is_fixed(x_i))
                .unwrap();

            let reason = self
                .terms
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != unfixed_x_i)
                .map(|(_, x_i)| predicate![x_i == context.lower_bound(x_i)])
                .collect::<PropositionalConjunction>();
            context.remove(
                &self.terms[unfixed_x_i],
                value_to_remove
                    .try_into()
                    .expect("Expected to be able to fit i64 into i32"),
                reason,
            )?;
        } else if num_fixed == self.terms.len() && lhs == self.rhs.into() {
            let failure_reason: PropositionalConjunction = self
                .terms
                .iter()
                .map(|x_i| predicate![x_i == context.lower_bound(x_i)])
                .collect();

            let ineq = new_explanation!(self.create_reason(context.assignments, true));

            return Err((failure_reason, ineq).into());
        }

        Ok(())
    }
}

impl<Var: IntegerVariable + 'static> LinearNotEqualPropagator<Var> {
    fn calculate_fixed_variables(&self, context: PropagationContext) -> (i32, usize) {
        self.terms
            .iter()
            .fold((0, 0), |(fixed_lhs, number_of_fixed_terms), term| {
                if context.is_fixed(term) {
                    (
                        fixed_lhs + context.lower_bound(term),
                        number_of_fixed_terms + 1,
                    )
                } else {
                    (fixed_lhs, number_of_fixed_terms)
                }
            })
    }

    /// This method is used to calculate the fixed left-hand side of the equation and keep track of
    /// the number of fixed variables.
    ///
    /// Note that this method always sets the `unfixed_variable_has_been_updated` to true; this
    /// might be too lenient as it could be the case that synchronisation does not lead to the
    /// re-adding of the removed value.
    fn recalculate_fixed_variables(&mut self, context: PropagationContext) {
        self.unfixed_variable_has_been_updated = false;
        (self.fixed_lhs, self.number_of_fixed_terms) = self.calculate_fixed_variables(context);
    }

    fn is_conflicting(
        &self,
        context: PropagationContext,
        fixed_lhs: i32,
        number_of_fixed_terms: usize,
    ) -> Result<(), PropagationReason> {
        if number_of_fixed_terms == self.terms.len() && fixed_lhs == self.rhs {
            let failure_reason: PropositionalConjunction = self
                .terms
                .iter()
                .map(|x_i| predicate![x_i == context.lower_bound(x_i)])
                .collect();

            return Err(failure_reason.into());
        }

        Ok(())
    }

    fn is_conflicting_mut(
        &self,
        context: &mut PropagationContextMut,
        fixed_lhs: i32,
        number_of_fixed_terms: usize,
    ) -> Result<(), PropagationReason> {
        self.is_conflicting(context.as_readonly(), fixed_lhs, number_of_fixed_terms)
            .map_err(|reason| {
                // As there is a conflict, Ax = b. This means that Ax >= b also holds,
                // meaning the auxiliary variable is false at this point. Therefore,
                // pick the explanation that is currently conflicting (option 2)
                let failure_reason_linleq =
                    new_explanation!(self.create_reason(context.assignments, true));
                (reason.0, failure_reason_linleq).into()
            })
    }

    /// Determines whether a conflict has occurred and calculate the reason for the conflict
    fn check_for_conflict_mut(
        &self,
        context: &mut PropagationContextMut,
    ) -> Result<(), PropagationReason> {
        pumpkin_assert_simple!(!self.should_recalculate_lhs);
        self.is_conflicting_mut(context, self.fixed_lhs, self.number_of_fixed_terms)
    }

    /// Determines whether a conflict has occurred and calculate the reason for the conflict
    fn check_for_conflict(&self, context: PropagationContext) -> Result<(), PropagationReason> {
        pumpkin_assert_simple!(!self.should_recalculate_lhs);
        self.is_conflicting(context, self.fixed_lhs, self.number_of_fixed_terms)
    }

    /// Checks whether the number of fixed terms is equal to the number of fixed terms in the
    /// provided [`PropagationContext`] and whether the value of the fixed lhs is the same as in the
    /// provided [`PropagationContext`].
    fn is_propagator_state_consistent(&self, context: PropagationContext) -> bool {
        let expected_number_of_fixed_terms = self
            .terms
            .iter()
            .filter(|&x_i| context.is_fixed(x_i))
            .count();
        let number_of_fixed_terms_is_correct =
            self.number_of_fixed_terms == expected_number_of_fixed_terms;

        let expected_fixed_lhs = self
            .terms
            .iter()
            .filter_map(|x_i| {
                if context.is_fixed(x_i) {
                    Some(context.lower_bound(x_i))
                } else {
                    None
                }
            })
            .sum();
        let lhs_is_outdated_or_correct =
            self.should_recalculate_lhs || self.fixed_lhs == expected_fixed_lhs;

        number_of_fixed_terms_is_correct && lhs_is_outdated_or_correct
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basic_types::Inconsistency;
    use crate::conjunction;
    use crate::engine::test_solver::TestSolver;
    use crate::engine::variables::TransformableVariable;

    #[test]
    fn test_value_is_removed() {
        let mut solver = TestSolver::default();
        let x = solver.new_variable(2, 2);
        let y = solver.new_variable(1, 5);

        let propagator = solver
            .new_propagator(LinearNotEqualPropagator::new(
                [x.scaled(1), y.scaled(-1)].into(),
                0,
            ))
            .expect("non-empty domain");

        solver.propagate(propagator).expect("non-empty domain");

        solver.assert_bounds(x, 2, 2);
        solver.assert_bounds(y, 1, 5);
        assert!(!solver.contains(y, 2));
    }

    #[test]
    fn test_empty_domain_is_detected() {
        let mut solver = TestSolver::default();
        let x = solver.new_variable(2, 2);
        let y = solver.new_variable(2, 2);

        let err = solver
            .new_propagator(LinearNotEqualPropagator::new(
                [x.scaled(1), y.scaled(-1)].into(),
                0,
            ))
            .expect_err("empty domain");

        let expected: Inconsistency = conjunction!([x == 2] & [y == 2]).into();
        assert_eq!(expected, err);
    }

    #[test]
    fn explanation_for_propagation() {
        let mut solver = TestSolver::default();
        let x = solver.new_variable(2, 2).scaled(1);
        let y = solver.new_variable(1, 5).scaled(-1);

        let propagator = solver
            .new_propagator(LinearNotEqualPropagator::new([x, y].into(), 0))
            .expect("non-empty domain");

        solver.propagate(propagator).expect("non-empty domain");

        let reason = solver.get_reason_int(predicate![y != -2]);

        assert_eq!(conjunction!([x == 2]), reason);
    }

    #[test]
    fn satisfied_constraint_does_not_trigger_conflict() {
        let mut solver = TestSolver::default();
        let x = solver.new_variable(0, 3);
        let y = solver.new_variable(0, 3);

        let propagator = solver
            .new_propagator(LinearNotEqualPropagator::new(
                [x.scaled(1), y.scaled(-1)].into(),
                0,
            ))
            .expect("non-empty domain");

        solver.remove(x, 0).expect("non-empty domain");
        solver.remove(x, 2).expect("non-empty domain");
        solver.remove(x, 3).expect("non-empty domain");

        solver.remove(y, 0).expect("non-empty domain");
        solver.remove(y, 1).expect("non-empty domain");
        solver.remove(y, 2).expect("non-empty domain");

        solver.notify_propagator(propagator);

        solver.propagate(propagator).expect("non-empty domain");
    }
}
