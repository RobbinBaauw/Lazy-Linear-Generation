use itertools::Itertools;

use crate::basic_types::linear_less_or_equal::FilterNonZero;
use crate::basic_types::linear_less_or_equal::LinearLessOrEqual;
use crate::basic_types::PropagationReason;
use crate::basic_types::PropagationStatusCP;
use crate::basic_types::PropositionalConjunction;
use crate::conjunction;
use crate::engine::cp::propagation::ReadDomains;
use crate::engine::domain_events::DomainEvents;
use crate::engine::propagation::LocalId;
use crate::engine::propagation::PropagationContextMut;
use crate::engine::propagation::Propagator;
use crate::engine::propagation::PropagatorId;
use crate::engine::propagation::PropagatorInitialisationContext;
use crate::engine::propagation::PropagatorVarId;
use crate::engine::variables::IntegerVariable;
use crate::engine::Assignments;
use crate::engine::AuxiliaryVariable;
use crate::if_llg;
use crate::new_explanation;
use crate::predicate;
use crate::variables::DomainId;
use crate::variables::TransformableVariable;

/// Bounds-consistent propagator which enforces `max(array) = rhs`. Can be constructed through
/// [`MaximumConstructor`].
#[derive(Clone, Debug)]
pub(crate) struct MaximumPropagator<ElementVar, Rhs> {
    array: Box<[ElementVar]>,
    rhs: Rhs,
}

impl<ElementVar: IntegerVariable, Rhs: IntegerVariable> MaximumPropagator<ElementVar, Rhs> {
    pub(crate) fn new(array: Box<[ElementVar]>, rhs: Rhs) -> Self {
        MaximumPropagator { array, rhs }
    }

    fn create_unlinked_auxes<'a>(
        &self,
        assignments: &'a mut Assignments,
        propagator_id: PropagatorId,
    ) -> &'a Vec<DomainId> {
        if assignments
            .unlinked_aux_variables_for_prop(propagator_id)
            .is_none()
        {
            let array_index_auxes = self
                .array
                .iter()
                .map(|_| assignments.new_unlinked_aux_variable(propagator_id))
                .collect_vec();

            // Add watchers for the auxes, as they are also used for propagations
            array_index_auxes.iter().enumerate().for_each(|(i, aux)| {
                let propagator_var = PropagatorVarId {
                    propagator: propagator_id,
                    variable: LocalId::from(i as u32),
                };
                assignments.watch_aux_variable(*aux, propagator_var, DomainEvents::BOUNDS);
            });

            // Constraint 1: only one aux is true, low side
            // sum aux_i <= 1
            let one_aux_lt_lhs = array_index_auxes
                .iter()
                .map(|aux| aux.flatten())
                .collect_vec();
            let one_aux_lt = LinearLessOrEqual::new(one_aux_lt_lhs, 1);

            // Constraint 2: only one aux is true, high side
            // sum aux_i => 1, or -sum aux <= -1
            let one_aux_gt_lhs = array_index_auxes
                .iter()
                .map(|aux| aux.flatten().scaled(-1))
                .collect_vec();
            let one_aux_gt = LinearLessOrEqual::new(one_aux_gt_lhs, -1);

            assignments
                .new_auxiliaries
                .push(AuxiliaryVariable::Unlinked(
                    array_index_auxes,
                    vec![one_aux_lt, one_aux_gt],
                ))
        }

        assignments
            .unlinked_aux_variables_for_prop(propagator_id)
            .unwrap()
    }

    fn create_conditional_explanation(
        &self,
        var_idx: usize,
        assignments: &mut Assignments,
        propagator_id: PropagatorId,
    ) -> LinearLessOrEqual {
        // With the auxiliary variables, we can represent rhs <= a_i if a_i is the largest.
        // This simply becomes rhs - a_i <= M(1-p_i), or rhs - a_i + Mp_i <= M
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let p_i = self.create_unlinked_auxes(assignments, propagator_id)[var_idx];
        let var = &self.array[var_idx];

        let rhs_ub = self.rhs.upper_bound_initial(assignments);
        let var_lb = var.lower_bound_initial(assignments);
        let big_m = (rhs_ub - var_lb).max(0);

        LinearLessOrEqual::new_expl(
            vec![
                self.rhs.flatten(),
                var.flatten().scaled(-1),
                p_i.scaled(big_m),
            ]
            .non_zero_scale(),
            big_m,
            300,
        )
    }
}

impl<ElementVar: IntegerVariable + 'static, Rhs: IntegerVariable + 'static> Propagator
    for MaximumPropagator<ElementVar, Rhs>
{
    fn initialise_at_root(
        &mut self,
        context: &mut PropagatorInitialisationContext,
    ) -> Result<(), PropagationReason> {
        self.array
            .iter()
            .cloned()
            .enumerate()
            .for_each(|(idx, var)| {
                let _ =
                    context.register(var.clone(), DomainEvents::BOUNDS, LocalId::from(idx as u32));
            });
        let _ = context.register(
            self.rhs.clone(),
            DomainEvents::BOUNDS,
            LocalId::from(self.array.len() as u32),
        );

        Ok(())
    }

    fn priority(&self) -> u32 {
        1
    }

    fn name(&self) -> &str {
        "Maximum"
    }

    fn debug_propagate_from_scratch(
        &self,
        context: &mut PropagationContextMut,
    ) -> PropagationStatusCP {
        // This is the constraint that is being propagated:
        // max(a_0, a_1, ..., a_{n-1}) = rhs

        let rhs_ub = context.upper_bound(&self.rhs);

        let mut max_ub = context.upper_bound(&self.array[0]);
        let mut ub_var_idx = 0;

        let mut max_lb = context.lower_bound(&self.array[0]);
        let mut lb_reason = predicate![self.array[0] >= max_lb];
        let mut lb_var = &self.array[0];

        for (var_idx, var) in self.array.iter().enumerate() {
            // Rule 1.
            // UB(a_i) <= UB(rhs)
            context.set_upper_bound(
                var,
                rhs_ub,
                (
                    conjunction!([self.rhs <= rhs_ub]),
                    // a_i <= b, or a_i - b <= 0
                    new_explanation!(LinearLessOrEqual::new_expl(
                        vec![var.flatten().scaled(1), self.rhs.flatten().scaled(-1)],
                        0,
                        301,
                    )),
                ),
            )?;

            let var_lb = context.lower_bound(var);
            let var_ub = context.upper_bound(var);

            if var_lb > max_lb {
                max_lb = var_lb;
                lb_reason = predicate![var >= var_lb];
                lb_var = var;
            }

            if var_ub > max_ub {
                max_ub = var_ub;
                ub_var_idx = var_idx;
            }
        }
        // Rule 2.
        // LB(rhs) >= max{LB(a_i)}.
        context.set_lower_bound(
            &self.rhs,
            max_lb,
            (
                PropositionalConjunction::from(lb_reason),
                // Again, b >= a_i, or a_i - b <= 0
                new_explanation!(LinearLessOrEqual::new_expl(
                    vec![lb_var.flatten().scaled(1), self.rhs.flatten().scaled(-1)],
                    0,
                    302
                )),
            ),
        )?;

        // Rule 3.
        // UB(rhs) <= max{UB(a_i)}.
        // Note that this implicitly also covers the rule:
        // 'if LB(rhs) > UB(a_i) for all i, then conflict'.
        if rhs_ub > max_ub {
            let ub_reason: PropositionalConjunction = self
                .array
                .iter()
                .map(|var| predicate![var <= max_ub])
                .collect();

            let ub_reason_linleq = new_explanation!(self.create_conditional_explanation(
                ub_var_idx,
                context.assignments,
                context.propagator_id,
            ));

            context.set_upper_bound(&self.rhs, max_ub, (ub_reason, ub_reason_linleq))?;
        }

        // Rule 4.
        // If there is only one variable with UB(a_i) >= LB(rhs),
        // then the bounds for rhs and that variable should be intersected.
        let rhs_lb = context.lower_bound(&self.rhs);
        let mut propagating_variable: Option<(usize, &ElementVar)> = None;
        let mut propagation_reason = PropositionalConjunction::default();
        for (var_idx, var) in self.array.iter().enumerate() {
            if context.upper_bound(var) >= rhs_lb {
                if propagating_variable.is_none() {
                    propagating_variable = Some((var_idx, var));
                } else {
                    propagating_variable = None;
                    break;
                }
            } else {
                propagation_reason.add(predicate![var <= rhs_lb - 1]);
            }
        }
        // If there is exactly one variable UB(a_i) >= LB(rhs), then the propagating variable is
        // Some. In that case, intersect the bounds of that variable and the rhs. Given previous
        // rules, only the lower bound of the propagated variable needs to be propagated.
        if let Some((propagating_variable_idx, propagating_variable)) = propagating_variable {
            let var_lb = context.lower_bound(propagating_variable);
            if var_lb < rhs_lb {
                propagation_reason.add(predicate![self.rhs >= rhs_lb]);

                let propagation_reason_linleq = new_explanation!(self
                    .create_conditional_explanation(
                        propagating_variable_idx,
                        context.assignments,
                        context.propagator_id,
                    ));

                context.set_lower_bound(
                    propagating_variable,
                    rhs_lb,
                    (propagation_reason, propagation_reason_linleq),
                )?;
            }
        }

        let _ = if_llg!({
            // Rule 5: if auxiliary variable a_i is 0, do nothing. If aux variable a_i is 1,
            // set rhs <= a_i. This rule prevents the need to create all these conditions up front.
            let auxes = self
                .create_unlinked_auxes(context.assignments, context.propagator_id)
                .clone();

            let true_aux = new_explanation!(auxes
                .iter()
                .enumerate()
                .find(|(_, aux)| context.is_fixed(*aux) && context.lower_bound(*aux) == 1))
            .flatten();

            if let Some((true_var_idx, true_aux)) = true_aux {
                // Match the LB & UB of true_aux & rhs
                let true_var = &self.array[true_var_idx];

                let true_var_ub = context.upper_bound(true_var);
                let rhs_lb = context.lower_bound(&self.rhs);

                let linleq_reason = new_explanation!(self.create_conditional_explanation(
                    true_var_idx,
                    context.assignments,
                    context.propagator_id,
                ));

                context.set_upper_bound(
                    &self.rhs,
                    true_var_ub,
                    (
                        conjunction!([true_aux >= 1] & [true_var <= true_var_ub]),
                        linleq_reason.clone(),
                    ),
                )?;
                context.set_lower_bound(
                    true_var,
                    rhs_lb,
                    (
                        conjunction!([true_aux >= 1] & [self.rhs >= rhs_lb]),
                        linleq_reason,
                    ),
                )?;
            } else {
                // Rule 6: if no aux is fixed and there is any fixed variable a_i for
                // which holds lb(a_i) = ub(rhs), select this variable to be the maximum
                let rhs_ub = context.upper_bound(&self.rhs);
                for (var_idx, var) in self.array.iter().enumerate() {
                    if context.is_fixed(var) && context.lower_bound(var) >= rhs_ub {
                        let var_aux = &auxes[var_idx];

                        // We need a custom linear explanation here, in the form:
                        // p_i >= 1 if [a_i >= rhs_ub] & [rhs <= rhs_ub]
                        //
                        // We express this in two auxiliaries r_1 = [a_i >= rhs_ub] and
                        // r_2 = [rhs <= rhs_ub]
                        //
                        // p_i >= 1 - M(1-r_1) - M(1-r_2)
                        // p_i - Mr_1 - Mr_2 >= 1 - 2M
                        // -p_i + Mr_1 + Mr_2 <= -1 + 2M
                        //
                        // Since p_i is binary, M = 1

                        let r_1 = var.min_aux(context.assignments, rhs_ub);
                        let r_2 = self.rhs.max_aux(context.assignments, rhs_ub);

                        context.set_lower_bound(
                            var_aux,
                            1,
                            (
                                conjunction!([var >= rhs_ub] & [self.rhs <= rhs_ub]),
                                LinearLessOrEqual::new_expl(
                                    vec![var_aux.scaled(-1), r_1.flatten(), r_2.flatten()],
                                    1,
                                    303,
                                ),
                            ),
                        )?;

                        break;
                    }
                }
            }
        });

        Ok(())
    }

    fn create_unlinked_auxiliaries(
        &self,
        assignments: &mut Assignments,
        propagator_id: PropagatorId,
    ) -> Option<Vec<DomainId>> {
        Some(
            self.create_unlinked_auxes(assignments, propagator_id)
                .clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::test_solver::TestSolver;

    #[test]
    fn upper_bound_of_rhs_matches_maximum_upper_bound_of_array_at_initialise() {
        let mut solver = TestSolver::default();

        let a = solver.new_variable(1, 3);
        let b = solver.new_variable(1, 4);
        let c = solver.new_variable(1, 5);

        let rhs = solver.new_variable(1, 10);

        let _ = solver
            .new_propagator(MaximumPropagator::new([a, b, c].into(), rhs))
            .expect("no empty domain");

        solver.assert_bounds(rhs, 1, 5);

        let reason = solver.get_reason_int(predicate![rhs <= 5]);
        assert_eq!(conjunction!([a <= 5] & [b <= 5] & [c <= 5]), reason);
    }

    #[test]
    fn lower_bound_of_rhs_is_maximum_of_lower_bounds_in_array() {
        let mut solver = TestSolver::default();

        let a = solver.new_variable(3, 10);
        let b = solver.new_variable(4, 10);
        let c = solver.new_variable(5, 10);

        let rhs = solver.new_variable(1, 10);

        let _ = solver
            .new_propagator(MaximumPropagator::new([a, b, c].into(), rhs))
            .expect("no empty domain");

        solver.assert_bounds(rhs, 5, 10);

        let reason = solver.get_reason_int(predicate![rhs >= 5]);
        assert_eq!(conjunction!([c >= 5]), reason);
    }

    #[test]
    fn upper_bound_of_all_array_elements_at_most_rhs_max_at_initialise() {
        let mut solver = TestSolver::default();

        let array = (1..=5)
            .map(|idx| solver.new_variable(1, 4 + idx))
            .collect::<Box<_>>();

        let rhs = solver.new_variable(1, 3);

        let _ = solver
            .new_propagator(MaximumPropagator::new(array.clone(), rhs))
            .expect("no empty domain");

        for var in array.iter() {
            solver.assert_bounds(*var, 1, 3);
            let reason = solver.get_reason_int(predicate![var <= 3]);
            assert_eq!(conjunction!([rhs <= 3]), reason);
        }
    }

    #[test]
    fn single_variable_propagate() {
        let mut solver = TestSolver::default();

        let array = (1..=5)
            .map(|idx| solver.new_variable(1, 1 + 10 * idx))
            .collect::<Box<_>>();

        let rhs = solver.new_variable(45, 60);

        let _ = solver
            .new_propagator(MaximumPropagator::new(array.clone(), rhs))
            .expect("no empty domain");

        solver.assert_bounds(*array.last().unwrap(), 45, 51);
        solver.assert_bounds(rhs, 45, 51);
    }
}
