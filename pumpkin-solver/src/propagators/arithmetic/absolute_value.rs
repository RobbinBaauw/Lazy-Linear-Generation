use crate::basic_types::linear_less_or_equal::FilterNonZero;
use crate::basic_types::linear_less_or_equal::LinearLessOrEqual;
use crate::basic_types::PropagationReason;
use crate::basic_types::PropagationStatusCP;
use crate::conjunction;
use crate::engine::cp::propagation::ReadDomains;
use crate::engine::domain_events::DomainEvents;
use crate::engine::propagation::LocalId;
use crate::engine::propagation::PropagationContextMut;
use crate::engine::propagation::Propagator;
use crate::engine::propagation::PropagatorInitialisationContext;
use crate::engine::variables::IntegerVariable;
use crate::engine::Assignments;
use crate::new_explanation;
use crate::variables::TransformableVariable;

/// Propagator for `absolute = |signed|`, where `absolute` and `signed` are integer variables.
///
/// The propagator is bounds consistent wrt signed. That means that if `signed \in {-2, -1, 1, 2}`,
/// the propagator will not propagate `[absolute >= 1]`.
#[derive(Clone, Debug)]
pub(crate) struct AbsoluteValuePropagator<VA, VB> {
    signed: VA,
    absolute: VB,
}

impl<VA: IntegerVariable + 'static, VB: IntegerVariable + 'static> AbsoluteValuePropagator<VA, VB> {
    pub(crate) fn new(signed: VA, absolute: VB) -> Self {
        AbsoluteValuePropagator { signed, absolute }
    }

    fn create_conditional_explanation(
        &self,
        assignments: &mut Assignments,
        increase_lower_bound_inequality: bool,
    ) -> LinearLessOrEqual {
        // We have two options that are sometimes true.

        // (1) we have absolute + signed == 0 (true when signed <= 0).
        // This can be extracted into absolute + signed <= 0 (true when signed <= 0)
        // and absolute + signed >= 0 (always true, added elsewhere)
        //
        // (2) we have -absolute + signed == 0 (true when signed >= 0).
        // This can be extracted into -absolute + signed <= 0 (always true, added elsewhere)
        // and -absolute + signed >= 0 (true when signed >= 0).
        // Equivalent formulation: absolute - signed <= 0
        //
        // So here, we focus on absolute + signed <= 0 and absolute - signed <= 0
        //
        // To represent the conditions <= 0 and >= 0 we actually need 2 variables...
        // TODO Minizinc just doesn't care and uses x >= 0 <=> p and then
        // z-x <= 0 if p, z+x <= 0 if not p... We can investigate later

        let signed_lb = self.signed.lower_bound_initial(assignments);
        let signed_ub = self.signed.upper_bound_initial(assignments);
        let absolute_ub = self.absolute.upper_bound_initial(assignments);

        // Option 1: absolute + signed - M(1-p1) <= 0 with p1 <=> signed <= 0
        // Equivalent to: absolute + signed + Mp1 <= M
        // If M <= 0, it means that ub(signed) <= 0 as ub(abs) >= 0, and this condition would always
        // hold
        let big_m_opt_1 = (absolute_ub + signed_ub).max(0);
        let opt_1 = LinearLessOrEqual::new_expl(
            vec![
                self.absolute.flatten().scaled(1),
                self.signed.flatten().scaled(1),
                self.signed.max_aux(assignments, 0).scaled(big_m_opt_1),
            ]
            .non_zero_scale(),
            big_m_opt_1,
            400,
        );

        // Option 2: absolute - signed - M(1-p2) <= 0 with p2 <=> signed >= 0, or -signed <= 0
        // Equivalent to: absolute - signed + Mp2 <= M
        // If M <= 0, it means that lb(signed) >= ub(abs), meaning lb(signed) >= 0, and this
        // condition would always hold
        let big_m_opt_2 = (absolute_ub - signed_lb).max(0);
        let opt_2 = LinearLessOrEqual::new_expl(
            vec![
                self.absolute.flatten().scaled(1),
                self.signed.flatten().scaled(-1),
                self.signed.min_aux(assignments, 0).scaled(big_m_opt_2),
            ]
            .non_zero_scale(),
            big_m_opt_2,
            401,
        );

        // Pick the best
        if increase_lower_bound_inequality {
            opt_2
        } else {
            opt_1
        }
    }
}

impl<VA: IntegerVariable + 'static, VB: IntegerVariable + 'static> Propagator
    for AbsoluteValuePropagator<VA, VB>
{
    fn initialise_at_root(
        &mut self,
        context: &mut PropagatorInitialisationContext,
    ) -> Result<(), PropagationReason> {
        let _ = context.register(self.signed.clone(), DomainEvents::BOUNDS, LocalId::from(0));
        let _ = context.register(
            self.absolute.clone(),
            DomainEvents::BOUNDS,
            LocalId::from(1),
        );

        Ok(())
    }

    fn priority(&self) -> u32 {
        1
    }

    fn name(&self) -> &str {
        "IntAbs"
    }

    fn debug_propagate_from_scratch(
        &self,
        context: &mut PropagationContextMut,
    ) -> PropagationStatusCP {
        // The bound of absolute may be tightened further during propagation, but it is at least
        // zero at the root.
        context.set_lower_bound(
            &self.absolute,
            0,
            (
                conjunction!(),
                // absolute >= 0, -absolute <= 0
                LinearLessOrEqual::new_expl(vec![self.absolute.flatten().scaled(-1)], 0, 402),
            ),
        )?;

        // Propagating absolute value can be broken into a few cases:
        // - `signed` is sign-fixed (i.e. `upper_bound <= 0` or `lower_bound >= 0`), in which case
        //   the bounds of `signed` can be propagated to `absolute` (taking care of swapping bounds
        //   when the `signed` is negative).
        // - `signed` is not sign-fixed (i.e. `lower_bound <= 0` and `upper_bound >= 0`), in which
        //   case the lower bound of `absolute` cannot be tightened without looking into specific
        //   domain values for `signed`, which we don't do.
        let signed_lb = context.lower_bound(&self.signed);
        let signed_ub = context.upper_bound(&self.signed);

        let signed_absolute_ub = i32::max(signed_lb.abs(), signed_ub.abs());

        let expl_abs_ub =
            new_explanation!(self.create_conditional_explanation(context.assignments, false));
        context.set_upper_bound(
            &self.absolute,
            signed_absolute_ub,
            (
                conjunction!([self.signed >= signed_lb] & [self.signed <= signed_ub]),
                expl_abs_ub,
            ),
        )?;

        if signed_lb > 0 {
            context.set_lower_bound(
                &self.absolute,
                signed_lb,
                (
                    conjunction!([self.signed >= signed_lb]),
                    // absolute >= signed
                    // absolute - signed >= 0
                    // -absolute + signed <= 0
                    new_explanation!(LinearLessOrEqual::new_expl(
                        vec![
                            self.absolute.flatten().scaled(-1),
                            self.signed.flatten().scaled(1),
                        ],
                        0,
                        403,
                    )),
                ),
            )?;
        } else if signed_ub < 0 {
            context.set_lower_bound(
                &self.absolute,
                signed_ub.abs(),
                (
                    conjunction!([self.signed <= signed_ub]),
                    // absolute >= -signed
                    // signed + absolute >= 0
                    // -signed - absolute <= 0
                    new_explanation!(LinearLessOrEqual::new_expl(
                        vec![
                            self.signed.flatten().scaled(-1),
                            self.absolute.flatten().scaled(-1),
                        ],
                        0,
                        404,
                    )),
                ),
            )?;
        }

        let absolute_ub = context.upper_bound(&self.absolute);
        let absolute_lb = context.lower_bound(&self.absolute);
        context.set_lower_bound(
            &self.signed,
            -absolute_ub,
            (
                conjunction!([self.absolute <= absolute_ub]),
                // signed >= -absolute
                // signed + absolute >= 0
                // -signed - absolute <= 0
                new_explanation!(LinearLessOrEqual::new_expl(
                    vec![
                        self.signed.flatten().scaled(-1),
                        self.absolute.flatten().scaled(-1),
                    ],
                    0,
                    405,
                )),
            ),
        )?;
        context.set_upper_bound(
            &self.signed,
            absolute_ub,
            (
                conjunction!([self.absolute <= absolute_ub]),
                // signed <= absolute
                // signed - absolute <= 0
                new_explanation!(LinearLessOrEqual::new_expl(
                    vec![
                        self.signed.flatten().scaled(1),
                        self.absolute.flatten().scaled(-1),
                    ],
                    0,
                    406,
                )),
            ),
        )?;

        if signed_ub <= 0 {
            let expl_signed_lb =
                new_explanation!(self.create_conditional_explanation(context.assignments, false));
            context.set_upper_bound(
                &self.signed,
                -absolute_lb,
                (
                    conjunction!([self.signed <= 0] & [self.absolute >= absolute_lb]),
                    expl_signed_lb,
                ),
            )?;
        } else if signed_lb >= 0 {
            let expl_signed_lb =
                new_explanation!(self.create_conditional_explanation(context.assignments, true));
            context.set_lower_bound(
                &self.signed,
                absolute_lb,
                (
                    conjunction!([self.signed >= 0] & [self.absolute >= absolute_lb]),
                    expl_signed_lb,
                ),
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::test_solver::TestSolver;

    #[test]
    fn absolute_bounds_are_propagated_at_initialise() {
        let mut solver = TestSolver::default();

        let signed = solver.new_variable(-3, 4);
        let absolute = solver.new_variable(-2, 10);

        let _ = solver
            .new_propagator(AbsoluteValuePropagator::new(signed, absolute))
            .expect("no empty domains");

        solver.assert_bounds(absolute, 0, 4);
    }

    #[test]
    fn signed_bounds_are_propagated_at_initialise() {
        let mut solver = TestSolver::default();

        let signed = solver.new_variable(-5, 5);
        let absolute = solver.new_variable(0, 3);

        let _ = solver
            .new_propagator(AbsoluteValuePropagator::new(signed, absolute))
            .expect("no empty domains");

        solver.assert_bounds(signed, -3, 3);
    }

    #[test]
    fn absolute_lower_bound_can_be_strictly_positive() {
        let mut solver = TestSolver::default();

        let signed = solver.new_variable(3, 6);
        let absolute = solver.new_variable(0, 10);

        let _ = solver
            .new_propagator(AbsoluteValuePropagator::new(signed, absolute))
            .expect("no empty domains");

        solver.assert_bounds(absolute, 3, 6);
    }

    #[test]
    fn strictly_negative_signed_value_can_propagate_lower_bound_on_absolute() {
        let mut solver = TestSolver::default();

        let signed = solver.new_variable(-5, -3);
        let absolute = solver.new_variable(1, 5);

        let _ = solver
            .new_propagator(AbsoluteValuePropagator::new(signed, absolute))
            .expect("no empty domains");

        solver.assert_bounds(absolute, 3, 5);
    }

    #[test]
    fn lower_bound_on_absolute_can_propagate_negative_upper_bound_on_signed() {
        let mut solver = TestSolver::default();

        let signed = solver.new_variable(-5, 0);
        let absolute = solver.new_variable(1, 5);

        let _ = solver
            .new_propagator(AbsoluteValuePropagator::new(signed, absolute))
            .expect("no empty domains");

        solver.assert_bounds(signed, -5, -1);
    }

    #[test]
    fn lower_bound_on_absolute_can_propagate_positive_lower_bound_on_signed() {
        let mut solver = TestSolver::default();

        let signed = solver.new_variable(1, 5);
        let absolute = solver.new_variable(3, 5);

        let _ = solver
            .new_propagator(AbsoluteValuePropagator::new(signed, absolute))
            .expect("no empty domains");

        solver.assert_bounds(signed, 3, 5);
    }
}
