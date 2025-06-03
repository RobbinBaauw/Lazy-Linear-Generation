use crate::basic_types::linear_less_or_equal::FilterNonZero;
use crate::basic_types::linear_less_or_equal::LinearLessOrEqual;
use crate::basic_types::PropagationReason;
use crate::basic_types::PropagationStatusCP;
use crate::engine::opaque_domain_event::OpaqueDomainEvent;
use crate::engine::propagation::EnqueueDecision;
use crate::engine::propagation::LocalId;
use crate::engine::propagation::PropagationContext;
use crate::engine::propagation::PropagationContextMut;
use crate::engine::propagation::Propagator;
use crate::engine::propagation::PropagatorInitialisationContext;
use crate::engine::propagation::ReadDomains;
use crate::engine::Assignments;
use crate::engine::DomainEvents;
use crate::new_explanation;
use crate::predicate;
use crate::predicates::PropositionalConjunction;
use crate::variables::DomainId;
use crate::variables::IntegerVariable;
use crate::variables::Literal;
use crate::variables::TransformableVariable;

#[derive(Clone, Debug)]
pub struct LinearInequalityLiteralPropagator {
    linear_inequality: LinearLessOrEqual,
    linear_inequality_inverse: LinearLessOrEqual,
    literal: Literal,
}

impl LinearInequalityLiteralPropagator {
    pub fn new(linear_inequality: LinearLessOrEqual, literal: DomainId) -> Self {
        let linear_inequality_inverse = linear_inequality.invert();

        LinearInequalityLiteralPropagator {
            linear_inequality,
            linear_inequality_inverse,
            literal: Literal::new(literal),
        }
    }

    fn get_propagation_reason_constraint(
        &self,
        assignments: &Assignments,
        increase_lower_bound_inequality: bool,
    ) -> LinearLessOrEqual {
        // We're linking Ax <= b <-> p, meaning we have two equations:
        // * Ax <= b + M(1-p)
        // * Ax > b - Mp, or Ax >= b + 1 - Mp, or -Ax <= -b - 1 + Mp
        //
        // Rewriting to linear inequalities leads to
        // * Ax + Mp <= b + M
        // * -Ax - Mp <= -b - 1
        //
        // Determining M: we need M to be sufficiently large, and then as small as possible.
        // Assume the same equations in which M has to take effect:
        // * Ax - b <= M
        // * -Ax + b + 1 <= M
        //
        // We can find the value for M by finding the maximal value of the LHS now (using initial
        // domains):
        // * ub(Ax) - b <= M
        // * -lb(Ax) + b + 1 <= M
        //
        // We take the maximum of both found M's to find the final M.
        // If M is negative, we do not need it, so we have found a global constraint and can just
        // set M to 0

        let lb_lhs = self.linear_inequality.lhs.lb_initial(assignments) as i32;
        let ub_lhs = self.linear_inequality.lhs.ub_initial(assignments) as i32;
        let rhs = self.linear_inequality.rhs;

        // Option 1: Ax + Mp <= b + M
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m_opt_1 = (ub_lhs - rhs).max(0);
        let mut opt_1_lhs = self.linear_inequality.lhs.clone();
        opt_1_lhs
            .0
            .push(self.literal.domain_id().scaled(big_m_opt_1));
        let opt_1 = LinearLessOrEqual::new_expl(opt_1_lhs.non_zero_scale(), rhs + big_m_opt_1, 800);

        // Option 2: -Ax - Mp <= -b - 1
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m_opt_2 = (-lb_lhs + rhs + 1).max(0);
        let mut opt_2_lhs = self.linear_inequality.lhs.clone();
        opt_2_lhs.iter_mut().for_each(|var| *var = var.scaled(-1));
        opt_2_lhs
            .0
            .push(self.literal.domain_id().scaled(-big_m_opt_2));
        let opt_2 = LinearLessOrEqual::new_expl(opt_2_lhs.non_zero_scale(), -rhs - 1, 801);

        if increase_lower_bound_inequality {
            opt_2
        } else {
            opt_1
        }
    }
}

impl Propagator for LinearInequalityLiteralPropagator {
    fn name(&self) -> &str {
        "LinearInequalityLiteralPropagator"
    }

    fn debug_propagate_from_scratch(
        &self,
        context: &mut PropagationContextMut,
    ) -> PropagationStatusCP {
        // It could be that the trail is completely empty (case in debug tests).
        // In that case, just take index 0
        let trail_position = context.assignments.num_trail_entries().max(1) - 1;

        match self.linear_inequality.evaluate_at_trail_position(
            context.assignments,
            trail_position,
            false,
        ) {
            Some(true) => {
                let conjunction: PropositionalConjunction = self
                    .linear_inequality
                    .lhs
                    .iter()
                    .map(|var| {
                        predicate![
                            var <= context.upper_bound_at_trail_position(var, trail_position)
                        ]
                    })
                    .collect();

                context.set_lower_bound(
                    &self.literal,
                    1,
                    (
                        conjunction,
                        new_explanation!(
                            self.get_propagation_reason_constraint(context.assignments, true)
                        ),
                    ),
                )?
            }
            Some(false) => {
                let conjunction: PropositionalConjunction = self
                    .linear_inequality
                    .lhs
                    .iter()
                    .map(|var| {
                        predicate![
                            var >= context.lower_bound_at_trail_position(var, trail_position)
                        ]
                    })
                    .collect();

                context.set_upper_bound(
                    &self.literal,
                    0,
                    (
                        conjunction,
                        new_explanation!(
                            self.get_propagation_reason_constraint(context.assignments, false)
                        ),
                    ),
                )?
            }
            None => {}
        };

        // Predicate that can be propagated!
        if context.is_literal_fixed(&self.literal) {
            if context.is_literal_true(&self.literal) {
                for (prop_var, prop_bound) in self.linear_inequality.variables_propagating(
                    context.assignments,
                    trail_position,
                    false,
                ) {
                    let mut conjunction: PropositionalConjunction = self
                        .linear_inequality
                        .lhs
                        .iter()
                        .filter_map(|var| {
                            if *var == prop_var {
                                None
                            } else {
                                let pred = predicate![
                                    var >= context
                                        .lower_bound_at_trail_position(var, trail_position)
                                ];
                                Some(pred)
                            }
                        })
                        .collect();
                    conjunction.add(predicate![self.literal >= 1]);

                    context.set_upper_bound(
                        &prop_var,
                        prop_bound,
                        (
                            conjunction,
                            new_explanation!(
                                self.get_propagation_reason_constraint(context.assignments, false)
                            ),
                        ),
                    )?
                }
            }

            if context.is_literal_false(&self.literal) {
                for (prop_var, prop_bound) in self.linear_inequality_inverse.variables_propagating(
                    context.assignments,
                    trail_position,
                    false,
                ) {
                    let mut conjunction: PropositionalConjunction = self
                        .linear_inequality_inverse
                        .lhs
                        .iter()
                        .filter_map(|var| {
                            if *var == prop_var {
                                None
                            } else {
                                let pred = predicate![
                                    var >= context
                                        .lower_bound_at_trail_position(var, trail_position)
                                ];
                                Some(pred)
                            }
                        })
                        .collect();
                    conjunction.add(predicate![self.literal <= 0]);

                    context.set_upper_bound(
                        &prop_var,
                        prop_bound,
                        (
                            conjunction,
                            new_explanation!(
                                self.get_propagation_reason_constraint(context.assignments, true)
                            ),
                        ),
                    )?
                }
            }
        }

        Ok(())
    }

    fn synchronise(&mut self, _context: PropagationContext) -> EnqueueDecision {
        // This is needed in case there is a conflict when this propagator hasn't been able to
        // propagate yet. In that case, a backtrack is triggered in which the variables of this
        // propagator might not be involved, and this propagator would not trigger. This way, we
        // force it to enqueue.
        EnqueueDecision::Enqueue
    }

    fn notify_backtrack(
        &mut self,
        _: PropagationContext,
        _: LocalId,
        _: OpaqueDomainEvent,
    ) -> EnqueueDecision {
        EnqueueDecision::Enqueue
    }

    fn priority(&self) -> u32 {
        0
    }

    fn initialise_at_root(
        &mut self,
        context: &mut PropagatorInitialisationContext,
    ) -> Result<(), PropagationReason> {
        self.linear_inequality.lhs.iter().for_each(|var| {
            let local_var_id = context.get_next_local_id();
            let _ = context.register_unchecked(*var, DomainEvents::ANY_INT, local_var_id);

            let _ =
                context.register_for_backtrack_events(*var, DomainEvents::ANY_INT, local_var_id);
        });

        let literal_id = context.get_next_local_id();
        let _ = context.register_unchecked(self.literal, DomainEvents::ANY_INT, literal_id);
        let _ =
            context.register_for_backtrack_events(self.literal, DomainEvents::ANY_INT, literal_id);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::basic_types::LinearLessOrEqual;
    use crate::conjunction;
    use crate::engine::test_solver::TestSolver;
    use crate::predicate;
    use crate::propagators::LinearInequalityLiteralPropagator;
    use crate::variables::IntegerVariable;
    use crate::variables::TransformableVariable;

    #[test]
    fn propagate_literal_on_positive_inequality() {
        let mut solver = TestSolver::default();

        let reification_literal = solver.new_literal();

        let a = solver.new_variable(2, 2);
        let b = solver.new_variable(2, 2);

        // 2a + 2b <= 10 ==> propagate
        let ineq = LinearLessOrEqual::new(vec![a.scaled(2), b.scaled(2)], 10);

        let _ = solver
            .new_propagator(LinearInequalityLiteralPropagator::new(
                ineq,
                reification_literal.domain_id(),
            ))
            .expect("no conflict");

        assert!(solver.is_literal_true(reification_literal));

        let reason = solver.get_reason_bool(reification_literal, true);
        assert_eq!(reason, conjunction!([a <= 2] & [b <= 2]));
    }

    #[test]
    fn propagate_literal_on_negative_inequality() {
        let mut solver = TestSolver::default();

        let reification_literal = solver.new_literal();

        let a = solver.new_variable(4, 4);
        let b = solver.new_variable(4, 4);

        // 2a + 2b <= 10 ==> propagate
        let ineq = LinearLessOrEqual::new(vec![a.scaled(2), b.scaled(2)], 10);

        let _ = solver
            .new_propagator(LinearInequalityLiteralPropagator::new(
                ineq,
                reification_literal.domain_id(),
            ))
            .expect("no conflict");

        assert!(solver.is_literal_false(reification_literal));

        let reason = solver.get_reason_bool(reification_literal, false);
        assert_eq!(reason, conjunction!([a >= 4] & [b >= 4]));
    }

    #[test]
    fn dont_propagate_literal_on_uncertain_inequality() {
        let mut solver = TestSolver::default();

        let reification_literal = solver.new_literal();

        let a = solver.new_variable(2, 4);
        let b = solver.new_variable(2, 4);

        // 2a + 2b <= 10 ==> propagate
        let ineq = LinearLessOrEqual::new(vec![a.scaled(2), b.scaled(2)], 10);

        let _ = solver
            .new_propagator(LinearInequalityLiteralPropagator::new(
                ineq,
                reification_literal.domain_id(),
            ))
            .expect("no conflict");

        assert!(
            !solver.is_literal_true(reification_literal)
                && !solver.is_literal_false(reification_literal)
        );
    }

    #[test]
    fn propagate_bounds_on_fixed_true_literal() {
        let mut solver = TestSolver::default();

        let reification_literal = solver.new_literal();
        reification_literal
            .set_lower_bound(&mut solver.assignments, 1, None)
            .expect("no conflict");

        let a = solver.new_variable(1, 8);
        let b = solver.new_variable(1, 8);

        // true => 2a + 2b <= 10
        let ineq = LinearLessOrEqual::new(vec![a.scaled(2), b.scaled(2)], 10);

        let _ = solver
            .new_propagator(LinearInequalityLiteralPropagator::new(
                ineq,
                reification_literal.domain_id(),
            ))
            .expect("no conflict");

        assert_eq!(solver.upper_bound(a), 4);
        assert_eq!(solver.upper_bound(b), 4);

        let reason = solver.get_reason_int(predicate!(a <= 4));
        assert_eq!(reason, conjunction!([reification_literal >= 1] & [b >= 1]));
    }

    #[test]
    fn propagate_bounds_on_fixed_false_literal() {
        let mut solver = TestSolver::default();

        let reification_literal = solver.new_literal();
        reification_literal
            .set_upper_bound(&mut solver.assignments, 0, None)
            .expect("no conflict");

        let a = solver.new_variable(0, 4);
        let b = solver.new_variable(0, 4);

        // true => 2a + 2b <= 10
        let ineq = LinearLessOrEqual::new(vec![a.scaled(2), b.scaled(2)], 10);

        let _ = solver
            .new_propagator(LinearInequalityLiteralPropagator::new(
                ineq,
                reification_literal.domain_id(),
            ))
            .expect("no conflict");

        assert_eq!(solver.lower_bound(a), 2);
        assert_eq!(solver.lower_bound(b), 2);

        let reason = solver.get_reason_int(predicate!(a >= 2));
        assert_eq!(reason, conjunction!([reification_literal <= 0] & [b <= 4]));
    }

    #[test]
    fn dont_propagate_bounds_on_unfixed_literal() {
        let mut solver = TestSolver::default();

        let reification_literal = solver.new_literal();

        let a = solver.new_variable(1, 4);
        let b = solver.new_variable(1, 4);

        // true => 2a + 2b <= 10
        let ineq = LinearLessOrEqual::new(vec![a.scaled(2), b.scaled(2)], 10);

        let _ = solver
            .new_propagator(LinearInequalityLiteralPropagator::new(
                ineq,
                reification_literal.domain_id(),
            ))
            .expect("no conflict");

        assert_eq!(solver.lower_bound(a), 1);
        assert_eq!(solver.upper_bound(b), 4);

        assert_eq!(solver.lower_bound(b), 1);
        assert_eq!(solver.upper_bound(b), 4);
    }
}
