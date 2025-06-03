use crate::basic_types::linear_less_or_equal::FilterNonZero;
use crate::basic_types::linear_less_or_equal::LinearLessOrEqual;
use crate::basic_types::Inconsistency;
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
use crate::pumpkin_assert_simple;
use crate::variables::IntegerVariable;
use crate::variables::Literal;
use crate::variables::TransformableVariable;

/// Propagator for the constraint `r -> p`, where `r` is a Boolean literal and `p` is an arbitrary
/// propagator.
///
/// When a propagator is reified, it will only propagate whenever `r` is set to true. However, if
/// the propagator implements [`Propagator::detect_inconsistency`], the result of that method may
/// be used to propagate `r` to false. If that method is not implemented, `r` will never be
/// propagated to false.
#[derive(Clone, Debug)]
pub(crate) struct ReifiedPropagator<WrappedPropagator> {
    propagator: WrappedPropagator,
    reification_literal: Literal,
    /// An inconsistency that is identified by `propagator`.
    inconsistency: Option<PropagationReason>,
    /// The formatted name of the propagator.
    name: String,
    /// The `LocalId` of the reification literal. Is guaranteed to be a larger ID than any of the
    /// registered ids of the wrapped propagator.
    reification_literal_id: LocalId,
}

impl<WrappedPropagator: Propagator> ReifiedPropagator<WrappedPropagator> {
    pub(crate) fn new(propagator: WrappedPropagator, reification_literal: Literal) -> Self {
        let name = format!("Reified({})", propagator.name());
        ReifiedPropagator {
            reification_literal,
            propagator,
            inconsistency: None,
            name,
            reification_literal_id: LocalId::from(0), /* Place-holder, will be set in
                                                       * `initialise_at_root` */
        }
    }
}

impl ReifiedPropagator<()> {
    pub(crate) fn add_reified_literal_to_explanation(
        reification_literal: Literal,
        linleq: &Option<LinearLessOrEqual>,
        assignments: &Assignments,
    ) -> Option<LinearLessOrEqual> {
        let Some(linleq) = linleq else { return None };

        // Add reification literal to linear explanation. Similar to the other reified propagator,
        // there are two options for optionally using explanation Ax <= b if r is true.
        // So r -> Ax <= b can be reflected by either:
        // * Ax <= b + M(1-r)
        // * Ax > b - Mr, or Ax >= b + 1 - Mr, or -Ax <= -b - 1 + Mr
        //
        // Rewriting to linear inequalities leads to
        // * Ax + Mr <= b + M
        // * -Ax - Mr <= -b - 1
        //
        // Note, in this case we only need the first option, since we only decrease r's upper bound
        let rhs = linleq.rhs;

        let big_m = (linleq.lhs.ub_initial(assignments) as i32 - rhs).max(0);

        // Ax + Mr <= b + M
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let mut lhs = linleq.lhs.clone();
        lhs.0.push(reification_literal.flatten().scaled(big_m));

        Some(LinearLessOrEqual::new_expl(
            lhs.non_zero_scale(),
            rhs + big_m,
            linleq.explanation_id.unwrap_or(0) + 1000,
        ))
    }
}

impl<WrappedPropagator: Propagator> Propagator for ReifiedPropagator<WrappedPropagator> {
    fn notify(
        &mut self,
        context: PropagationContext,
        local_id: LocalId,
        event: OpaqueDomainEvent,
    ) -> EnqueueDecision {
        if local_id < self.reification_literal_id {
            let decision = self.propagator.notify(context, local_id, event);
            self.filter_enqueue_decision(context, decision)
        } else {
            pumpkin_assert_simple!(local_id == self.reification_literal_id);
            EnqueueDecision::Enqueue
        }
    }

    fn notify_backtrack(
        &mut self,
        context: PropagationContext,
        local_id: LocalId,
        event: OpaqueDomainEvent,
    ) -> EnqueueDecision {
        if local_id < self.reification_literal_id {
            self.propagator.notify_backtrack(context, local_id, event)
        } else {
            pumpkin_assert_simple!(local_id == self.reification_literal_id);
            EnqueueDecision::Skip
        }
    }

    fn initialise_at_root(
        &mut self,
        context: &mut PropagatorInitialisationContext,
    ) -> Result<(), PropagationReason> {
        // Since we cannot propagate here, we store a conflict which the wrapped propagator
        // identifies at the root, and propagate the reification literal to false in the
        // `propagate` method.
        if let Err(conjunction) = self.propagator.initialise_at_root(context) {
            self.inconsistency = Some(conjunction);
        }

        self.reification_literal_id = context.get_next_local_id();

        let _ = context.register(
            self.reification_literal,
            DomainEvents::BOUNDS,
            self.reification_literal_id,
        );

        Ok(())
    }

    fn priority(&self) -> u32 {
        self.propagator.priority()
    }

    fn synchronise(&mut self, context: PropagationContext) -> EnqueueDecision {
        // We remove the inconsistency upon backtracking since it might be invalid now
        self.inconsistency = None;

        self.propagator.synchronise(context)
    }

    fn propagate(&mut self, context: &mut PropagationContextMut) -> PropagationStatusCP {
        if let Some(mut conjunction) = self.inconsistency.take() {
            conjunction.1 = ReifiedPropagator::add_reified_literal_to_explanation(
                self.reification_literal,
                &conjunction.1,
                context.assignments,
            );
            context.assign_literal(&self.reification_literal, false, conjunction)?;
        }

        self.propagate_reification(context)?;

        if context.is_literal_true(&self.reification_literal) {
            context.with_reification(self.reification_literal);

            let result = self.propagator.propagate(context);

            self.map_propagation_status(result, context.assignments)?;
        }

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn debug_propagate_from_scratch(
        &self,
        context: &mut PropagationContextMut,
    ) -> PropagationStatusCP {
        self.propagate_reification(context)?;

        if context.is_literal_true(&self.reification_literal) {
            context.with_reification(self.reification_literal);

            let result = self.propagator.debug_propagate_from_scratch(context);

            self.map_propagation_status(result, context.assignments)?;
        }

        Ok(())
    }
}

impl<Prop: Propagator> ReifiedPropagator<Prop> {
    fn map_propagation_status(
        &self,
        mut status: PropagationStatusCP,
        assignments: &Assignments,
    ) -> PropagationStatusCP {
        if let Err(Inconsistency::Conflict(ref mut explanation)) = status {
            // Add reification literal to nogood explanation
            explanation
                .0
                .add(self.reification_literal.get_true_predicate());

            explanation.1 = ReifiedPropagator::add_reified_literal_to_explanation(
                self.reification_literal,
                &explanation.1,
                assignments,
            );
        }
        status
    }

    fn propagate_reification(&self, context: &mut PropagationContextMut<'_>) -> PropagationStatusCP
    where
        Prop: Propagator,
    {
        if !context.is_literal_fixed(&self.reification_literal) {
            if let Some(mut conjunction) = self.propagator.detect_inconsistency_mut(context) {
                conjunction.1 = ReifiedPropagator::add_reified_literal_to_explanation(
                    self.reification_literal,
                    &conjunction.1,
                    context.assignments,
                );
                context.assign_literal(&self.reification_literal, false, conjunction)?;
            }
        }

        Ok(())
    }

    fn find_inconsistency(&mut self, context: PropagationContext<'_>) -> bool {
        if self.inconsistency.is_none() {
            self.inconsistency = self.propagator.detect_inconsistency(context);
        }

        self.inconsistency.is_some()
    }

    fn filter_enqueue_decision(
        &mut self,
        context: PropagationContext<'_>,
        decision: EnqueueDecision,
    ) -> EnqueueDecision {
        if decision == EnqueueDecision::Skip {
            // If the original propagator skips then we always skip
            return EnqueueDecision::Skip;
        }

        if context.is_literal_true(&self.reification_literal) {
            // If the propagator would have enqueued and the literal is true then the reified
            // propagator is also enqueued
            return EnqueueDecision::Enqueue;
        }

        if !context.is_literal_false(&self.reification_literal) && self.find_inconsistency(context)
        {
            // Or the literal is not false already and there the propagator has found an
            // inconsistency (i.e. we should and can propagate the reification variable)
            return EnqueueDecision::Enqueue;
        }

        EnqueueDecision::Skip
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basic_types::Inconsistency;
    use crate::basic_types::PropagationReason;
    use crate::conjunction;
    use crate::engine::test_solver::TestSolver;
    use crate::predicate;
    use crate::predicates::PropositionalConjunction;
    use crate::variables::DomainId;

    #[test]
    fn a_detected_inconsistency_is_given_as_reason_for_propagating_reification_literal_to_false() {
        let mut solver = TestSolver::default();

        let reification_literal = solver.new_literal();
        let a = solver.new_variable(1, 1);
        let b = solver.new_variable(2, 2);

        let triggered_conflict = conjunction!([a == 1] & [b == 2]);
        let t1 = triggered_conflict.clone();
        let t2 = triggered_conflict.clone();

        let _ = solver
            .new_propagator(ReifiedPropagator::new(
                GenericPropagator::new(
                    move |_: &mut PropagationContextMut| Err(t1.clone().into()),
                    move |_: PropagationContext| Some(t2.clone().into()),
                    |_: &mut PropagatorInitialisationContext| Ok(()),
                ),
                reification_literal,
            ))
            .expect("no conflict");

        assert!(solver.is_literal_false(reification_literal));

        let reason = solver.get_reason_bool(reification_literal, false);
        assert_eq!(reason, triggered_conflict);
    }

    #[test]
    fn a_true_literal_is_added_to_reason_for_propagation() {
        let mut solver = TestSolver::default();

        let reification_literal = solver.new_literal();
        let var = solver.new_variable(1, 5);

        let propagator = solver
            .new_propagator(ReifiedPropagator::new(
                GenericPropagator::new(
                    move |ctx: &mut PropagationContextMut| {
                        ctx.set_lower_bound(&var, 3, conjunction!())?;
                        Ok(())
                    },
                    |_: PropagationContext| None,
                    |_: &mut PropagatorInitialisationContext| Ok(()),
                ),
                reification_literal,
            ))
            .expect("no conflict");

        solver.assert_bounds(var, 1, 5);

        let _ = solver.set_literal(reification_literal, true);
        solver.propagate(propagator).expect("no conflict");

        solver.assert_bounds(var, 3, 5);
        let reason = solver.get_reason_int(predicate![var >= 3]);
        assert_eq!(
            reason,
            PropositionalConjunction::from(reification_literal.get_true_predicate())
        );
    }

    #[test]
    fn a_true_literal_is_added_to_a_conflict_conjunction() {
        let mut solver = TestSolver::default();

        let reification_literal = solver.new_literal();
        let _ = solver.set_literal(reification_literal, true);

        let var = solver.new_variable(1, 1);

        let inconsistency = solver
            .new_propagator(ReifiedPropagator::new(
                GenericPropagator::new(
                    move |_: &mut PropagationContextMut| Err(conjunction!([var >= 1]).into()),
                    |_: PropagationContext| None,
                    |_: &mut PropagatorInitialisationContext| Ok(()),
                ),
                reification_literal,
            ))
            .expect_err("eagerly triggered the conflict");

        match inconsistency {
            Inconsistency::Conflict(conflict_nogood) => {
                assert_eq!(
                    conflict_nogood.0,
                    PropositionalConjunction::from(vec![
                        reification_literal.get_true_predicate(),
                        predicate![var >= 1]
                    ])
                )
            }

            other => panic!("Inconsistency {other:?} is not expected."),
        }
    }

    #[test]
    fn a_root_level_conflict_propagates_reification_literal() {
        let mut solver = TestSolver::default();

        let reification_literal = solver.new_literal();
        let var = solver.new_variable(1, 1);

        let _ = solver
            .new_propagator(ReifiedPropagator::new(
                GenericPropagator::new(
                    |_: &mut PropagationContextMut| Ok(()),
                    |_: PropagationContext| None,
                    move |_: &mut PropagatorInitialisationContext| {
                        Err(conjunction!([var >= 0]).into())
                    },
                ),
                reification_literal,
            ))
            .expect("eagerly triggered the conflict");

        assert!(solver.is_literal_false(reification_literal));
    }

    #[test]
    fn notify_propagator_is_enqueued_if_inconsistency_can_be_detected() {
        let mut solver = TestSolver::default();

        let reification_literal = solver.new_literal();
        let var = solver.new_variable(1, 5);

        let propagator = solver
            .new_propagator(ReifiedPropagator::new(
                GenericPropagator::new(
                    |_: &mut PropagationContextMut| Ok(()),
                    move |context: PropagationContext| {
                        if context.is_fixed(&var) {
                            Some(conjunction!([var == 5]).into())
                        } else {
                            None
                        }
                    },
                    |_: &mut PropagatorInitialisationContext| Ok(()),
                )
                .with_variables(&[var]),
                reification_literal,
            ))
            .expect("No conflict expected");

        let enqueue = solver.increase_lower_bound_and_notify(propagator, 0, var, 5);
        assert!(matches!(enqueue, EnqueueDecision::Enqueue))
    }

    struct GenericPropagator<Propagation, ConsistencyCheck, Init> {
        propagation: Propagation,
        consistency_check: ConsistencyCheck,
        init: Init,
        variables_to_register: Vec<DomainId>,
    }

    impl<Propagation, ConsistencyCheck, Init> Propagator
        for GenericPropagator<Propagation, ConsistencyCheck, Init>
    where
        Propagation: Fn(&mut PropagationContextMut) -> PropagationStatusCP + 'static,
        ConsistencyCheck: Fn(PropagationContext) -> Option<PropagationReason> + 'static,
        Init: Fn(&mut PropagatorInitialisationContext) -> Result<(), PropagationReason> + 'static,
    {
        fn name(&self) -> &str {
            "Generic Propagator"
        }

        fn debug_propagate_from_scratch(
            &self,
            context: &mut PropagationContextMut,
        ) -> PropagationStatusCP {
            (self.propagation)(context)
        }

        fn detect_inconsistency(&self, context: PropagationContext) -> Option<PropagationReason> {
            (self.consistency_check)(context)
        }

        fn initialise_at_root(
            &mut self,
            context: &mut PropagatorInitialisationContext,
        ) -> Result<(), PropagationReason> {
            self.variables_to_register
                .iter()
                .enumerate()
                .for_each(|(index, variable)| {
                    let _ = context.register(
                        *variable,
                        DomainEvents::ANY_INT,
                        LocalId::from(index as u32),
                    );
                });
            (self.init)(context)
        }
    }

    impl<Propagation, ConsistencyCheck, Init> GenericPropagator<Propagation, ConsistencyCheck, Init>
    where
        Propagation: Fn(&mut PropagationContextMut) -> PropagationStatusCP,
        ConsistencyCheck: Fn(PropagationContext) -> Option<PropagationReason>,
        Init: Fn(&mut PropagatorInitialisationContext) -> Result<(), PropagationReason>,
    {
        pub(crate) fn new(
            propagation: Propagation,
            consistency_check: ConsistencyCheck,
            init: Init,
        ) -> Self {
            GenericPropagator {
                propagation,
                consistency_check,
                init,
                variables_to_register: vec![],
            }
        }

        pub(crate) fn with_variables(mut self, variables: &[DomainId]) -> Self {
            // Necessary for ensuring that the local IDs are correct when notifying
            self.variables_to_register = variables.into();
            self
        }
    }
}
