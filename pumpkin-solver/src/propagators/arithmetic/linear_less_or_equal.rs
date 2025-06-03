use crate::basic_types::linear_less_or_equal::LinearLessOrEqual;
use crate::basic_types::PropagationReason;
use crate::basic_types::PropagationStatusCP;
use crate::create_statistics_struct;
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
use crate::new_explanation;
use crate::predicate;
use crate::pumpkin_assert_simple;
use crate::statistics::analysis_log::LearnedItemId;
use crate::statistics::Statistic;
use crate::statistics::StatisticLogger;

create_statistics_struct!(LinearLessOrEqualStatistics {
    number_of_executions: u64,
    number_of_propagations: u64,
    number_of_pb_vars: u64,
});

/// Propagator for the constraint `reif => \sum x_i <= c`.
#[derive(Clone, Debug)]
pub(crate) struct LinearLessOrEqualPropagator<Var> {
    x: Box<[Var]>,
    c: i32,

    /// The lower bound of the sum of the left-hand side. This is incremental state.
    lower_bound_left_hand_side: i64,
    /// The value at index `i` is the bound for `x[i]`.
    current_bounds: Box<[i32]>,

    is_learned: bool,
    is_unlinked_aux: bool,
    errored_initially: bool,
    statistics: LinearLessOrEqualStatistics,
}

impl<Var: 'static> LinearLessOrEqualPropagator<Var>
where
    Var: IntegerVariable,
{
    pub(crate) fn new(x: Box<[Var]>, c: i32) -> Self {
        let current_bounds = vec![0; x.len()].into();

        // incremental state will be properly initialized in `Propagator::initialise_at_root`.
        LinearLessOrEqualPropagator::<Var> {
            x,
            c,
            lower_bound_left_hand_side: 0,
            current_bounds,

            is_learned: false,
            is_unlinked_aux: false,
            errored_initially: false,
            statistics: LinearLessOrEqualStatistics::default(),
        }
    }

    pub(crate) fn new_unlinked_aux(x: Box<[Var]>, c: i32) -> Self {
        let mut new = Self::new(x, c);
        new.is_unlinked_aux = true;
        new
    }

    pub(crate) fn new_learned(x: Box<[Var]>, c: i32, assignments: &Assignments) -> Self {
        let mut new = Self::new(x, c);
        new.is_learned = true;

        new.statistics.number_of_pb_vars = new
            .x
            .iter()
            .filter(|v| {
                let lb_pb = v.lower_bound(assignments) == 0;
                let ub_pb = v.upper_bound(assignments) == 1;
                lb_pb && ub_pb
            })
            .count() as u64;

        new
    }

    /// Recalculates the incremental state from scratch.
    fn recalculate_incremental_state(&mut self, context: PropagationContext) {
        self.lower_bound_left_hand_side = self
            .x
            .iter()
            .map(|var| context.lower_bound(var) as i64)
            .sum();

        self.current_bounds
            .iter_mut()
            .enumerate()
            .for_each(|(index, bound)| {
                *bound = context.lower_bound(&self.x[index]);
            });
    }

    fn create_propagation_reason(
        &self,
        context: PropagationContext,
        skip_i: Option<usize>,
    ) -> PropagationReason {
        let conjunction = self
            .x
            .iter()
            .enumerate()
            .filter_map(|(j, var)| {
                if let Some(i) = skip_i {
                    if i == j {
                        return None;
                    }
                }
                Some(predicate![var >= context.lower_bound(var)])
            })
            .collect();

        let inequality = new_explanation!(LinearLessOrEqual::new_expl(&self.x, self.c, 100));
        PropagationReason(conjunction, inequality)
    }

    fn initialise_base(
        &mut self,
        context: &mut PropagatorInitialisationContext,
    ) -> Result<(), PropagationReason> {
        self.recalculate_incremental_state(context.as_readonly());

        if let Some(conjunction) = self.detect_inconsistency(context.as_readonly()) {
            Err(conjunction)
        } else {
            Ok(())
        }
    }

    fn log_conflict(&self, context: &mut PropagationContextMut) {
        if self.is_learned {
            if let Some(log) = &mut context.analysis_log {
                log.log_error(
                    LearnedItemId::Inequality(context.propagator_id),
                    context.assignments,
                );
            }
        }
    }
}

impl<Var: 'static> Propagator for LinearLessOrEqualPropagator<Var>
where
    Var: IntegerVariable,
{
    fn initialise_at_root(
        &mut self,
        context: &mut PropagatorInitialisationContext,
    ) -> Result<(), PropagationReason> {
        self.x.iter().enumerate().for_each(|(i, x_i)| {
            let _ = context.register(
                x_i.clone(),
                DomainEvents::LOWER_BOUND,
                LocalId::from(i as u32),
            );
        });

        self.initialise_base(context)
    }

    fn initialise_at_non_root(
        &mut self,
        context: &mut PropagatorInitialisationContext,
    ) -> Result<(), PropagationReason> {
        self.x.iter().enumerate().for_each(|(i, x_i)| {
            let _ = context.register_unchecked(
                x_i.clone(),
                DomainEvents::LOWER_BOUND,
                LocalId::from(i as u32),
            );

            let _ = context.register_for_backtrack_events(
                x_i.clone(),
                DomainEvents::LOWER_BOUND,
                LocalId::from(i as u32),
            );
        });

        self.initialise_base(context)
    }

    fn detect_inconsistency(&self, context: PropagationContext) -> Option<PropagationReason> {
        if (self.c as i64) < self.lower_bound_left_hand_side {
            Some(self.create_propagation_reason(context, None))
        } else {
            None
        }
    }

    fn notify(
        &mut self,
        context: PropagationContext,
        local_id: LocalId,
        _event: OpaqueDomainEvent,
    ) -> EnqueueDecision {
        let index = local_id.unpack() as usize;

        let x_i = &self.x[index];
        let old_bound = self.current_bounds[index];
        let new_bound = context.lower_bound(x_i);

        // It could be that this propagator is created at the same point that this notify is called.
        // At that point, the bounds are still the same, but the "notify" is not invalid.
        let is_first_unlinked_exec =
            self.is_unlinked_aux && self.statistics.number_of_executions == 0;

        if !is_first_unlinked_exec {
            pumpkin_assert_simple!(
                old_bound < new_bound,
                "propagator should only be triggered when lower bounds are tightened, old_bound={old_bound}, new_bound={new_bound}"
            );
        }

        self.current_bounds[index] = new_bound;
        self.lower_bound_left_hand_side += (new_bound - old_bound) as i64;

        EnqueueDecision::Enqueue
    }

    fn synchronise(&mut self, context: PropagationContext) -> EnqueueDecision {
        self.recalculate_incremental_state(context);
        EnqueueDecision::Skip
    }

    fn priority(&self) -> u32 {
        if self.is_unlinked_aux {
            0
        } else {
            1
        }
    }

    fn name(&self) -> &str {
        "LinearLeq"
    }

    fn propagate(&mut self, context: &mut PropagationContextMut) -> PropagationStatusCP {
        self.statistics.number_of_executions += 1;

        if let Some(conjunction) = self.detect_inconsistency_mut(context) {
            if self.statistics.number_of_executions == 1 {
                self.errored_initially = true;
            }
            self.log_conflict(context);
            return Err(conjunction.into());
        }

        let lower_bound_left_hand_side = match i32::try_from(self.lower_bound_left_hand_side) {
            Ok(bound) => bound,
            Err(_) if self.lower_bound_left_hand_side.is_positive() => {
                // We cannot fit the `lower_bound_left_hand_side` into an i32 due to an
                // overflow (hence the check that the lower-bound on the left-hand side is
                // positive)
                //
                // This means that the lower-bounds of the current variables will always be
                // higher than the right-hand side (with a maximum value of i32). We thus
                // return a conflict
                self.log_conflict(context);
                return Err(self
                    .create_propagation_reason(context.as_readonly(), None)
                    .into());
            }
            Err(_) => {
                // We cannot fit the `lower_bound_left_hand_side` into an i32 due to an
                // underflow
                //
                // This means that the constraint is always satisfied
                return Ok(());
            }
        };

        for (i, x_i) in self.x.iter().enumerate() {
            let Ok(lower_bound) = i32::try_from(context.lower_bound_i64(x_i)) else {
                // If an individual item overflows, we cannot do much... Explanations would have to
                // use the 64-bit lower bound as well, which is not useful.
                return Ok(());
            };

            // We still need to check lb_lhs being i32 such that we can be sure
            // this will not overflow.
            let bound_i64 =
                (self.c as i64) - (lower_bound_left_hand_side as i64 - lower_bound as i64);
            let bound = match i32::try_from(bound_i64) {
                Ok(bound) => bound,
                Err(_) if bound_i64.is_positive() => {
                    // We cannot fit the `bound` into an i32 due to an
                    // overflow (hence the check that the bound is positive)
                    //
                    // This means that the upper-bound of the current variable will never be
                    // higher than the bound (with a maximum value of i32). This means
                    // that the upper-bound doesn't have to be updated.
                    continue;
                }
                Err(_) => {
                    // We cannot fit the `bound` into an i32 due to an
                    // underflow
                    //
                    // This means that the upper-bound of the current variable is always higher
                    // than this bound. This means that there is a conflict, as the upper
                    // bound would have to be set to i32::MIN.
                    self.log_conflict(context);
                    return Err(self
                        .create_propagation_reason(context.as_readonly(), Some(i))
                        .into());
                }
            };

            if context.upper_bound_i64(x_i) > bound as i64 {
                self.statistics.number_of_propagations += 1;

                if self.is_learned {
                    if let Some(log) = &mut context.analysis_log {
                        log.log_propagation(
                            LearnedItemId::Inequality(context.propagator_id),
                            context.assignments,
                        )
                    }
                }

                let reason = self.create_propagation_reason(context.as_readonly(), Some(i));
                context.set_upper_bound(x_i, bound, reason)?;
            }
        }

        if self.is_learned {
            // Observation: a learned constraint is only added if it is able to propagate or errors
            // after backtracking. If a learned constraint contains auxiliary variables, it will
            // only be able to propagate once they are fixed, which should happen
            // somewhere in the first propagation loop. Therefore, we check that all
            // auxiliaries are fixed before imposing the condition that this constraint
            // should propagate.
            //
            // UPDATE: the updated priorities should make auxiliaries propagate first!
            pumpkin_assert_simple!(
                self.errored_initially || self.statistics.number_of_propagations >= 1,
                "A newly learned constraint should always propagate!"
            );
        }

        Ok(())
    }

    fn log_statistics(&self, statistic_logger: StatisticLogger) {
        if self.is_learned {
            self.statistics.log(statistic_logger);
        }
    }

    fn debug_propagate_from_scratch(
        &self,
        context: &mut PropagationContextMut,
    ) -> PropagationStatusCP {
        let Ok(lower_bound_left_hand_side) = self
            .x
            .iter()
            .map(|var| i32::try_from(context.lower_bound_i64(var)))
            .try_fold(0i64, |acc, v| v.map(|v| acc + v as i64))
        else {
            // If an individual item overflows, we cannot do much... Explanations would have to
            // use the 64-bit lower bound as well, which is not useful.
            return Ok(());
        };

        let lower_bound_left_hand_side = match i32::try_from(lower_bound_left_hand_side) {
            Ok(bound) => bound,
            Err(_) if lower_bound_left_hand_side.is_positive() => {
                // We cannot fit the `lower_bound_left_hand_side` into an i32 due to an
                // overflow (hence the check that the lower-bound on the left-hand side is
                // positive)
                //
                // This means that the lower-bounds of the current variables will always be
                // higher than the right-hand side (with a maximum value of i32). We thus
                // return a conflict
                return Err(self
                    .create_propagation_reason(context.as_readonly(), None)
                    .into());
            }
            Err(_) => {
                // We cannot fit the `lower_bound_left_hand_side` into an i32 due to an
                // underflow
                //
                // This means that the constraint is always satisfied
                return Ok(());
            }
        };

        for (i, x_i) in self.x.iter().enumerate() {
            // We still need to check lb_lhs being i32 such that we can be sure
            // this will not overflow.
            let bound_i64 = (self.c as i64)
                - (lower_bound_left_hand_side as i64 - context.lower_bound(x_i) as i64);
            let bound = match i32::try_from(bound_i64) {
                Ok(bound) => bound,
                Err(_) if bound_i64.is_positive() => {
                    // We cannot fit the `bound` into an i32 due to an
                    // overflow (hence the check that the bound is positive)
                    //
                    // This means that the upper-bound of the current variable will never be
                    // higher than the bound (with a maximum value of i32). This means
                    // that the upper-bound doesn't have to be updated.
                    continue;
                }
                Err(_) => {
                    // We cannot fit the `bound` into an i32 due to an
                    // underflow
                    //
                    // This means that the upper-bound of the current variable is always higher
                    // than this bound. This means that there is a conflict, as the upper
                    // bound would have to be set to i32::MIN.
                    return Err(self
                        .create_propagation_reason(context.as_readonly(), Some(i))
                        .into());
                }
            };

            if context.upper_bound_i64(x_i) > bound as i64 {
                let reason = self.create_propagation_reason(context.as_readonly(), Some(i));
                context.set_upper_bound(x_i, bound, reason)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conjunction;
    use crate::engine::test_solver::TestSolver;

    #[test]
    fn test_bounds_are_propagated() {
        let mut solver = TestSolver::default();
        let x = solver.new_variable(1, 5);
        let y = solver.new_variable(0, 10);

        let propagator = solver
            .new_propagator(LinearLessOrEqualPropagator::new([x, y].into(), 7))
            .expect("no empty domains");

        solver.propagate(propagator).expect("non-empty domain");

        solver.assert_bounds(x, 1, 5);
        solver.assert_bounds(y, 0, 6);
    }

    #[test]
    fn test_explanations() {
        let mut solver = TestSolver::default();
        let x = solver.new_variable(1, 5);
        let y = solver.new_variable(0, 10);

        let propagator = solver
            .new_propagator(LinearLessOrEqualPropagator::new([x, y].into(), 7))
            .expect("no empty domains");

        solver.propagate(propagator).expect("non-empty domain");

        let reason = solver.get_reason_int(predicate![y <= 6]);

        assert_eq!(conjunction!([x >= 1]), reason);
    }

    #[test]
    fn overflow_leads_to_conflict() {
        let mut solver = TestSolver::default();

        let x = solver.new_variable(i32::MAX, i32::MAX);
        let y = solver.new_variable(1, 1);

        let _ = solver
            .new_propagator(LinearLessOrEqualPropagator::new([x, y].into(), i32::MAX))
            .expect_err("Expected overflow to be detected");
    }

    #[test]
    fn underflow_leads_to_no_propagation() {
        let mut solver = TestSolver::default();

        let x = solver.new_variable(i32::MIN, i32::MIN);
        let y = solver.new_variable(-1, -1);

        let _ = solver
            .new_propagator(LinearLessOrEqualPropagator::new([x, y].into(), i32::MIN))
            .expect("Expected no error to be detected");
    }
}
