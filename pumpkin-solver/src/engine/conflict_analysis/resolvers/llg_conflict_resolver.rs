use std::time::Instant;

use itertools::Itertools;
use log::debug;
use log::trace;
use num::integer::div_ceil;

use crate::basic_types::linear_less_or_equal::LinearLessOrEqual;
use crate::basic_types::moving_averages::MovingAverage;
use crate::basic_types::HashMap;
use crate::engine::conflict_analysis::ConflictAnalysisContext;
use crate::engine::conflict_analysis::ConflictResolveResult;
use crate::engine::conflict_analysis::ConflictResolveResult::Constraint;
use crate::engine::conflict_analysis::ConflictResolveResult::Nogood;
use crate::engine::conflict_analysis::ConflictResolver;
use crate::engine::conflict_analysis::LearnedConstraint;
use crate::engine::conflict_analysis::LearnedNogood;
use crate::engine::nogoods::Lbd;
use crate::engine::propagation::store::PropagatorStore;
use crate::engine::propagation::CurrentNogood;
use crate::engine::propagation::ExplanationContext;
use crate::engine::propagation::Propagator;
use crate::engine::propagation::PropagatorId;
use crate::engine::propagation::PropagatorInitialisationContext;
use crate::engine::Assignments;
use crate::engine::PropagatorQueue;
use crate::engine::ResolutionResolver;
use crate::engine::WatchListCP;
use crate::options::LearningOptions;
use crate::predicates::Predicate;
use crate::propagators::linear_less_or_equal::LinearLessOrEqualPropagator;
use crate::propagators::nogoods::ConflictingCheckBeforeCutStrategy;
use crate::pumpkin_assert_ne_simple;
use crate::pumpkin_assert_simple;
use crate::statistics::analysis_log::ExplanationStats;
use crate::statistics::analysis_log::LearnedInequalityStats;
use crate::statistics::analysis_log::LearnedItemId;
use crate::statistics::analysis_log::LearnedNogoodStats;
use crate::variables::AffineView;
use crate::variables::DomainId;
use crate::variables::IntegerVariable;
use crate::variables::TransformableVariable;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Timer {
    Analysis,
    Cutting,
    CheckingBackjump,
    CheckingOverflow,
    CheckingConflicting,
}

#[derive(Debug, Default)]
pub(crate) struct LLGConflictResolver {
    resolution_resolver: ResolutionResolver,
    learning_options: LearningOptions,
    lbd: Lbd,

    timers: HashMap<Timer, Instant>,
}

#[derive(Debug, Default)]
struct CutSuccess {
    inequality: LinearLessOrEqual,
    skip_early_backjump: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum FallbackReason {
    ProofCompletion = 0,
    NotConflicting = 1,
    Overflow = 2,
    NogoodExplanation = 3,
    NogoodConflict = 4,
    DecisionReached = 5,
    NothingLearned = 6,
}

#[derive(Debug)]
enum CutError {
    NothingLearned,
    Overflow,
    Contradiction,
}

// Timer stuff
impl LLGConflictResolver {
    fn start_timer(&mut self, timer: Timer) {
        let _ = self.timers.insert(timer, Instant::now());
    }

    fn finish_all_timers(&mut self, context: &mut ConflictAnalysisContext) {
        let timers = self.timers.keys().cloned().collect_vec();
        timers.into_iter().for_each(|timer| {
            self.finish_timer(&timer, context);
        });
    }

    fn finish_timer(&mut self, timer: &Timer, context: &mut ConflictAnalysisContext) {
        // If a timer is finished, it must have been running
        let start_time = self.timers[timer];

        let elapsed_ns = start_time.elapsed().as_nanos() as u64;
        match timer {
            Timer::Analysis => {
                context.counters.llg_statistics.llg_time_spent_analysis_ns += elapsed_ns
            }
            Timer::Cutting => {
                context.counters.llg_statistics.llg_time_spent_cutting_ns += elapsed_ns
            }
            Timer::CheckingBackjump => {
                context
                    .counters
                    .llg_statistics
                    .llg_time_spent_checking_backjump_ns += elapsed_ns
            }
            Timer::CheckingOverflow => {
                context
                    .counters
                    .llg_statistics
                    .llg_time_spent_checking_overflow_ns += elapsed_ns
            }
            Timer::CheckingConflicting => {
                context
                    .counters
                    .llg_statistics
                    .llg_time_spent_checking_conflicting_ns += elapsed_ns
            }
        }

        let _ = self.timers.remove(&timer);
    }
}

impl LLGConflictResolver {
    pub(crate) fn with_options(learning_options: LearningOptions) -> Self {
        Self {
            learning_options,
            ..Default::default()
        }
    }

    fn apply_fallback(
        &mut self,
        context: &mut ConflictAnalysisContext,
        reason: FallbackReason,
        used_explanations: Vec<ExplanationStats>,
    ) -> Option<ConflictResolveResult> {
        debug!("==>==> {reason:?}, trying resolution!");
        context.counters.llg_statistics.llg_fallback_used += 1;
        self.finish_all_timers(context);

        if let Some(analysis_log) = &mut context.analysis_log {
            analysis_log.log_analysis_fallback(reason, used_explanations)
        }

        let learned_nogood = self.resolution_resolver.resolve_conflict(context);
        if !self.learning_options.llg_clause_to_inequality {
            return learned_nogood;
        }

        // We cannot do anything with that for linears
        if reason == FallbackReason::ProofCompletion {
            return learned_nogood;
        }

        if let Some(Nogood(learned_nogood)) = &learned_nogood {
            let learned_clause = learned_nogood
                .predicates
                .iter()
                .map(|&predicate| !predicate)
                .collect_vec();

            if let Some(constraint) =
                Self::transform_to_constraint(learned_clause, context.assignments)
            {
                context.counters.llg_statistics.llg_clauses_to_inequalities += 1;

                return Some(Constraint(LearnedConstraint {
                    constraint,
                    backjump_level: learned_nogood.backjump_level,
                }));
            }
        }

        learned_nogood
    }

    fn create_new_propagator(
        propagators: &mut PropagatorStore,
        assignments: &Assignments,
        watch_list_cp: &mut WatchListCP,
        propagator_queue: &mut PropagatorQueue,
        propagator: impl Propagator,
    ) -> PropagatorId {
        let new_pred_prop_id = propagators.alloc(Box::new(propagator), None);
        let new_propagator = &mut propagators[new_pred_prop_id];

        let mut initialisation_context =
            PropagatorInitialisationContext::new(watch_list_cp, new_pred_prop_id, assignments);

        let _ = new_propagator.initialise_at_non_root(&mut initialisation_context);

        propagator_queue.enqueue_propagator(new_pred_prop_id, new_propagator.priority());

        new_pred_prop_id
    }

    fn clause_to_constraint(
        clause: Vec<Predicate>,
        var_mult: i32,
    ) -> (Vec<AffineView<DomainId>>, i32) {
        clause.iter().fold(
            (vec![], -var_mult),
            |(mut prev_lhs, prev_rhs), curr_pred| {
                let is_lb = match curr_pred {
                    Predicate::LowerBound { .. } => true,
                    Predicate::UpperBound { .. } => false,
                    Predicate::NotEqual {
                        not_equal_constant, ..
                    } => *not_equal_constant == 0,
                    Predicate::Equal {
                        equality_constant, ..
                    } => *equality_constant == 0,
                };

                let var_scale = if is_lb { -var_mult } else { var_mult };
                prev_lhs.push(curr_pred.get_domain().scaled(var_scale));

                let rhs_contribution = 1 - is_lb as i32;
                (prev_lhs, prev_rhs + var_mult * rhs_contribution)
            },
        )
    }

    fn transform_to_constraint(
        clause: Vec<Predicate>,
        assignments: &Assignments,
    ) -> Option<LinearLessOrEqual> {
        let non_bools = clause
            .iter()
            .filter(|p| {
                let domain_id = p.get_domain();
                let lb_init = domain_id.lower_bound_initial(assignments);
                let ub_init = domain_id.upper_bound_initial(assignments);
                lb_init != 0 || ub_init != 1
            })
            .collect_vec();

        if non_bools.len() == 1 {
            let bound = non_bools[0];
            let bound_domain = bound.get_domain();

            let clause_part = clause
                .iter()
                .filter(|p| *p != bound)
                .map(|p| *p)
                .collect_vec();

            match bound {
                Predicate::LowerBound { lower_bound, .. } => {
                    // x v y v [a >= 2] with init lb -2 turns into a >= 2 - 4x - 4y
                    let lb_diff = lower_bound - assignments.get_initial_lower_bound(bound_domain);
                    let (mut lhs, rhs) = Self::clause_to_constraint(clause_part, lb_diff);
                    lhs.push(bound.get_domain().scaled(-1));
                    Some(LinearLessOrEqual::new(lhs, -lower_bound + lb_diff + rhs))
                }
                Predicate::UpperBound { upper_bound, .. } => {
                    // x v y v [a <= -2] with init ub 2 turns into a <= -2 + 4x + 4y
                    let ub_diff = assignments.get_initial_upper_bound(bound_domain) - upper_bound;
                    let (mut lhs, rhs) = Self::clause_to_constraint(clause_part, ub_diff);
                    lhs.push(bound.get_domain().scaled(1));
                    Some(LinearLessOrEqual::new(lhs, upper_bound + ub_diff + rhs))
                }
                Predicate::NotEqual { .. } => None,
                Predicate::Equal { .. } => None,
            }
        } else if non_bools.len() == 0 {
            let (lhs, rhs) = Self::clause_to_constraint(clause, 1);
            Some(LinearLessOrEqual::new(lhs, rhs))
        } else {
            None
        }
    }

    fn apply_cut(
        var: DomainId,
        c1: &LinearLessOrEqual,
        c2: &LinearLessOrEqual,
    ) -> Result<CutSuccess, CutError> {
        let c1_scale = c1.lhs.find_variable_scale(var).unwrap();
        let c2_scale = c2.lhs.find_variable_scale(var).unwrap();

        // A pre-condition to apply a cut is that both constraints have 'var'
        // and that they have opposite signs
        pumpkin_assert_ne_simple!(c1_scale.is_positive(), c2_scale.is_positive());

        let g = gcd(c1_scale.abs(), c2_scale.abs());
        let mult_c1 = c2_scale.abs() / g;
        let mult_c2 = c1_scale.abs() / g;

        let mut skip_early_backjump = true;

        let mut c1_sorted = c1
            .lhs
            .iter()
            .map(|var| (var.domain_id(), var.scale))
            .sorted_by_key(|(id, _)| id.id)
            .peekable();
        let mut c2_sorted = c2
            .lhs
            .iter()
            .map(|var| (var.domain_id(), var.scale))
            .sorted_by_key(|(id, _)| id.id)
            .peekable();

        let mut new_lhs: Vec<(DomainId, i32)> = vec![];

        let mult_or_err = |item: Option<&(DomainId, i32)>, mult: i32| {
            item.map(|(id, curr_scale)| {
                let new_scale = mult.checked_mul(*curr_scale).ok_or(CutError::Overflow)?;
                Ok((*id, new_scale))
            })
            .transpose()
        };

        while c1_sorted.peek().is_some() || c2_sorted.peek().is_some() {
            let c1_item = mult_or_err(c1_sorted.peek(), mult_c1)?;
            let c2_item = mult_or_err(c2_sorted.peek(), mult_c2)?;

            match (c1_item, c2_item) {
                (Some((c1_id, c1_scale)), Some((c2_id, _))) if c1_id.id < c2_id.id => {
                    new_lhs.push((c1_id, c1_scale));
                    let _ = c1_sorted.next();
                }
                (Some((c1_id, _)), Some((c2_id, c2_scale))) if c2_id.id < c1_id.id => {
                    new_lhs.push((c2_id, c2_scale));
                    let _ = c2_sorted.next();
                }
                (Some((c1_id, c1_scale)), Some((c2_id, c2_scale))) if c1_id.id == c2_id.id => {
                    // Don't skip early backjump in case there is a clash between variables that
                    // are not 'var'
                    if c1_id != var {
                        skip_early_backjump = false;
                    }

                    let new_scale = c1_scale.checked_add(c2_scale).ok_or(CutError::Overflow)?;
                    if new_scale != 0 {
                        new_lhs.push((c1_id, new_scale));
                    }

                    let _ = c1_sorted.next();
                    let _ = c2_sorted.next();
                }
                (Some((c1_id, c1_scale)), None) => {
                    new_lhs.push((c1_id, c1_scale));
                    let _ = c1_sorted.next();
                }
                (None, Some((c2_id, c2_scale))) => {
                    new_lhs.push((c2_id, c2_scale));
                    let _ = c2_sorted.next();
                }
                _ => unreachable!("Shouldn't be possible"),
            }
        }

        pumpkin_assert_simple!(
            !new_lhs.iter().any(|(k, _)| *k == var),
            "variable not eliminated"
        );

        let c1_rhs_scaled = c1.rhs.checked_mul(mult_c1).ok_or(CutError::Overflow)?;
        let c2_rhs_scaled = c2.rhs.checked_mul(mult_c2).ok_or(CutError::Overflow)?;
        let mut new_rhs = c1_rhs_scaled
            .checked_add(c2_rhs_scaled)
            .ok_or(CutError::Overflow)?;

        if new_lhs.len() == 0 {
            return Err(if new_rhs < 0 {
                CutError::Contradiction
            } else {
                CutError::NothingLearned
            });
        }

        // Normalization
        let mut new_gcd = new_lhs
            .iter()
            .map(|(_, scale)| *scale)
            .reduce(|a, b| gcd(a, b))
            .unwrap_or(new_rhs);
        new_gcd = gcd(new_gcd, new_rhs);

        new_lhs.iter_mut().for_each(|(_, scale)| {
            *scale = div_ceil(*scale, new_gcd);
        });
        new_rhs = div_ceil(new_rhs, new_gcd);

        Ok(CutSuccess {
            inequality: LinearLessOrEqual::new(new_lhs, new_rhs),
            skip_early_backjump,
        })
    }
}

impl ConflictResolver for LLGConflictResolver {
    fn resolve_conflict(
        &mut self,
        context: &mut ConflictAnalysisContext,
    ) -> Option<ConflictResolveResult> {
        self.start_timer(Timer::Analysis);

        if context.is_completing_proof {
            // We do not do proof completion
            return self.apply_fallback(context, FallbackReason::ProofCompletion, vec![]);
        }

        pumpkin_assert_ne_simple!(context.assignments.get_decision_level(), 0);

        let Some(mut conflicting_constraint) = context.get_conflict_constraint() else {
            return self.apply_fallback(context, FallbackReason::NogoodConflict, vec![]);
        };

        let current_decision_level = context.assignments.get_decision_level();
        let mut trail_index = context.assignments.num_trail_entries() - 1;

        debug!(
            "Conflicting constraint: {conflicting_constraint} (expl {}, slack {})",
            conflicting_constraint.explanation_id.unwrap_or(0),
            conflicting_constraint.slack(context.assignments, trail_index, false)
        );

        let mut used_explanations: Vec<ExplanationStats> =
            vec![conflicting_constraint.to_explanation_stats(context.assignments)];

        loop {
            trace!("====== LOOP AT {current_decision_level}/{trail_index}");

            // Find trail entry at which the conflicting constraint is not conflicting anymore
            let trail_entry = context.assignments.get_trail_entry(trail_index);
            let trail_entry_var = trail_entry.predicate.get_domain();

            // When a decision is reached, and we haven't found a conflicting solution yet, skip
            if trail_entry.reason.is_none() {
                return self.apply_fallback(
                    context,
                    FallbackReason::DecisionReached,
                    used_explanations,
                );
            }

            self.start_timer(Timer::CheckingOverflow);
            if conflicting_constraint.overflows(context.assignments, trail_index) {
                return self.apply_fallback(context, FallbackReason::Overflow, used_explanations);
            }
            self.finish_timer(&Timer::CheckingOverflow, context);

            // If the conflicting constraint doesn't contain this variable, go to next level
            if !conflicting_constraint
                .lhs
                .contains_variable(trail_entry_var)
            {
                trace!("==>==> Not containing {trail_entry_var} at {trail_index}, skip");
                trail_index -= 1;
                continue;
            };

            // Once we have found a conflicting trail level, use this level to start our
            // analysis
            self.start_timer(Timer::CheckingConflicting);
            match self.learning_options.llg_conflicting_check_before_cut {
                // What IntSat does, continue until we make the constraint non-conflicting, and
                // then apply the cut to make it conflicting
                ConflictingCheckBeforeCutStrategy::ContinueUntilNonConflicting => {
                    if conflicting_constraint.is_conflicting(context.assignments, trail_index, true)
                    {
                        trail_index -= 1;
                        continue;
                    }
                }
                // Instead of skipping until we have found a decision, immediately return. Should
                // have the same result as the default, just a bit more time efficient.
                ConflictingCheckBeforeCutStrategy::ReturnNonConflicting => {
                    if !conflicting_constraint.is_conflicting(
                        context.assignments,
                        trail_index,
                        true,
                    ) {
                        return self.apply_fallback(
                            context,
                            FallbackReason::NotConflicting,
                            used_explanations,
                        );
                    }
                }
                // Doesn't do any checking, just continues and will do whatever the next check
                // after the cut decides
                ConflictingCheckBeforeCutStrategy::Skip => {
                    // Pass
                }
            }
            self.finish_timer(&Timer::CheckingConflicting, context);

            trace!("==>==> Using {trail_index}");
            trail_index -= 1;

            // Find the scale of the variable of its reason
            let reason_ref = trail_entry
                .reason
                .expect("Cannot be a null reason for propagation.");

            let explanation_context =
                ExplanationContext::new(context.assignments, CurrentNogood::empty());

            let reason = context
                .reason_store
                .get_or_compute(reason_ref, explanation_context, context.propagators)
                .expect("reason reference should not be stale");

            let Some(prop_constraint_expl) = reason.1 else {
                if self.learning_options.llg_continue_on_nogoods {
                    continue;
                }

                return self.apply_fallback(
                    context,
                    FallbackReason::NogoodExplanation,
                    used_explanations,
                );
            };

            trace!(
                "==>==> Merging with {:?}: {prop_constraint_expl}",
                trail_entry.predicate.get_domain()
            );

            // Because a lineq propagator propagates multiple upper bounds at the same time, the
            // last one might not be the one actually causing the conflict. The one that caused
            // the conflict should have a different sign. We search until we find that one.
            let cutting_var = trail_entry.predicate.get_domain();
            let c1_scale = conflicting_constraint
                .lhs
                .find_variable_scale(cutting_var)
                .unwrap();
            let c2_scale = prop_constraint_expl
                .lhs
                .find_variable_scale(cutting_var)
                .unwrap();

            if c1_scale.is_positive() == c2_scale.is_positive() {
                trace!(
                    "==> Not different signs (expl {:?}), retry",
                    prop_constraint_expl.explanation_id
                );
                continue;
            }

            if self.learning_options.llg_skip_high_slack {
                let expl_slack =
                    prop_constraint_expl.slack(context.assignments, trail_index, false);
                if expl_slack != 0 && expl_slack != 1 {
                    context.counters.llg_statistics.llg_skipped_high_slack += 1;
                    continue;
                }
            }

            // Logging this AFTER sign check as this explanation was not relevant for the current
            // conflict
            used_explanations.push(prop_constraint_expl.to_explanation_stats(context.assignments));

            self.start_timer(Timer::Cutting);
            let (new_conflicting_constraint, skip_early_backjump) = match Self::apply_cut(
                cutting_var,
                &conflicting_constraint,
                &prop_constraint_expl,
            ) {
                Err(CutError::NothingLearned) => {
                    return self.apply_fallback(
                        context,
                        FallbackReason::NothingLearned,
                        used_explanations,
                    );
                }
                Err(CutError::Overflow) => {
                    return self.apply_fallback(
                        context,
                        FallbackReason::Overflow,
                        used_explanations,
                    );
                }
                Err(CutError::Contradiction) => {
                    debug!("==>==> Contradiction, unsat!");
                    self.finish_all_timers(context);
                    return Some(Nogood(LearnedNogood {
                        predicates: vec![Predicate::trivially_true()],
                        backjump_level: 0,
                    }));
                }
                Ok(CutSuccess {
                    inequality: constraint,
                    skip_early_backjump,
                }) => (constraint, skip_early_backjump),
            };
            self.finish_timer(&Timer::Cutting, context);

            trace!("==> New conflicting constraint after eliminating {:?}: {new_conflicting_constraint}", trail_entry.predicate.get_domain());

            self.start_timer(Timer::CheckingOverflow);
            // Check whether the newly learned conflicting constraint overflows with the current
            // assignments
            if new_conflicting_constraint.overflows(context.assignments, trail_index) {
                return self.apply_fallback(context, FallbackReason::Overflow, used_explanations);
            }
            self.finish_timer(&Timer::CheckingOverflow, context);

            self.start_timer(Timer::CheckingConflicting);
            // If this new constraint is not false at the current height, we skip it and apply
            // resolution
            if !self.learning_options.llg_skip_conflicting_check_after_cut {
                if !new_conflicting_constraint.is_conflicting(
                    context.assignments,
                    trail_index,
                    true,
                ) {
                    return self.apply_fallback(
                        context,
                        FallbackReason::NotConflicting,
                        used_explanations,
                    );
                }
            }
            self.finish_timer(&Timer::CheckingConflicting, context);

            conflicting_constraint = new_conflicting_constraint;

            if !self.learning_options.llg_skip_early_backjump_check && skip_early_backjump {
                trace!("==> No clash in cuts, skipping early backjump check!");
                continue;
            }

            // TODO improve this to be binary search & check prev decision level first
            self.start_timer(Timer::CheckingBackjump);
            for backjump_level in 0..current_decision_level {
                // The get_trail_position_for_decision_level(backjump_level) returns the length of
                // the trail including the entire backjump level This means we will
                // be jumping to one index lower
                let backjump_trail_level = context
                    .assignments
                    .trail
                    .get_trail_position_for_decision_level(backjump_level)
                    - 1;

                // Check whether the newly learned conflicting constraint overflows with the
                // assignments at that level
                if conflicting_constraint.overflows(context.assignments, backjump_trail_level) {
                    return self.apply_fallback(
                        context,
                        FallbackReason::Overflow,
                        used_explanations,
                    );
                }

                let is_propagating = conflicting_constraint.is_propagating(
                    context.assignments,
                    backjump_trail_level,
                    true,
                );
                let is_false = conflicting_constraint.is_conflicting(
                    context.assignments,
                    backjump_trail_level,
                    true,
                );
                trace!("==> Checking decision/trail level ({backjump_level}/{backjump_trail_level}) for propagation/false: {is_propagating}/{is_false}");

                if is_propagating || is_false {
                    debug!(
                        "==> Intending to backtrack to {backjump_level}: {conflicting_constraint}"
                    );

                    context.counters.llg_statistics.llg_learned_constraints += 1;
                    context
                        .counters
                        .llg_statistics
                        .llg_learned_constraints_avg_length
                        .add_term(conflicting_constraint.lhs.0.len() as u64);
                    context
                        .counters
                        .llg_statistics
                        .llg_constraint_avg_lhs_coeff
                        .add_term(
                            conflicting_constraint
                                .lhs
                                .iter()
                                .map(|var| var.scale.abs())
                                .max()
                                .unwrap() as u64,
                        );

                    // Running resolution resolver to update activities
                    let res = self.resolution_resolver.resolve_conflict(context);
                    let Some(Nogood(learned_nogood)) = res else {
                        unreachable!("resolution should always learn something")
                    };

                    if let Some(analysis_log) = &mut context.analysis_log {
                        analysis_log.log_analysis_success(
                            learned_nogood.predicates.clone().into(),
                            LearnedNogoodStats {
                                backtrack_distance: current_decision_level
                                    - learned_nogood.backjump_level,
                                lbd: self.lbd.compute_lbd(
                                    learned_nogood.predicates.as_slice(),
                                    context.assignments,
                                ),
                            },
                            conflicting_constraint.clone(),
                            LearnedInequalityStats {
                                backtrack_distance: current_decision_level - backjump_level,
                                slack: conflicting_constraint.slack(
                                    context.assignments,
                                    trail_index,
                                    true,
                                ),
                                used_explanations,
                            },
                        );
                    }

                    self.finish_all_timers(context);
                    return if self.learning_options.llg_only_learn_nogoods {
                        Some(Nogood(learned_nogood))
                    } else {
                        Some(Constraint(LearnedConstraint {
                            constraint: conflicting_constraint,
                            backjump_level,
                        }))
                    };
                }
            }
            self.finish_timer(&Timer::CheckingBackjump, context);
        }
    }

    fn process(
        &mut self,
        context: &mut ConflictAnalysisContext,
        resolve_result: &Option<ConflictResolveResult>,
    ) -> Result<(), ()> {
        let resolve_result_unwrap = resolve_result
            .as_ref()
            .expect("Expected nogood / constraint");

        let Constraint(learned_constraint) = resolve_result_unwrap else {
            return self.resolution_resolver.process(context, resolve_result);
        };

        debug!(
            "==> Backtrack to {:?} (current = {:?})",
            learned_constraint.backjump_level,
            context.assignments.num_trail_entries() - 1
        );

        context.backtrack(learned_constraint.backjump_level);

        debug!(
            "==> Backtracked to {:?} (current = {:?})",
            context.assignments.get_decision_level(),
            context.assignments.num_trail_entries() - 1
        );

        let new_linear_prop = LinearLessOrEqualPropagator::new_learned(
            learned_constraint
                .constraint
                .lhs
                .clone()
                .0
                .into_boxed_slice(),
            learned_constraint.constraint.rhs,
            context.assignments,
        );
        let new_propagator_id = Self::create_new_propagator(
            context.propagators,
            context.assignments,
            context.watch_list_cp,
            context.propagator_queue,
            new_linear_prop,
        );

        if let Some(analysis_log) = &mut context.analysis_log {
            analysis_log.log_new_learned(LearnedItemId::Inequality(new_propagator_id))
        }

        Ok(())
    }
}

// Taken from https://docs.rs/num-integer/latest/src/num_integer/lib.rs.html#420-422
fn gcd(a: i32, b: i32) -> i32 {
    let mut m = a;
    let mut n = b;
    if m == 0 || n == 0 {
        return (m | n).abs();
    }

    // find common factors of 2
    let shift = (m | n).trailing_zeros();

    // The algorithm needs positive numbers, but the minimum value
    // can't be represented as a positive one.
    // It's also a power of two, so the gcd can be
    // calculated by bitshifting in that case

    // Assuming two's complement, the number created by the shift
    // is positive for all numbers except gcd = abs(min value)
    // The call to .abs() causes a panic in debug mode
    if m == i32::MIN || n == i32::MIN {
        let i: i32 = 1 << shift;
        return i.abs();
    }

    // guaranteed to be positive now, rest like unsigned algorithm
    m = m.abs();
    n = n.abs();

    // divide n and m by 2 until odd
    m >>= m.trailing_zeros();
    n >>= n.trailing_zeros();

    while m != n {
        if m > n {
            m -= n;
            m >>= m.trailing_zeros();
        } else {
            n -= m;
            n >>= n.trailing_zeros();
        }
    }
    m << shift
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::basic_types::linear_less_or_equal::LinearLessOrEqual;
    use crate::conjunction;
    use crate::engine::conflict_analysis::resolvers::llg_conflict_resolver::CutError::Contradiction;
    use crate::engine::conflict_analysis::resolvers::llg_conflict_resolver::CutError::NothingLearned;
    use crate::engine::conflict_analysis::resolvers::llg_conflict_resolver::CutError::Overflow;
    use crate::engine::conflict_analysis::resolvers::llg_conflict_resolver::CutSuccess;
    use crate::engine::conflict_analysis::LLGConflictResolver;
    use crate::engine::Assignments;
    use crate::variables::DomainId;
    use crate::variables::TransformableVariable;

    fn construct_test_vars() -> [DomainId; 5] {
        let a = DomainId::new(0);
        let b = DomainId::new(1);
        let c = DomainId::new(2);
        let d = DomainId::new(3);
        let e = DomainId::new(4);

        [a, b, c, d, e]
    }

    #[test]
    fn test_cut_simple() {
        let [a, b, c, d, e] = construct_test_vars();

        let x = LinearLessOrEqual::new(vec![(a, 10), (b, 2), (c, 4), (d, -5)], 12);
        let y = LinearLessOrEqual::new(vec![(a, -4), (b, -4), (c, 6), (e, 8)], -6);

        let CutSuccess {
            skip_early_backjump,
            inequality,
        } = LLGConflictResolver::apply_cut(a, &x, &y).unwrap();

        let z = LinearLessOrEqual::new(vec![(b, -8), (c, 19), (d, -5), (e, 20)], -3);

        assert_eq!(
            skip_early_backjump, false,
            "should be backjumping early due to clash"
        );
        assert_eq!(inequality, z);
    }

    #[test]
    fn test_cut_same_coeff() {
        let [a, b, c, d, e] = construct_test_vars();

        let x = LinearLessOrEqual::new(vec![(a, -4), (b, 2), (c, 4), (d, -5)], 10);
        let y = LinearLessOrEqual::new(vec![(a, 4), (b, -4), (c, 6), (e, 8)], -5);

        let CutSuccess {
            skip_early_backjump,
            inequality,
        } = LLGConflictResolver::apply_cut(a, &x, &y).unwrap();

        let z = LinearLessOrEqual::new(vec![(b, -2), (c, 10), (d, -5), (e, 8)], 5);

        assert_eq!(
            skip_early_backjump, false,
            "should be backjumping early due to clash"
        );
        assert_eq!(inequality, z);
    }

    #[test]
    fn test_cut_no_clash() {
        let [a, b, c, d, e] = construct_test_vars();

        let x = LinearLessOrEqual::new(vec![(a, -4), (b, 2), (c, 4), (d, -5)], 10);
        let y = LinearLessOrEqual::new(vec![(a, 4), (e, 8)], -5);

        let CutSuccess {
            skip_early_backjump,
            inequality,
        } = LLGConflictResolver::apply_cut(a, &x, &y).unwrap();

        let z = LinearLessOrEqual::new(vec![(b, 2), (c, 4), (d, -5), (e, 8)], 5);

        assert_eq!(
            skip_early_backjump, true,
            "should not be backjumping early due to lack of clash"
        );
        assert_eq!(inequality, z);
    }

    #[test]
    fn test_cut_fully_clash() {
        let [a, b, c, d, e] = construct_test_vars();

        let x = LinearLessOrEqual::new(vec![(a, -10), (b, 2), (c, 4), (d, -5), (e, -1)], 10);
        let y = LinearLessOrEqual::new(vec![(a, 4), (b, -4), (c, 6), (d, 5), (e, 8)], -5);

        let CutSuccess {
            skip_early_backjump,
            inequality,
        } = LLGConflictResolver::apply_cut(a, &x, &y).unwrap();

        let z = LinearLessOrEqual::new(vec![(b, -16), (c, 38), (d, 15), (e, 38)], -5);

        assert_eq!(
            skip_early_backjump, false,
            "should be backjumping early due to clash"
        );
        assert_eq!(inequality, z);
    }

    #[test]
    fn test_cut_contradiction() {
        let [a, b, c, d, e] = construct_test_vars();

        let x = LinearLessOrEqual::new(vec![(a, -2), (b, 2), (c, -3), (d, -5), (e, -4)], 0);
        let y = LinearLessOrEqual::new(vec![(a, 4), (b, -4), (c, 6), (d, 10), (e, 8)], -5);

        let cut_result = LLGConflictResolver::apply_cut(a, &x, &y).unwrap_err();

        assert!(matches!(cut_result, Contradiction {}));
    }

    #[test]
    fn test_cut_nothing_learned() {
        let [a, b, c, d, e] = construct_test_vars();

        let x = LinearLessOrEqual::new(vec![(a, -2), (b, 2), (c, -3), (d, -5), (e, -4)], 10);
        let y = LinearLessOrEqual::new(vec![(a, 4), (b, -4), (c, 6), (d, 10), (e, 8)], -5);

        let cut_result = LLGConflictResolver::apply_cut(a, &x, &y).unwrap_err();

        assert!(matches!(cut_result, NothingLearned {}));
    }

    #[test]
    fn test_cut_overflow() {
        let [a, b, c, d, e] = construct_test_vars();

        let x = LinearLessOrEqual::new(
            vec![(a, -99997), (b, 223545223), (c, -3), (d, -5), (e, -4)],
            10,
        );
        let y = LinearLessOrEqual::new(vec![(a, 99995), (b, -1000), (c, 6), (d, 10), (e, 8)], -5);

        let cut_result = LLGConflictResolver::apply_cut(a, &x, &y).unwrap_err();

        assert!(matches!(cut_result, Overflow {}));
    }

    #[test]
    fn test_transform_full_clause() {
        let mut assignment = Assignments::default();
        let b1 = assignment.grow(0, 1);
        let b2 = assignment.grow(0, 1);
        let b3 = assignment.grow(0, 1);
        let b4 = assignment.grow(0, 1);

        let clause = conjunction!([b1 >= 1] & [b2 >= 1] & [b3 <= 0] & [b4 <= 0]);
        let constraint = LLGConflictResolver::transform_to_constraint(
            clause.into_iter().collect_vec(),
            &mut assignment,
        );

        assert_eq!(
            constraint.unwrap(),
            LinearLessOrEqual::new(
                vec![b1.scaled(-1), b2.scaled(-1), b3.scaled(1), b4.scaled(1)],
                1
            )
        )
    }

    #[test]
    fn test_transform_lower_bound_clause() {
        let mut assignment = Assignments::default();
        let b1 = assignment.grow(0, 1);
        let b2 = assignment.grow(0, 1);
        let b3 = assignment.grow(0, 1);
        let v1 = assignment.grow(-5, 5);

        let clause = conjunction!([b1 >= 1] & [b2 >= 1] & [b3 <= 0] & [v1 >= 2]);
        let constraint = LLGConflictResolver::transform_to_constraint(
            clause.into_iter().collect_vec(),
            &mut assignment,
        );

        assert_eq!(
            constraint.unwrap(),
            LinearLessOrEqual::new(
                vec![b1.scaled(-7), b2.scaled(-7), b3.scaled(7), v1.scaled(-1)],
                5
            )
        )
    }

    #[test]
    fn test_transform_upper_bound_clause() {
        let mut assignment = Assignments::default();
        let b1 = assignment.grow(0, 1);
        let b2 = assignment.grow(0, 1);
        let b3 = assignment.grow(0, 1);
        let v1 = assignment.grow(-5, 5);

        let clause = conjunction!([b1 >= 1] & [b2 >= 1] & [b3 <= 0] & [v1 <= -2]);
        let constraint = LLGConflictResolver::transform_to_constraint(
            clause.into_iter().collect_vec(),
            &mut assignment,
        );

        assert_eq!(
            constraint.unwrap(),
            LinearLessOrEqual::new(
                vec![b1.scaled(-7), b2.scaled(-7), b3.scaled(7), v1.scaled(1)],
                5
            )
        )
    }
}
