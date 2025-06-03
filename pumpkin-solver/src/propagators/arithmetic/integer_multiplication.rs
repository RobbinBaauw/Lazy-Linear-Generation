use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;

use crate::basic_types::linear_less_or_equal::FilterNonZero;
use crate::basic_types::linear_less_or_equal::LinearLessOrEqual;
use crate::basic_types::Inconsistency;
use crate::basic_types::PropagationReason;
use crate::basic_types::PropagationStatusCP;
use crate::conjunction;
use crate::engine::cp::propagation::ReadDomains;
use crate::engine::domain_events::DomainEvents;
use crate::engine::opaque_domain_event::OpaqueDomainEvent;
use crate::engine::propagation::EnqueueDecision;
use crate::engine::propagation::LocalId;
use crate::engine::propagation::PropagationContext;
use crate::engine::propagation::PropagationContextMut;
use crate::engine::propagation::Propagator;
use crate::engine::propagation::PropagatorId;
use crate::engine::propagation::PropagatorInitialisationContext;
use crate::engine::variables::IntegerVariable;
use crate::engine::Assignments;
use crate::new_explanation;
use crate::predicates::Predicate;
use crate::pumpkin_assert_simple;
use crate::variables::AffineView;
use crate::variables::DomainId;
use crate::variables::TransformableVariable;

/// A propagator for maintaining the constraint `a * b = c`. The propagator
/// (currently) only propagates the signs of the variables, the case where a, b, c >= 0, and detects
/// a conflict if the variables are fixed.
pub(crate) struct IntegerMultiplicationPropagator<VA, VB, VC> {
    pub(crate) a: VA,
    pub(crate) b: VB,
    pub(crate) c: VC,
    last_propagated_var: u8,
}

const ID_A: LocalId = LocalId::from(0);
const ID_B: LocalId = LocalId::from(1);
const ID_C: LocalId = LocalId::from(2);

impl<VA, VB, VC> IntegerMultiplicationPropagator<VA, VB, VC>
where
    VA: IntegerVariable + 'static,
    VB: IntegerVariable + 'static,
    VC: IntegerVariable + 'static,
{
    fn is_binary(&self, var: AffineView<DomainId>, assignments: &Assignments) -> bool {
        var.lower_bound_initial(assignments) == 0 && var.upper_bound_initial(assignments) == 1
    }

    fn explain_binary_c_0(
        &self,
        binary_var: AffineView<DomainId>,
        assignments: &Assignments,
    ) -> LinearLessOrEqual {
        // Assuming binary_var = a

        let big_m = self.c.upper_bound_initial(assignments).max(0);

        // c <= 0 + Ma
        // c - Ma <= 0
        LinearLessOrEqual::new_expl(
            vec![self.c.flatten(), binary_var.scaled(big_m)].non_zero_scale(),
            big_m,
            519,
        )
    }

    fn explain_binary_c_ub(
        &self,
        binary_var: AffineView<DomainId>,
        other_var: AffineView<DomainId>,
        assignments: &Assignments,
    ) -> LinearLessOrEqual {
        // Assuming binary_var = a, other_var = b

        // c <= b if a == 1 (ub c)
        assert_eq!(binary_var.lower_bound(assignments), 1);
        assert_eq!(binary_var.lower_bound_initial(assignments), 0);
        assert_eq!(binary_var.upper_bound_initial(assignments), 1);

        let big_m = (other_var.scaled(-1).upper_bound_initial(assignments)
            + self.c.upper_bound_initial(assignments))
        .max(0);

        // c <= b + M(1-a)
        // -b + c + Ma <= M
        LinearLessOrEqual::new_expl(
            vec![
                other_var.flatten().scaled(-1),
                self.c.flatten(),
                binary_var.flatten().scaled(big_m),
            ]
            .non_zero_scale(),
            big_m,
            524,
        )
    }

    fn explain_binary_c_lb(
        &self,
        binary_var: AffineView<DomainId>,
        other_var: AffineView<DomainId>,
        assignments: &Assignments,
    ) -> LinearLessOrEqual {
        // Assuming binary_var = a, other_var = b

        // b <= c if a == 1 (lb c)
        assert_eq!(binary_var.lower_bound(assignments), 1);
        assert_eq!(binary_var.lower_bound_initial(assignments), 0);
        assert_eq!(binary_var.upper_bound_initial(assignments), 1);

        let big_m = (other_var.upper_bound_initial(assignments)
            + self.c.scaled(-1).upper_bound_initial(assignments))
        .max(0);

        // b <= c + M(1-a)
        // b - c + Ma <= M
        LinearLessOrEqual::new_expl(
            vec![
                other_var.flatten().scaled(1),
                self.c.flatten().scaled(-1),
                binary_var.flatten().scaled(big_m),
            ]
            .non_zero_scale(),
            big_m,
            522,
        )
    }

    fn explain_lb_a(
        &self,
        _bound: i32,
        assignments: &mut Assignments,
        _propagator_id: PropagatorId,
    ) -> LinearLessOrEqual {
        // NOTE: This method will serve as documentation for all related methods.
        //
        // We can observe four inequalities that will allow us to explain all propagations:
        // * a_min * b <= c
        // * a_max * b >= c
        // * a * b_min <= c
        // * a * b_max >= c
        //
        // For the a_{min, max} and b_{min, max}, we can either use the initial upper bound,
        // or create a new auxiliary variable. This can be tested, but for now we will use
        // a new auxiliary variable for this purpose.
        //
        // The preconditions for these inequalities are important. Going over each of the 6
        // possible propagations:
        // * lb(a): a * b_max >= c if b_min >= 0 and b_max >= 1 and c >= 0.
        //
        // * lb(b): a_max * b >= c if a_min >= 0 and a_max >= 1 and c >= 0.
        //
        // * lb(c): a_min * b <= c if a_min >= 0 and b >= 0. Equivalent for a * b_min <= c. Choose
        //   explanation based on which initial domain of a & b is smallest. Or... choose
        //   explanation based on which diverts from original bounds most?
        //
        // * ub(a): a * b_min <= c if b_min >= 1 and c >= 0.
        //
        // * ub(b): a_min * b <= c if a_min >= 1 and c >= 0.
        //
        // * ub(c): a_max * b >= c if a_min >= 0 and a_max >= 1 and b >= 0. Equivalent for a * b_max
        //   >= c. Choose explanation based on which initial domain of a & b is smallest. Or...
        //   choose explanation based on which diverts from original bounds most?
        //
        // The conditions can be expressed using big M, in different ways. Take lb(a):
        // * b_min >= 0 and c >= 0 can be expressed using the sign auxiliaries
        // * b_max >= 1 is checked before, but we need to add the condition that b <= b_max, which
        //   is done using an additional auxiliary. In order to not make the number of auxiliaries
        //   grow too much, we express b >= b_min as !(b <= b_min-1)
        //
        // About the big M. Take lb(a) as an example. If M <= 0, we know that with the given b_max,
        // a * b_max >= c always holds. Therefore, if M <= 0, we can ignore all conditions! It
        // doesn't matter that b_max might be too low, and that with a different b_max, the
        // condition is invalid. Because for these specific numbers, this condition is implied
        // from the model.
        //
        // This still differs from Minizinc, which only uses one auxiliary variable per explanation.
        // They fix the value of the smallest domain, and for each possible value introduce an
        // explanation. This way, the only precondition required is that the auxiliary is true.
        // You can then express using for instance a * val >= c and a * val <= c given that
        // b == val. It's a trade-off between number of auxiliary variables, and the strength
        // of a single variable. For now, I will go for the option as described above, as during
        // propagation, we can be sure that this condition will be applicable because these
        // preconditions are already checked before propagating anyway.

        // lb(a): a * b_max >= c if b_min >= 0 and b_max >= 1 and c >= 0.
        let b_max = self.b.upper_bound(assignments);

        if self.is_binary(self.b.flatten(), assignments) && self.b.lower_bound(assignments) == 1 {
            return self.explain_binary_c_ub(self.b.flatten(), self.a.flatten(), assignments);
        };

        let a_bmax_ub_init = self.a.scaled(-b_max).upper_bound_initial(assignments);
        let c_ub_init = self.c.upper_bound_initial(assignments);
        let big_m = (a_bmax_ub_init + c_ub_init).max(0);

        // a * b_max >= c - M(1-is_b_max) - Mnb - Mnc
        // a * b_max - c - Mis_b_max + Mnb + Mnc >= -M
        // a * -b_max + c + Mis_b_max - Mnb - Mnc <= M
        LinearLessOrEqual::new_expl(
            vec![
                self.a.flatten().scaled(-b_max),
                self.c.flatten().scaled(1),
                self.b.max_aux(assignments, b_max).scaled(big_m),
                self.b.neg_aux(assignments).scaled(-big_m),
                self.c.neg_aux(assignments).scaled(-big_m),
            ]
            .non_zero_scale(),
            big_m,
            500,
        )
    }

    fn explain_lb_b(
        &self,
        _bound: i32,
        assignments: &mut Assignments,
        _propagator_id: PropagatorId,
    ) -> LinearLessOrEqual {
        // lb(b): a_max * b >= c if a_min >= 0 and a_max >= 1 and c >= 0.
        let a_max = self.a.upper_bound(assignments);

        if self.is_binary(self.a.flatten(), assignments) && self.a.lower_bound(assignments) == 1 {
            return self.explain_binary_c_ub(self.a.flatten(), self.b.flatten(), assignments);
        };

        let b_amax_ub_init = self.b.scaled(-a_max).upper_bound_initial(assignments);
        let c_ub_init = self.c.upper_bound_initial(assignments);
        let big_m = (b_amax_ub_init + c_ub_init).max(0);

        // a_max * b >= c - M(1-is_a_max) - Mna - Mnc
        // a_max * b - c - Mis_a_max + Mna + Mnc >= -M
        // -a_max * b + c + Mis_a_max - Mna - Mnc <= M
        LinearLessOrEqual::new_expl(
            vec![
                self.b.flatten().scaled(-a_max),
                self.c.flatten().scaled(1),
                self.a.max_aux(assignments, a_max).scaled(big_m),
                self.a.neg_aux(assignments).scaled(-big_m),
                self.c.neg_aux(assignments).scaled(-big_m),
            ]
            .non_zero_scale(),
            big_m,
            501,
        )
    }

    fn explain_lb_c(
        &self,
        _bound: i32,
        assignments: &mut Assignments,
        _propagator_id: PropagatorId,
        last_propagated_var: u8,
    ) -> LinearLessOrEqual {
        // lb(c): a_min * b <= c if a_min >= 0 and b >= 0. Equivalent for a * b_min <= c.
        // In this case, we pick a or b based on which difference between the current and initial
        // bounds is largest.

        let mut create_explanation =
            |trigger_var: AffineView<DomainId>, other_var: AffineView<DomainId>| {
                // In these comments, I assume min_var = a and var = b

                if self.is_binary(trigger_var, assignments)
                    && trigger_var.lower_bound(assignments) == 1
                {
                    // If the var that caused this propagation is binary, this can cause both
                    // other_var <= c
                    return self.explain_binary_c_lb(trigger_var, other_var, assignments);
                } else if self.is_binary(other_var, assignments)
                    && other_var.lower_bound(assignments) == 1
                {
                    // If the var that caused this propagation is not binary, and the other side
                    // is, only trigger_var <= c is possible
                    return self.explain_binary_c_lb(other_var, trigger_var, assignments);
                };

                let min_var_lb = other_var.lower_bound(assignments);

                let x_min_ub_init = trigger_var
                    .scaled(min_var_lb)
                    .upper_bound_initial(assignments);
                let c_neg_ub_init = self.c.scaled(-1).upper_bound_initial(assignments);
                let big_m = (x_min_ub_init + c_neg_ub_init).max(0);

                let var_neg = trigger_var.neg_aux(assignments);

                // Assume we chose to use a_min:
                // a_min * b <= c + M(1-is_min) + Mnb
                // a_min * b - c + Mis_min - Mnb <= M
                LinearLessOrEqual::new_expl(
                    vec![
                        trigger_var.scaled(min_var_lb),
                        self.c.flatten().scaled(-1),
                        other_var.min_aux(assignments, min_var_lb).scaled(big_m),
                        var_neg.scaled(-big_m),
                    ]
                    .non_zero_scale(),
                    big_m,
                    502,
                )
            };

        if last_propagated_var & 1 == 1 {
            // A propagated last, so use curr lb of b
            create_explanation(self.a.flatten(), self.b.flatten())
        } else if last_propagated_var & 2 == 2 {
            // B propagated last, so use curr lb of a
            create_explanation(self.b.flatten(), self.a.flatten())
        } else {
            // C was propagated by another propagator. Pick arbitrary explanation...
            create_explanation(self.a.flatten(), self.b.flatten())
        }
    }

    fn explain_ub_a(
        &self,
        _bound: i32,
        assignments: &mut Assignments,
        _propagator_id: PropagatorId,
    ) -> LinearLessOrEqual {
        // ub(a): a * b_min <= c if b_min >= 1 and c >= 0.
        let b_min = self.b.lower_bound(assignments);
        //
        if self.is_binary(self.b.flatten(), assignments) && self.b.lower_bound(assignments) == 1 {
            return self.explain_binary_c_lb(self.b.flatten(), self.a.flatten(), assignments);
        };

        let a_bmin_ub_init = self.a.scaled(b_min).upper_bound_initial(assignments);
        let c_neg_ub_init = self.c.scaled(-1).upper_bound_initial(assignments);
        let big_m = (a_bmin_ub_init + c_neg_ub_init).max(0);

        // a * b_min <= c + M(1-is_b_min) + Mnc
        // a * b_min - c + Mis_b_min - Mnc <= M
        LinearLessOrEqual::new_expl(
            vec![
                self.a.flatten().scaled(b_min),
                self.c.flatten().scaled(-1),
                self.b.min_aux(assignments, b_min).scaled(big_m),
                self.c.neg_aux(assignments).scaled(-big_m),
            ]
            .non_zero_scale(),
            big_m,
            503,
        )
    }

    fn explain_ub_b(
        &self,
        _bound: i32,
        assignments: &mut Assignments,
        _propagator_id: PropagatorId,
    ) -> LinearLessOrEqual {
        // ub(b): a_min * b <= c if a_min >= 1 and c >= 0.
        let a_min = self.a.lower_bound(assignments);

        if self.is_binary(self.a.flatten(), assignments) && self.a.lower_bound(assignments) == 1 {
            return self.explain_binary_c_lb(self.a.flatten(), self.b.flatten(), assignments);
        };

        let b_amin_ub_init = self.b.scaled(a_min).upper_bound_initial(assignments);
        let c_neg_ub_init = self.c.scaled(-1).upper_bound_initial(assignments);
        let big_m = (b_amin_ub_init + c_neg_ub_init).max(0);

        // a_min * b <= c + M(1-is_a_min) + Mnc
        // a_min * b - c + Mis_a_min - Mnc <= M
        LinearLessOrEqual::new_expl(
            vec![
                self.b.flatten().scaled(a_min),
                self.c.flatten().scaled(-1),
                self.a.min_aux(assignments, a_min).scaled(big_m),
                self.c.neg_aux(assignments).scaled(-big_m),
            ]
            .non_zero_scale(),
            big_m,
            504,
        )
    }

    fn explain_ub_c(
        &self,
        _bound: i32,
        assignments: &mut Assignments,
        _propagator_id: PropagatorId,
        last_propagated_var: u8,
    ) -> LinearLessOrEqual {
        // ub(c): a_max * b >= c if a_min >= 0 and b >= 0. Equivalent for a * b_max >= c.

        let mut create_explanation =
            |trigger_var: AffineView<DomainId>, other_var: AffineView<DomainId>| {
                // In these comments, I assume other_var = a and trigger_var = b

                if self.is_binary(trigger_var, assignments) {
                    // If the var that caused this propagation is binary, this can cause both
                    // c <= 0 or c <= other_var
                    if trigger_var.upper_bound(assignments) == 0 {
                        return self.explain_binary_c_0(other_var, assignments);
                    } else if trigger_var.lower_bound(assignments) == 1 {
                        return self.explain_binary_c_ub(trigger_var, other_var, assignments);
                    }
                } else if self.is_binary(other_var, assignments)
                    && other_var.lower_bound(assignments) == 1
                {
                    // If the var that caused this propagation is not binary, and the other side
                    // is, only c <= trigger_var is possible
                    return self.explain_binary_c_ub(other_var, trigger_var, assignments);
                };

                // Get current upper bound
                let other_var_ub = other_var.upper_bound(assignments);

                let x_max_ub_init = trigger_var
                    .scaled(-other_var_ub)
                    .upper_bound_initial(assignments);
                let c_ub_init = self.c.upper_bound_initial(assignments);
                let big_m = (x_max_ub_init + c_ub_init).max(0);

                // Assume we chose to use a_min:
                // a_max * b >= c - M(1-is_max) - Mna - Mnb
                // a_max * b - c - Mis_max + Mna + Mnb >= -M
                // -a_max * b + c + Mis_max - Mna - Mnb <= M
                LinearLessOrEqual::new_expl(
                    vec![
                        trigger_var.scaled(-other_var_ub),
                        self.c.flatten(),
                        other_var.max_aux(assignments, other_var_ub).scaled(big_m),
                        self.a.neg_aux(assignments).scaled(-big_m),
                        self.b.neg_aux(assignments).scaled(-big_m),
                    ]
                    .non_zero_scale(),
                    big_m,
                    505,
                )
            };

        if last_propagated_var & 1 == 1 {
            // A propagated last, so use curr ub of b
            create_explanation(self.a.flatten(), self.b.flatten())
        } else if last_propagated_var & 2 == 2 {
            // B propagated last, so use curr ub of a
            create_explanation(self.b.flatten(), self.a.flatten())
        } else {
            // C was propagated by another propagator. Pick arbitrary explanation...
            create_explanation(self.a.flatten(), self.b.flatten())
        }
    }
}

impl<VA, VB, VC> IntegerMultiplicationPropagator<VA, VB, VC>
where
    VA: IntegerVariable + 'static,
    VB: IntegerVariable + 'static,
    VC: IntegerVariable + 'static,
{
    pub(crate) fn new(a: VA, b: VB, c: VC) -> Self {
        IntegerMultiplicationPropagator {
            a,
            b,
            c,
            last_propagated_var: 0,
        }
    }

    fn perform_propagation(&self, context: &mut PropagationContextMut) -> PropagationStatusCP {
        let a = &self.a;
        let b = &self.b;
        let c = &self.c;

        // First we propagate the signs
        let prior_propagation = propagate_signs(context, a, b, c)?;

        let a_min = context.lower_bound(a);
        let a_max = context.upper_bound(a);
        let b_min = context.lower_bound(b);
        let b_max = context.upper_bound(b);
        let c_min = context.lower_bound(c);
        let c_max = context.upper_bound(c);

        if a_min >= 0 && b_min >= 0 {
            let new_max_c = a_max * b_max;
            let new_min_c = a_min * b_min;

            // c is smaller than the maximum value that a * b can take
            //
            // We need the lower-bounds in the explanation as well because the reasoning does not
            // hold in the case of a negative lower-bound
            let ub_c_explanation = if new_max_c < c_max {
                new_explanation!(self.explain_ub_c(
                    new_max_c,
                    context.assignments,
                    context.propagator_id,
                    prior_propagation.unwrap_or(self.last_propagated_var)
                ))
            } else {
                None
            };

            context.set_upper_bound(
                c,
                new_max_c,
                (
                    conjunction!([a >= 0] & [a <= a_max] & [b >= 0] & [b <= b_max]),
                    ub_c_explanation,
                ),
            )?;

            // c is larger than the minimum value that a * b can take
            let lb_c_explanation = if new_min_c > c_min {
                new_explanation!(self.explain_lb_c(
                    new_min_c,
                    context.assignments,
                    context.propagator_id,
                    prior_propagation.unwrap_or(self.last_propagated_var)
                ))
            } else {
                None
            };
            context.set_lower_bound(
                c,
                new_min_c,
                (conjunction!([a >= a_min] & [b >= b_min]), lb_c_explanation),
            )?;
        }

        if b_min >= 0 && b_max >= 1 && c_min >= 1 {
            // a >= ceil(c.min / b.max)
            let bound = div_ceil_pos(c_min, b_max);
            let lb_a_explanation = if bound > a_min {
                new_explanation!(self.explain_lb_a(
                    bound,
                    context.assignments,
                    context.propagator_id
                ))
            } else {
                None
            };
            context.set_lower_bound(
                a,
                bound,
                (
                    conjunction!([c >= c_min] & [b >= 0] & [b <= b_max]),
                    lb_a_explanation,
                ),
            )?;
        }

        if b_min >= 1 && c_min >= 0 {
            // a <= floor(c.max / b.min)
            let bound = c_max / b_min;
            let ub_a_explanation = if bound < a_max {
                new_explanation!(self.explain_ub_a(
                    bound,
                    context.assignments,
                    context.propagator_id
                ))
            } else {
                None
            };
            context.set_upper_bound(
                a,
                bound,
                (
                    conjunction!([c >= 0] & [c <= c_max] & [b >= b_min]),
                    ub_a_explanation,
                ),
            )?;
        }

        if a_min >= 1 && c_min >= 0 {
            // b <= floor(c.max / a.min)
            let bound = c_max / a_min;
            let ub_b_explanation = if bound < b_max {
                new_explanation!(self.explain_ub_b(
                    bound,
                    context.assignments,
                    context.propagator_id
                ))
            } else {
                None
            };
            context.set_upper_bound(
                b,
                bound,
                (
                    conjunction!([c >= 0] & [c <= c_max] & [a >= a_min]),
                    ub_b_explanation,
                ),
            )?;
        }

        // b >= ceil(c.min / a.max)
        if a_min >= 0 && a_max >= 1 && c_min >= 1 {
            let bound = div_ceil_pos(c_min, a_max);

            let lb_b_explanation = if bound > b_min {
                new_explanation!(self.explain_lb_b(
                    bound,
                    context.assignments,
                    context.propagator_id
                ))
            } else {
                None
            };
            context.set_lower_bound(
                b,
                bound,
                (
                    conjunction!([c >= c_min] & [a >= 0] & [a <= a_max]),
                    lb_b_explanation,
                ),
            )?;
        }

        if context.is_fixed(a)
            && context.is_fixed(b)
            && context.is_fixed(c)
            && (context.lower_bound(a) * context.lower_bound(b)) != context.lower_bound(c)
        {
            // All variables are assigned but the resulting value is not correct, so we report a
            // conflict

            // Error can be explained by any of the explanations we have seen before.
            // We can find the last trail entry that propagated a, b or c and check the direction
            // in which it was propagated.
            let last_propagated_var = context
                .assignments
                .trail
                .iter()
                .rev()
                .find(|entry| {
                    let pred_domain = entry.predicate.get_domain();
                    pred_domain == a.domain_id()
                        || pred_domain == b.domain_id()
                        || pred_domain == c.domain_id()
                })
                .cloned();

            let linear_expl = new_explanation!(last_propagated_var.map(|var| {
                let lb_increased = match var.predicate {
                    Predicate::LowerBound { .. } => true,
                    Predicate::UpperBound { .. } => false,
                    Predicate::NotEqual {
                        not_equal_constant, ..
                    } => not_equal_constant == var.old_lower_bound,
                    Predicate::Equal {
                        equality_constant, ..
                    } => equality_constant > var.old_lower_bound,
                };

                // If the lower bound of a variable causing the conflict was recently increased,
                // we return a conflicting constraint in which the variable is positive. That way,
                // the sign is opposite of the explanation of this previous propagation.
                let pred_rhs = var.predicate.get_right_hand_side();
                match var.predicate.get_domain() {
                    d if d == a.domain_id() => {
                        if lb_increased {
                            self.explain_ub_a(pred_rhs, context.assignments, context.propagator_id)
                        } else {
                            self.explain_lb_a(pred_rhs, context.assignments, context.propagator_id)
                        }
                    }
                    d if d == b.domain_id() => {
                        if lb_increased {
                            self.explain_ub_b(pred_rhs, context.assignments, context.propagator_id)
                        } else {
                            self.explain_lb_b(pred_rhs, context.assignments, context.propagator_id)
                        }
                    }
                    d if d == c.domain_id() => {
                        if lb_increased {
                            self.explain_ub_c(
                                pred_rhs,
                                context.assignments,
                                context.propagator_id,
                                prior_propagation.unwrap_or(self.last_propagated_var),
                            )
                        } else {
                            self.explain_lb_c(
                                pred_rhs,
                                context.assignments,
                                context.propagator_id,
                                prior_propagation.unwrap_or(self.last_propagated_var),
                            )
                        }
                    }
                    _ => unreachable!("Unexpected domain"),
                }
            }));

            return Err(PropagationReason(
                conjunction!(
                    [a == context.lower_bound(a)]
                        & [b == context.lower_bound(b)]
                        & [c == context.lower_bound(c)]
                ),
                linear_expl.flatten(),
            )
            .into());
        }

        Ok(())
    }
}

impl<VA: 'static, VB: 'static, VC: 'static> Propagator
    for IntegerMultiplicationPropagator<VA, VB, VC>
where
    VA: IntegerVariable,
    VB: IntegerVariable,
    VC: IntegerVariable,
{
    fn initialise_at_root(
        &mut self,
        context: &mut PropagatorInitialisationContext,
    ) -> Result<(), PropagationReason> {
        let _ = context.register(self.a.clone(), DomainEvents::ANY_INT, ID_A);
        let _ = context.register(self.b.clone(), DomainEvents::ANY_INT, ID_B);
        let _ = context.register(self.c.clone(), DomainEvents::ANY_INT, ID_C);

        Ok(())
    }

    fn priority(&self) -> u32 {
        1
    }

    fn name(&self) -> &str {
        "IntTimes"
    }

    fn notify(
        &mut self,
        _context: PropagationContext,
        local_id: LocalId,
        _event: OpaqueDomainEvent,
    ) -> EnqueueDecision {
        self.last_propagated_var = match local_id {
            ID_A => 1,
            ID_B => 2,
            ID_C => self.last_propagated_var,
            _ => unreachable!(),
        };

        EnqueueDecision::Enqueue
    }

    fn propagate(&mut self, context: &mut PropagationContextMut) -> PropagationStatusCP {
        let status = self.perform_propagation(context);
        self.last_propagated_var = 0;
        status
    }

    fn debug_propagate_from_scratch(
        &self,
        context: &mut PropagationContextMut,
    ) -> PropagationStatusCP {
        self.perform_propagation(context)
    }
}

/// Propagates the signs of the variables, it performs the following propagations:
/// - Propagating based on positive bounds
///     - If a is positive and b is positive then c is positive
///     - If a is positive and c is positive then b is positive
///     - If b is positive and c is positive then a is positive
/// - Propagating based on negative bounds
///     - If a is negative and b is negative then c is positive
///     - If a is negative and c is negative then b is positive
///     - If b is negative and c is negative then b is positive
/// - Propagating based on mixed bounds
///     - Propagating c based on a and b
///         - If a is negative and b is positive then c is negative
///         - If a is positive and b is negative then c is negative
///     - Propagating b based on a and c
///         - If a is negative and c is positive then b is negative
///         - If a is positive and c is negative then b is negative
///     - Propagating a based on b and c
///         - If b is negative and c is positive then a is negative
///         - If b is positive and c is negative then a is negative
///
/// Note that this method does not propagate a value if 0 is in the domain as, for example, 0 * -3 =
/// 0 and 0 * 3 = 0 are both equally valid.
fn propagate_signs<VA: IntegerVariable, VB: IntegerVariable, VC: IntegerVariable>(
    context: &mut PropagationContextMut,
    a: &VA,
    b: &VB,
    c: &VC,
) -> Result<Option<u8>, Inconsistency> {
    let mut prior_propagation: Option<u8> = None;

    let a_min = context.lower_bound(a);
    let a_max = context.upper_bound(a);
    let b_min = context.lower_bound(b);
    let b_max = context.upper_bound(b);
    let c_min = context.lower_bound(c);
    let c_max = context.upper_bound(c);

    let a_pos_ub_init = a.scaled(1).upper_bound_initial(context.assignments);
    let a_neg_ub_init = a.scaled(-1).upper_bound_initial(context.assignments);
    let b_pos_ub_init = b.scaled(1).upper_bound_initial(context.assignments);
    let b_neg_ub_init = b.scaled(-1).upper_bound_initial(context.assignments);
    let c_pos_ub_init = c.scaled(1).upper_bound_initial(context.assignments);
    let c_neg_ub_init = c.scaled(-1).upper_bound_initial(context.assignments);

    // Propagating based on positive bounds
    // a is positive and b is positive -> c is positive
    if a_min >= 0 && b_min >= 0 {
        // Representing c >= 0 if a >= 0 && b >= 0
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m = c_neg_ub_init.max(0);

        // c >= 0 - Mna - Mnb, or
        // c + Mna + Mnb >= 0, or
        // -c - Mna - Mnb <= 0
        let lb_c_explanation = new_explanation!(LinearLessOrEqual::new_expl(
            vec![
                c.flatten().scaled(-1),
                a.neg_aux(context.assignments).scaled(-big_m),
                b.neg_aux(context.assignments).scaled(-big_m),
            ]
            .non_zero_scale(),
            0,
            506,
        ));

        context.set_lower_bound(c, 0, (conjunction!([a >= 0] & [b >= 0]), lb_c_explanation))?;
    }

    // a is positive and c is positive -> b is positive
    if a_min >= 1 && c_min >= 1 {
        // Representing b >= 1 if a >= 1 && c >= 1
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m = (b_neg_ub_init + 1).max(0);

        // b >= 1 - M(1-pa) - M(1-pc), or
        // b >= 1 - M + Mpa - M + Mpc, or
        // b - Mpa - Mpc >= 1 - 2M, or
        // -b + Mpa + Mpc <= -1 + 2M
        let lb_b_explanation = new_explanation!(LinearLessOrEqual::new_expl(
            vec![
                b.flatten().scaled(-1),
                a.pos_aux(context.assignments).scaled(big_m),
                c.pos_aux(context.assignments).scaled(big_m),
            ]
            .non_zero_scale(),
            -1 + 2 * big_m,
            507,
        ));

        if 1 > b_min {
            prior_propagation = Some(2);
        }

        context.set_lower_bound(b, 1, (conjunction!([a >= 1] & [c >= 1]), lb_b_explanation))?;
    }

    // b is positive and c is positive -> a is positive
    if b_min >= 1 && c_min >= 1 {
        // Representing a >= 1 if b >= 1 && c >= 1
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m = (a_neg_ub_init + 1).max(0);

        // a >= 1 - M(1-pb) - M(1-pc), or
        // a >= 1 - M + Mpb - M + Mpc, or
        // a - Mpb - Mpc >= 1 - 2M, or
        // -a + Mpb + Mpc <= -1 + 2M
        let lb_a_explanation = new_explanation!(LinearLessOrEqual::new_expl(
            vec![
                a.flatten().scaled(-1),
                b.pos_aux(context.assignments).scaled(big_m),
                c.pos_aux(context.assignments).scaled(big_m),
            ]
            .non_zero_scale(),
            -1 + 2 * big_m,
            508,
        ));

        if 1 > a_min {
            prior_propagation = Some(1);
        }

        context.set_lower_bound(a, 1, (conjunction!([b >= 1] & [c >= 1]), lb_a_explanation))?;
    }

    // Propagating based on negative bounds
    // a is negative and b is negative -> c is positive
    if a_max <= 0 && b_max <= 0 {
        // Representing c >= 0 if a <= 0 && b <= 0
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m = c_neg_ub_init.max(0);

        // c >= 0 - Mpa - Mpb, or
        // c + Mpa + Mpb >= 0, or
        // -c - Mpa - Mpb <= 0
        let lb_c_explanation = new_explanation!(LinearLessOrEqual::new_expl(
            vec![
                c.flatten().scaled(-1),
                a.pos_aux(context.assignments).scaled(-big_m),
                b.pos_aux(context.assignments).scaled(-big_m),
            ]
            .non_zero_scale(),
            0,
            509,
        ));

        context.set_lower_bound(c, 0, (conjunction!([a <= 0] & [b <= 0]), lb_c_explanation))?;
    }

    // a is negative and c is negative -> b is positive
    if a_max <= -1 && c_max <= -1 {
        // Representing b >= 1 if a <= -1 && c <= 1
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m = (b_neg_ub_init + 1).max(0);

        // b >= 1 - M(1-na) - M(1-nc), or
        // b >= 1 - M + Mna - M + Mnc, or
        // b - Mna - Mnc >= 1 - 2M, or
        // -b + Mna + Mnc <= -1 + 2M
        let lb_b_explanation = new_explanation!(LinearLessOrEqual::new_expl(
            vec![
                b.flatten().scaled(-1),
                a.neg_aux(context.assignments).scaled(big_m),
                c.neg_aux(context.assignments).scaled(big_m),
            ]
            .non_zero_scale(),
            -1 + 2 * big_m,
            510,
        ));

        if 1 > b_min {
            prior_propagation = Some(2);
        }

        context.set_lower_bound(
            b,
            1,
            (conjunction!([a <= -1] & [c <= -1]), lb_b_explanation),
        )?;
    }

    // b is negative and c is negative -> a is positive
    if b_max <= -1 && c_max <= -1 {
        // Representing a >= 1 if b <= -1 && c <= -1
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m = (a_neg_ub_init + 1).max(0);

        // a >= 1 - M(1-nb) - M(1-nc), or
        // a >= 1 - M + Mnb - M + Mnc, or
        // a - Mnb - Mnc >= 1 - 2M, or
        // -a + Mnb + Mnc <= -1 + 2M
        let lb_a_explanation = new_explanation!(LinearLessOrEqual::new_expl(
            vec![
                a.flatten().scaled(-1),
                b.neg_aux(context.assignments).scaled(big_m),
                c.neg_aux(context.assignments).scaled(big_m),
            ]
            .non_zero_scale(),
            -1 + 2 * big_m,
            511,
        ));

        if 1 > a_min {
            prior_propagation = Some(1);
        }

        context.set_lower_bound(
            a,
            1,
            (conjunction!([b <= -1] & [c <= -1]), lb_a_explanation),
        )?;
    }

    // Propagating based on mixed bounds (i.e. one positive and one negative)
    // Propagating c based on a and b
    // a is negative and b is positive -> c is negative
    if a_max <= 0 && b_min >= 0 {
        // Representing c <= 0 if a <= 0 && b >= 0
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m = c_pos_ub_init.max(0);

        // c <= 0 + Mpa + Mnb, or
        // c - Mpa - Mnb <= 0
        let ub_c_explanation = new_explanation!(LinearLessOrEqual::new_expl(
            vec![
                c.flatten().scaled(1),
                a.pos_aux(context.assignments).scaled(-big_m),
                b.neg_aux(context.assignments).scaled(-big_m),
            ]
            .non_zero_scale(),
            0,
            512,
        ));

        context.set_upper_bound(c, 0, (conjunction!([a <= 0] & [b >= 0]), ub_c_explanation))?;
    }

    // a is positive and b is negative -> c is negative
    if a_min >= 0 && b_max <= 0 {
        // Representing c <= 0 if a >= 0 && b <= 0
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m = c_pos_ub_init.max(0);

        // c <= 0 + Mna + Mpb, or
        // c - Mna - Mpb <= 0
        let ub_c_explanation = new_explanation!(LinearLessOrEqual::new_expl(
            vec![
                c.flatten().scaled(1),
                a.neg_aux(context.assignments).scaled(-big_m),
                b.pos_aux(context.assignments).scaled(-big_m),
            ]
            .non_zero_scale(),
            0,
            513
        ));

        context.set_upper_bound(c, 0, (conjunction!([a >= 0] & [b <= 0]), ub_c_explanation))?;
    }

    // Propagating b based on a and c
    // a is negative and c is positive -> b is negative
    if a_max <= -1 && c_min >= 1 {
        // Representing b <= -1 if a <= -1 && c >= 1
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m = (b_pos_ub_init + 1).max(0);

        // b <= -1 + M(1-na) + M(1-pc), or
        // b <= -1 + M - Mna + M - Mpc, or
        // b + Mna + Mpc <= -1 + 2M
        let ub_b_explanation = new_explanation!(LinearLessOrEqual::new_expl(
            vec![
                b.flatten().scaled(1),
                a.neg_aux(context.assignments).scaled(big_m),
                c.pos_aux(context.assignments).scaled(big_m),
            ]
            .non_zero_scale(),
            -1 + 2 * big_m,
            514,
        ));

        if -1 < b_max {
            prior_propagation = Some(2);
        }

        context.set_upper_bound(
            b,
            -1,
            (conjunction!([a <= -1] & [c >= 1]), ub_b_explanation),
        )?;
    }

    // a is positive and c is negative -> b is negative
    if a_min >= 1 && c_max <= -1 {
        // Representing b <= -1 if a >= 1 && c <= -1
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m = (b_pos_ub_init + 1).max(0);

        // b <= -1 + M(1-pa) + M(1-nc), or
        // b <= -1 + M - Mpa + M - Mnc, or
        // b + Mpa + Mnc <= -1 + 2M
        let ub_b_explanation = new_explanation!(LinearLessOrEqual::new_expl(
            vec![
                b.flatten().scaled(1),
                a.pos_aux(context.assignments).scaled(big_m),
                c.neg_aux(context.assignments).scaled(big_m),
            ]
            .non_zero_scale(),
            -1 + 2 * big_m,
            515,
        ));

        if -1 < b_max {
            prior_propagation = Some(2);
        }

        context.set_upper_bound(
            b,
            -1,
            (conjunction!([a >= 1] & [c <= -1]), ub_b_explanation),
        )?;
    }

    // Propagating a based on b and c
    // b is negative and c is positive -> a is negative
    if b_max <= -1 && c_min >= 1 {
        // Representing a <= -1 if b <= -1 && c >= 1
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m = (a_pos_ub_init + 1).max(0);

        // a <= -1 + M(1-nb) + M(1-pc), or
        // a <= -1 + M - Mnb + M - Mpc, or
        // a + Mnb + Mpc <= -1 + 2M
        let ub_a_explanation = new_explanation!(LinearLessOrEqual::new_expl(
            vec![
                a.flatten().scaled(1),
                b.neg_aux(context.assignments).scaled(big_m),
                c.pos_aux(context.assignments).scaled(big_m),
            ]
            .non_zero_scale(),
            -1 + 2 * big_m,
            516,
        ));

        if -1 < a_max {
            prior_propagation = Some(1);
        }

        context.set_upper_bound(
            a,
            -1,
            (conjunction!([b <= -1] & [c >= 1]), ub_a_explanation),
        )?;
    }

    // b is positive and c is negative -> a is negative
    if b_min >= 1 && c_max <= -1 {
        // Representing a <= -1 if b >= 1 && c <= -1
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m = (a_pos_ub_init + 1).max(0);

        // a <= -1 + M(1-pb) + M(1-nc), or
        // a <= -1 + M - Mpb + M - Mnc, or
        // a + Mpb + Mnc <= -1 + 2M
        let ub_a_explanation = new_explanation!(LinearLessOrEqual::new_expl(
            vec![
                a.flatten().scaled(1),
                b.pos_aux(context.assignments).scaled(big_m),
                c.neg_aux(context.assignments).scaled(big_m),
            ]
            .non_zero_scale(),
            -1 + 2 * big_m,
            517,
        ));

        if -1 < a_max {
            prior_propagation = Some(1);
        }

        context.set_upper_bound(
            a,
            -1,
            (conjunction!([b >= 1] & [c <= -1]), ub_a_explanation),
        )?;
    }

    Ok(prior_propagation)
}

impl<VA, VB, VC> Debug for IntegerMultiplicationPropagator<VA, VB, VC>
where
    VA: IntegerVariable + 'static + Debug,
    VB: IntegerVariable + 'static + Debug,
    VC: IntegerVariable + 'static + Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntegerMultiplicationPropagator")
            .field("a", &self.a)
            .field("b", &self.b)
            .field("c", &self.c)
            .finish()
    }
}

/// Compute `ceil(numerator / denominator)`.
///
/// Assumes `numerator, denominator > 0`.
#[inline]
fn div_ceil_pos(numerator: i32, denominator: i32) -> i32 {
    pumpkin_assert_simple!(numerator > 0 && denominator > 0, "Either the numerator {numerator} was non-positive or the denominator {denominator} was non-positive");
    numerator / denominator + (numerator % denominator).signum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conjunction;
    use crate::engine::test_solver::TestSolver;
    use crate::predicate;

    #[test]
    fn bounds_of_a_and_b_propagate_bounds_c() {
        let mut solver = TestSolver::default();
        let a = solver.new_variable(1, 3);
        let b = solver.new_variable(0, 4);
        let c = solver.new_variable(-10, 20);

        let propagator = solver
            .new_propagator(IntegerMultiplicationPropagator::new(a, b, c))
            .expect("no empty domains");

        solver.propagate(propagator).expect("no empty domains");

        assert_eq!(1, solver.lower_bound(a));
        assert_eq!(3, solver.upper_bound(a));
        assert_eq!(0, solver.lower_bound(b));
        assert_eq!(4, solver.upper_bound(b));
        assert_eq!(0, solver.lower_bound(c));
        assert_eq!(12, solver.upper_bound(c));

        let reason_lb = solver.get_reason_int(predicate![c >= 0]);
        assert_eq!(conjunction!([a >= 0] & [b >= 0]), reason_lb);

        let reason_ub = solver.get_reason_int(predicate![c <= 12]);
        assert_eq!(
            conjunction!([a >= 0] & [a <= 3] & [b >= 0] & [b <= 4]),
            reason_ub
        );
    }

    #[test]
    fn bounds_of_a_and_c_propagate_bounds_b() {
        let mut solver = TestSolver::default();
        let a = solver.new_variable(2, 3);
        let b = solver.new_variable(0, 12);
        let c = solver.new_variable(2, 12);

        let propagator = solver
            .new_propagator(IntegerMultiplicationPropagator::new(a, b, c))
            .expect("no empty domains");

        solver.propagate(propagator).expect("no empty domains");

        assert_eq!(2, solver.lower_bound(a));
        assert_eq!(3, solver.upper_bound(a));
        assert_eq!(1, solver.lower_bound(b));
        assert_eq!(6, solver.upper_bound(b));
        assert_eq!(2, solver.lower_bound(c));
        assert_eq!(12, solver.upper_bound(c));

        let reason_lb = solver.get_reason_int(predicate![b >= 1]);
        assert_eq!(conjunction!([a >= 1] & [c >= 1]), reason_lb);

        let reason_ub = solver.get_reason_int(predicate![b <= 6]);
        assert_eq!(conjunction!([a >= 2] & [c >= 0] & [c <= 12]), reason_ub);
    }

    #[test]
    fn bounds_of_b_and_c_propagate_bounds_a() {
        let mut solver = TestSolver::default();
        let a = solver.new_variable(0, 10);
        let b = solver.new_variable(3, 6);
        let c = solver.new_variable(2, 12);

        let propagator = solver
            .new_propagator(IntegerMultiplicationPropagator::new(a, b, c))
            .expect("no empty domains");

        solver.propagate(propagator).expect("no empty domains");

        assert_eq!(1, solver.lower_bound(a));
        assert_eq!(4, solver.upper_bound(a));
        assert_eq!(3, solver.lower_bound(b));
        assert_eq!(6, solver.upper_bound(b));
        assert_eq!(3, solver.lower_bound(c));
        assert_eq!(12, solver.upper_bound(c));

        let reason_lb = solver.get_reason_int(predicate![a >= 1]);
        assert_eq!(conjunction!([b >= 1] & [c >= 1]), reason_lb);

        let reason_ub = solver.get_reason_int(predicate![a <= 4]);
        assert_eq!(conjunction!([b >= 3] & [c >= 0] & [c <= 12]), reason_ub);
    }
}
