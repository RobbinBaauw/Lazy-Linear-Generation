use crate::basic_types::linear_less_or_equal::FilterNonZero;
use crate::basic_types::linear_less_or_equal::LinearLessOrEqual;
use crate::basic_types::PropagationReason;
use crate::basic_types::PropagationStatusCP;
use crate::conjunction;
use crate::engine::cp::propagation::propagation_context::ReadDomains;
use crate::engine::propagation::LocalId;
use crate::engine::propagation::PropagationContextMut;
use crate::engine::propagation::Propagator;
use crate::engine::propagation::PropagatorInitialisationContext;
use crate::engine::variables::IntegerVariable;
use crate::engine::Assignments;
use crate::engine::DomainEvents;
use crate::new_explanation;
use crate::pumpkin_assert_simple;
use crate::variables::AffineView;
use crate::variables::DomainId;
use crate::variables::TransformableVariable;

/// A propagator for maintaining the constraint `numerator / denominator = rhs`; note that this
/// propagator performs truncating division (i.e. rounding towards 0).
///
/// The propagator assumes that the `denominator` is a (non-zero) number.
///
/// The implementation is ported from [OR-tools](https://github.com/google/or-tools/blob/870edf6f7bff6b8ff0d267d936be7e331c5b8c2d/ortools/sat/integer_expr.cc#L1209C1-L1209C19).
#[derive(Clone, Debug)]
pub(crate) struct DivisionPropagator<VA, VB, VC> {
    numerator: VA,
    denominator: VB,
    rhs: VC,
}

const ID_NUMERATOR: LocalId = LocalId::from(0);
const ID_DENOMINATOR: LocalId = LocalId::from(1);
const ID_RHS: LocalId = LocalId::from(2);

impl<VA, VB, VC> DivisionPropagator<VA, VB, VC>
where
    VA: IntegerVariable + 'static,
    VB: IntegerVariable + 'static,
    VC: IntegerVariable + 'static,
{
    pub(crate) fn new(numerator: VA, denominator: VB, rhs: VC) -> Self {
        DivisionPropagator {
            numerator,
            denominator,
            rhs,
        }
    }
}

impl<VA: 'static, VB: 'static, VC: 'static> Propagator for DivisionPropagator<VA, VB, VC>
where
    VA: IntegerVariable,
    VB: IntegerVariable,
    VC: IntegerVariable,
{
    fn priority(&self) -> u32 {
        1
    }

    fn name(&self) -> &str {
        "Division"
    }

    fn initialise_at_root(
        &mut self,
        context: &mut PropagatorInitialisationContext,
    ) -> Result<(), PropagationReason> {
        pumpkin_assert_simple!(
            !context.contains(&self.denominator, 0),
            "Denominator cannot contain 0"
        );
        let _ = context.register(self.numerator.clone(), DomainEvents::BOUNDS, ID_NUMERATOR);
        let _ = context.register(
            self.denominator.clone(),
            DomainEvents::BOUNDS,
            ID_DENOMINATOR,
        );
        let _ = context.register(self.rhs.clone(), DomainEvents::BOUNDS, ID_RHS);

        Ok(())
    }

    fn debug_propagate_from_scratch(
        &self,
        context: &mut PropagationContextMut,
    ) -> PropagationStatusCP {
        perform_propagation(context, &self.numerator, &self.denominator, &self.rhs)
    }
}

fn explain_lb_num<VA: IntegerVariable, VB: IntegerVariable, VC: IntegerVariable>(
    numerator: &VA,
    denominator: &VB,
    rhs: &VC,
    assignments: &mut Assignments,
) -> LinearLessOrEqual {
    // NOTE: This method will serve as documentation for all related explanations.
    //
    // We have seen this propagation before, in the multiplication (check explain_lb_a).
    // There, we defined four inequalities:
    // * a_min * b <= c
    // * a_max * b >= c
    // * a * b_min <= c
    // * a * b_max >= c
    //
    // We can adapt these to work for the flooring division. Note that all of these constraints
    // require us to work only with positives (same with integer multiplication). So
    // the main adaptation that we need to do, is to observe that num can now be den-1 larger
    // without affecting the result of rhs. So we for instance get num <= (rhs+1) * den - 1.
    //
    // We update all the explanations accordingly, using a = rhs, c = num, b = den:
    // * rhs_min * den <= num
    // * rhs_max * den + den - 1 >= num
    // * rhs * den_min <= num
    // * rhs * den_max + den - 1 >= num
    //
    // The positivity conditions depend on the specific propagation that is currently being
    // done, similarly to the integer multiplication. Going over each propagation:
    // * lb(num): rhs_min * den <= num if den >= 1 and rhs >= 0. Equivalent idea for the alternative
    //   formulation.
    //
    // * lb(den): rhs_max * den + den - 1 >= num if num >= 0 and rhs >= 0
    //
    // * lb(rhs): rhs * den_max + den - 1 >= num if num >= 0 and den >= 1
    //
    // * ub(num): rhs_max * den + den - 1 >= num if rhs >= 0 and den >= 1. Equivalent idea for the
    //   alternative formulation.
    //
    // * ub(den): rhs_min * den <= num if num >= 0 and rhs >= 0
    //
    // * ub(rhs): rhs * den_min <= num if num >= 0 and den >= 1
    //
    // Again, we also need to introduce the proper conditions that check whether den <= den_max,
    // etc.

    // lb(num): rhs_min * den <= num if den >= 1 and rhs >= 0. Alternative formulation is equiv.
    // In this case, we pick rhs or den based on which difference between the current and initial
    // bounds is largest.

    let rhs_lb_init = rhs.lower_bound_initial(assignments);
    let rhs_lb = rhs.lower_bound(assignments);
    let rhs_diff = rhs_lb - rhs_lb_init;

    let den_lb_init = denominator.lower_bound_initial(assignments);
    let den_lb = denominator.lower_bound(assignments);
    let den_diff = den_lb - den_lb_init;

    let mut create_explanation =
        |var: AffineView<DomainId>, min_var: AffineView<DomainId>, min_var_lb| {
            // In these comments, I assume min_var = den and var = rhs

            let x_min_ub_init = var.scaled(min_var_lb).upper_bound_initial(assignments);
            let num_neg_ub_init = numerator.scaled(-1).upper_bound_initial(assignments);
            let big_m = (x_min_ub_init + num_neg_ub_init).max(0);

            // Assume we chose to use den_min:
            // rhs * den_min <= num + M(1-is_min) + Mn_rhs + M(1-p_den)
            // rhs * den_min - num + Mis_max - Mn_rhs + Mp_den <= 2M
            LinearLessOrEqual::new_expl(
                vec![
                    var.flatten().scaled(min_var_lb),
                    numerator.flatten().scaled(-1),
                    min_var.min_aux(assignments, min_var_lb).scaled(big_m),
                    rhs.neg_aux(assignments).scaled(-big_m),
                    denominator.pos_aux(assignments).scaled(big_m),
                ]
                .non_zero_scale(),
                2 * big_m,
                600,
            )
        };

    if rhs_diff > den_diff {
        create_explanation(denominator.flatten(), rhs.flatten(), rhs_lb)
    } else {
        create_explanation(rhs.flatten(), denominator.flatten(), den_lb)
    }
}

fn explain_lb_den<VA: IntegerVariable, VB: IntegerVariable, VC: IntegerVariable>(
    numerator: &VA,
    denominator: &VB,
    rhs: &VC,
    assignments: &mut Assignments,
) -> LinearLessOrEqual {
    // lb(den): rhs_max * den + den - 1 >= num if num >= 0 and rhs >= 0
    let rhs_max = rhs.upper_bound(assignments);

    let den_rhsmax_ub_init = denominator
        .scaled(-rhs_max)
        .upper_bound_initial(assignments);
    let den_neg_ub_init = denominator.scaled(-1).upper_bound_initial(assignments);
    let num_ub_init = numerator.upper_bound_initial(assignments);
    let big_m = (den_rhsmax_ub_init + den_neg_ub_init + num_ub_init + 1).max(0);

    // rhs_max * den + den - 1 >= num - M(1-is_rhs_max) - Mn_num - Mn_rhs
    // rhs_max * den + den - num - Mis_rhs_max + Mn_num + Mn_rhs >= 1 - M
    // -rhs_max * den - den + num + Mis_rhs_max - Mn_num - Mn_rhs <= -1 + M
    LinearLessOrEqual::new_expl(
        vec![
            denominator.flatten().scaled(-rhs_max),
            denominator.flatten().scaled(-1),
            numerator.flatten().scaled(1),
            rhs.max_aux(assignments, rhs_max).scaled(big_m),
            numerator.neg_aux(assignments).scaled(-big_m),
            rhs.neg_aux(assignments).scaled(-big_m),
        ]
        .non_zero_scale(),
        -1 + big_m,
        601,
    )
}

fn explain_lb_rhs<VA: IntegerVariable, VB: IntegerVariable, VC: IntegerVariable>(
    numerator: &VA,
    denominator: &VB,
    rhs: &VC,
    assignments: &mut Assignments,
) -> LinearLessOrEqual {
    // lb(rhs): rhs * den_max + den - 1 >= num if num >= 0 and den >= 1
    let den_max = denominator.upper_bound(assignments);

    let rhs_denmax_ub_init = rhs.scaled(-den_max).upper_bound_initial(assignments);
    let den_neg_ub_init = denominator.scaled(-1).upper_bound_initial(assignments);
    let num_ub_init = numerator.upper_bound_initial(assignments);
    let big_m = (rhs_denmax_ub_init + den_neg_ub_init + num_ub_init + 1).max(0);

    // rhs * den_max + den - 1 >= num - M(1-is_den_max) - Mn_num - M(1-p_den)
    // rhs * den_max + den - num - Mis_den_max + Mn_num - Mp_den >= 1 - 2M
    // rhs * -den_max - den + num + Mis_den_max - Mn_num + Mp_den <= -1 + 2M
    LinearLessOrEqual::new_expl(
        vec![
            rhs.flatten().scaled(-den_max),
            denominator.flatten().scaled(-1),
            numerator.flatten().scaled(1),
            denominator.max_aux(assignments, den_max).scaled(big_m),
            numerator.neg_aux(assignments).scaled(-big_m),
            denominator.pos_aux(assignments).scaled(big_m),
        ]
        .non_zero_scale(),
        -1 + 2 * big_m,
        602,
    )
}

fn explain_ub_num<VA: IntegerVariable, VB: IntegerVariable, VC: IntegerVariable>(
    numerator: &VA,
    denominator: &VB,
    rhs: &VC,
    assignments: &mut Assignments,
) -> LinearLessOrEqual {
    // ub(num): rhs_max * den + den - 1 >= num if rhs >= 0 and den >= 1. Alternative formulation is
    // equiv. In this case, we pick rhs or den based on which difference between the current and
    // initial bounds is largest.

    let rhs_ub_init = rhs.upper_bound_initial(assignments);
    let rhs_ub = rhs.upper_bound(assignments);
    let rhs_diff = rhs_ub_init - rhs_ub;

    let den_ub_init = denominator.upper_bound_initial(assignments);
    let den_ub = denominator.upper_bound(assignments);
    let den_diff = den_ub_init - den_ub;

    let mut create_explanation =
        |var: AffineView<DomainId>, max_var: AffineView<DomainId>, max_var_ub: i32| {
            // In these comments, I assume max_var = den and var = rhs

            let x_max_ub_init = var.scaled(-max_var_ub).upper_bound_initial(assignments);
            let den_neg_ub_init = denominator.scaled(-1).upper_bound_initial(assignments);
            let num_ub_init = numerator.upper_bound_initial(assignments);
            let big_m = (x_max_ub_init + den_neg_ub_init + num_ub_init + 1).max(0);

            // Assume we chose to use den_max:
            // rhs * den_max + den - 1 >= num - M(1-is_max) - Mn_rhs - M(1-p_den)
            // rhs * den_max + den - num - Mis_max + Mn_rhs - Mp_den >= 1 - 2M
            // rhs * -den_max - den + num + Mis_max - Mn_rhs + Mp_den <= -1 + 2M
            LinearLessOrEqual::new_expl(
                vec![
                    var.flatten().scaled(-max_var_ub),
                    denominator.flatten().scaled(-1),
                    numerator.flatten().scaled(1),
                    max_var.max_aux(assignments, max_var_ub).scaled(big_m),
                    rhs.neg_aux(assignments).scaled(-big_m),
                    denominator.pos_aux(assignments).scaled(big_m),
                ]
                .non_zero_scale(),
                -1 + 2 * big_m,
                603,
            )
        };

    if rhs_diff > den_diff {
        create_explanation(denominator.flatten(), rhs.flatten(), rhs_ub)
    } else {
        create_explanation(rhs.flatten(), denominator.flatten(), den_ub)
    }
}

fn explain_ub_den<VA: IntegerVariable, VB: IntegerVariable, VC: IntegerVariable>(
    numerator: &VA,
    denominator: &VB,
    rhs: &VC,
    assignments: &mut Assignments,
) -> LinearLessOrEqual {
    // ub(den): rhs_min * den <= num if num >= 0 and rhs >= 0
    let rhs_min = rhs.lower_bound(assignments);

    let den_rhsmin_ub_init = denominator.scaled(rhs_min).upper_bound_initial(assignments);
    let num_neg_ub_init = numerator.scaled(-1).upper_bound_initial(assignments);
    let big_m = (den_rhsmin_ub_init + num_neg_ub_init).max(0);

    // rhs_min * den <= num + M(1-is_rhs_min) + Mn_num + Mn_rhs
    // rhs_min * den - num + Mis_rhs_min - Mn_num - Mn_rhs <= M
    LinearLessOrEqual::new_expl(
        vec![
            denominator.flatten().scaled(rhs_min),
            numerator.flatten().scaled(-1),
            rhs.min_aux(assignments, rhs_min).scaled(big_m),
            numerator.neg_aux(assignments).scaled(-big_m),
            rhs.neg_aux(assignments).scaled(-big_m),
        ]
        .non_zero_scale(),
        big_m,
        604,
    )
}

fn explain_ub_rhs<VA: IntegerVariable, VB: IntegerVariable, VC: IntegerVariable>(
    numerator: &VA,
    denominator: &VB,
    rhs: &VC,
    assignments: &mut Assignments,
) -> LinearLessOrEqual {
    // ub(rhs): rhs * den_min <= num if num >= 0 and den >= 1
    let den_min = denominator.lower_bound(assignments);

    let rhs_denmin_ub_init = rhs.scaled(den_min).upper_bound_initial(assignments);
    let num_neg_ub_init = numerator.scaled(-1).upper_bound_initial(assignments);
    let big_m = (rhs_denmin_ub_init + num_neg_ub_init).max(0);

    // rhs * den_min <= num + M(1-is_den_min) + Mn_num + M(1-p_den)
    // rhs * den_min - num + Mis_den_min - Mn_num + Mp_den <= 2M
    LinearLessOrEqual::new_expl(
        vec![
            rhs.flatten().scaled(den_min),
            numerator.flatten().scaled(-1),
            denominator.min_aux(assignments, den_min).scaled(big_m),
            numerator.neg_aux(assignments).scaled(-big_m),
            denominator.pos_aux(assignments).scaled(big_m),
        ]
        .non_zero_scale(),
        2 * big_m,
        605,
    )
}

fn perform_propagation<VA: IntegerVariable, VB: IntegerVariable, VC: IntegerVariable>(
    context: &mut PropagationContextMut,
    numerator: &VA,
    denominator: &VB,
    rhs: &VC,
) -> PropagationStatusCP {
    if context.lower_bound(denominator) < 0 && context.upper_bound(denominator) > 0 {
        // For now we don't do anything in this case, note that this will not lead to incorrect
        // behaviour since any solution to this constraint will necessarily have to fix the
        // denominator.
        return Ok(());
    }

    let mut negated_numerator = &numerator.scaled(-1);
    let mut numerator = &numerator.scaled(1);

    let mut negated_denominator = &denominator.scaled(-1);
    let mut denominator = &denominator.scaled(1);

    if context.upper_bound(denominator) < 0 {
        // If the denominator is negative then we swap the numerator with its negated version and we
        // swap the denominator with its negated version.
        std::mem::swap(&mut numerator, &mut negated_numerator);
        std::mem::swap(&mut denominator, &mut negated_denominator);
    }

    let negated_rhs = &rhs.scaled(-1);

    // We propagate the domains to their appropriate signs (e.g. if the numerator is negative and
    // the denominator is positive then the rhs should also be negative)
    propagate_signs(context, numerator, denominator, rhs)?;

    // If the upper-bound of the numerator is positive and the upper-bound of the rhs is positive
    // then we can simply update the upper-bounds
    if context.upper_bound(numerator) >= 0 && context.upper_bound(rhs) >= 0 {
        propagate_upper_bounds(context, numerator, denominator, rhs)?;
    }

    // If the lower-bound of the numerator is negative and the lower-bound of the rhs is negative
    // then we negate these variables and update the upper-bounds
    if context.upper_bound(negated_numerator) >= 0 && context.upper_bound(negated_rhs) >= 0 {
        propagate_upper_bounds(context, negated_numerator, denominator, negated_rhs)?;
    }

    // If the domain of the numerator is positive and the domain of the rhs is positive (and we know
    // that our denominator is positive) then we can propagate based on the assumption that all the
    // domains are positive
    if context.lower_bound(numerator) >= 0 && context.lower_bound(rhs) >= 0 {
        propagate_positive_domains(context, numerator, denominator, rhs)?;
    }

    // If the domain of the numerator is negative and the domain of the rhs is negative (and we know
    // that our denominator is positive) then we propagate based on the views over the numerator and
    // rhs
    if context.lower_bound(negated_numerator) >= 0 && context.lower_bound(negated_rhs) >= 0 {
        propagate_positive_domains(context, negated_numerator, denominator, negated_rhs)?;
    }

    Ok(())
}

/// Propagates the domains of variables if all the domains are positive (if the variables are
/// sign-fixed then we simply transform them to positive domains using [`AffineView`]s); it performs
/// the following propagations:
/// - The minimum value that division can take on is the smallest value that `numerator /
///   denominator` can take on
/// - The numerator is at least as large as the smallest value that `denominator * rhs` can take on
/// - The value of the denominator is smaller than the largest value that `numerator / rhs` can take
///   on
/// - The denominator is at least as large as the ratio between the largest ceiled ratio between
///   `numerator + 1` and `rhs + 1`
fn propagate_positive_domains<VA: IntegerVariable, VB: IntegerVariable, VC: IntegerVariable>(
    context: &mut PropagationContextMut,
    numerator: &VA,
    denominator: &VB,
    rhs: &VC,
) -> PropagationStatusCP {
    let rhs_min = context.lower_bound(rhs);
    let rhs_max = context.upper_bound(rhs);
    let numerator_min = context.lower_bound(numerator);
    let numerator_max = context.upper_bound(numerator);
    let denominator_min = context.lower_bound(denominator);
    let denominator_max = context.upper_bound(denominator);

    // The new minimum value of the rhs is the minimum value that the division can take on
    let new_min_rhs = numerator_min / denominator_max;
    if rhs_min < new_min_rhs {
        let lb_rhs_explanation = new_explanation!(explain_lb_rhs(
            numerator,
            denominator,
            rhs,
            context.assignments
        ));
        context.set_lower_bound(
            rhs,
            new_min_rhs,
            (
                conjunction!(
                    [numerator >= numerator_min]
                        & [denominator <= denominator_max]
                        & [denominator >= 1]
                ),
                lb_rhs_explanation,
            ),
        )?;
    }

    // numerator / denominator >= rhs_min
    // numerator >= rhs_min * denominator
    // numerator >= rhs_min * denominator_min
    // Note that we use rhs_min rather than new_min_rhs, this appears to be a heuristic
    let new_min_numerator = denominator_min * rhs_min;
    if numerator_min < new_min_numerator {
        let lb_num_explanation = new_explanation!(explain_lb_num(
            numerator,
            denominator,
            rhs,
            context.assignments
        ));
        context.set_lower_bound(
            numerator,
            new_min_numerator,
            (
                conjunction!([denominator >= denominator_min] & [rhs >= rhs_min]),
                lb_num_explanation,
            ),
        )?;
    }

    // numerator / denominator >= rhs_min
    // numerator >= rhs_min * denominator
    // If rhs_min == 0 -> no propagations
    // Otherwise, denominator <= numerator / rhs_min & denominator <= numerator_max / rhs_min
    if rhs_min > 0 {
        let new_max_denominator = numerator_max / rhs_min;
        if denominator_max > new_max_denominator {
            let ub_den_explanation = new_explanation!(explain_ub_den(
                numerator,
                denominator,
                rhs,
                context.assignments
            ));
            context.set_upper_bound(
                denominator,
                new_max_denominator,
                (
                    conjunction!(
                        [numerator <= numerator_max]
                            & [numerator >= 0]
                            & [rhs >= rhs_min]
                            & [denominator >= 1]
                    ),
                    ub_den_explanation,
                ),
            )?;
        }
    }

    let new_min_denominator = {
        // Called the CeilRatio in OR-tools
        let dividend = numerator_min + 1;
        let positive_divisor = rhs_max + 1;

        let result = dividend / positive_divisor;
        let adjust = result * positive_divisor < dividend;
        result + adjust as i32
    };

    if denominator_min < new_min_denominator {
        let lb_den_explanation = new_explanation!(explain_lb_den(
            numerator,
            denominator,
            rhs,
            context.assignments
        ));
        context.set_lower_bound(
            denominator,
            new_min_denominator,
            (
                conjunction!(
                    [numerator >= numerator_min]
                        & [rhs <= rhs_max]
                        & [rhs >= 0]
                        & [denominator >= 1]
                ),
                lb_den_explanation,
            ),
        )?;
    }

    Ok(())
}

/// Propagates the upper-bounds of the right-hand side and the numerator, it performs the following
/// propagations
/// - The maximum value of the right-hand side can only be as large as the largest value that
///   `numerator / denominator` can take on
/// - The maximum value of the numerator is smaller than `(ub(rhs) + 1) * denominator - 1`, note
///   that this might not be the most constrictive bound
fn propagate_upper_bounds<VA: IntegerVariable, VB: IntegerVariable, VC: IntegerVariable>(
    context: &mut PropagationContextMut,
    numerator: &VA,
    denominator: &VB,
    rhs: &VC,
) -> PropagationStatusCP {
    let rhs_max = context.upper_bound(rhs);
    let numerator_max = context.upper_bound(numerator);
    let denominator_min = context.lower_bound(denominator);
    let denominator_max = context.upper_bound(denominator);

    // The new maximum value of the rhs is the maximum value that the division can take on (note
    // that numerator_max is positive and denominator_min is also positive)
    let new_max_rhs = numerator_max / denominator_min;
    if rhs_max > new_max_rhs {
        let ub_rhs_explanation = new_explanation!(explain_ub_rhs(
            numerator,
            denominator,
            rhs,
            context.assignments
        ));
        context.set_upper_bound(
            rhs,
            new_max_rhs,
            (
                conjunction!([numerator <= numerator_max] & [denominator >= denominator_min]),
                ub_rhs_explanation,
            ),
        )?;
    }

    // numerator / denominator <= rhs.max
    // numerator < (rhs.max + 1) * denominator
    // numerator + 1 <= (rhs.max + 1) * denominator.max
    // numerator <= (rhs.max + 1) * denominator.max - 1
    // Note that we use rhs_max here rather than the new upper-bound, this appears to be a heuristic
    let new_max_numerator = (rhs_max + 1) * denominator_max - 1;
    if numerator_max > new_max_numerator {
        let ub_num_explanation = new_explanation!(explain_ub_num(
            numerator,
            denominator,
            rhs,
            context.assignments
        ));
        context.set_upper_bound(
            numerator,
            new_max_numerator,
            (
                conjunction!(
                    [denominator <= denominator_max] & [denominator >= 1] & [rhs <= rhs_max]
                ),
                ub_num_explanation,
            ),
        )?;
    }

    Ok(())
}

/// Propagates the signs of the variables, more specifically, it performs the following propagations
/// (assuming that the denominator is always > 0):
/// - If the numerator is non-negative then the right-hand side must be non-negative as well
/// - If the right-hand side is positive then the numerator must be positive as well
/// - If the numerator is non-positive then the right-hand side must be non-positive as well
/// - If the right-hand is negative then the numerator must be negative as well
fn propagate_signs<VA: IntegerVariable, VB: IntegerVariable, VC: IntegerVariable>(
    context: &mut PropagationContextMut,
    numerator: &VA,
    denominator: &VB,
    rhs: &VC,
) -> PropagationStatusCP {
    let rhs_min = context.lower_bound(rhs);
    let rhs_max = context.upper_bound(rhs);
    let numerator_min = context.lower_bound(numerator);
    let numerator_max = context.upper_bound(numerator);

    // First we propagate the signs
    // If the numerator >= 0 (and we know that denominator > 0) then the rhs must be >= 0
    if numerator_min >= 0 && rhs_min < 0 {
        // Representing rhs >= 0 if num >= 0 and den >= 1
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m = rhs
            .scaled(-1)
            .upper_bound_initial(context.assignments)
            .max(0);

        // rhs >= 0 - Mn_num - M(1-p_den)
        // rhs + Mn_num - Mp_den >= -M
        // -rhs - Mn_num + Mp_den <= M
        let lb_rhs_explanation = new_explanation!(LinearLessOrEqual::new_expl(
            vec![
                rhs.flatten().scaled(-1),
                numerator.neg_aux(context.assignments).scaled(-big_m),
                denominator.pos_aux(context.assignments).scaled(big_m),
            ]
            .non_zero_scale(),
            big_m,
            606,
        ));

        context.set_lower_bound(
            rhs,
            0,
            (
                conjunction!([numerator >= 0] & [denominator >= 1]),
                lb_rhs_explanation,
            ),
        )?;
    }

    // If rhs > 0 (and we know that denominator > 0) then the numerator must be > 0
    if numerator_min <= 0 && rhs_min > 0 {
        // Representing num >= 1 if rhs > 0 and den > 0
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m = (numerator
            .scaled(-1)
            .upper_bound_initial(context.assignments)
            + 1)
        .max(0);

        // num >= 1 - M(1-p_rhs) - M(1-p_den)
        // num - Mp_rhs - Mp_den >= 1 - 2M
        // -num + Mp_rhs + Mp_den <= -1 + 2M
        let lb_num_explanation = new_explanation!(LinearLessOrEqual::new_expl(
            vec![
                numerator.flatten().scaled(-1),
                rhs.pos_aux(context.assignments).scaled(big_m),
                denominator.pos_aux(context.assignments).scaled(big_m),
            ]
            .non_zero_scale(),
            -1 + 2 * big_m,
            607,
        ));

        context.set_lower_bound(
            numerator,
            1,
            (
                conjunction!([rhs >= 1] & [denominator >= 1]),
                lb_num_explanation,
            ),
        )?;
    }

    // If numerator <= 0 (and we know that denominator > 0) then the rhs must be <= 0
    if numerator_max <= 0 && rhs_max > 0 {
        // Representing rhs <= 0 if num <= 0 and den > 0
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m = rhs.upper_bound_initial(context.assignments).max(0);

        // rhs <= 0 + Mp_num + M(1-p_den)
        // rhs - Mp_num + Mp_den <= M
        let ub_rhs_explanation = new_explanation!(LinearLessOrEqual::new_expl(
            vec![
                rhs.flatten(),
                numerator.pos_aux(context.assignments).scaled(-big_m),
                denominator.pos_aux(context.assignments).scaled(big_m),
            ]
            .non_zero_scale(),
            big_m,
            608,
        ));

        context.set_upper_bound(
            rhs,
            0,
            (
                conjunction!([numerator <= 0] & [denominator >= 1]),
                ub_rhs_explanation,
            ),
        )?;
    }

    // If the rhs < 0 (and we know that denominator > 0) then the numerator must be < 0
    if numerator_max >= 0 && rhs_max < 0 {
        // Representing num < 0 if rhs < 0 and den > 0
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m = (numerator.upper_bound_initial(context.assignments) + 1).max(0);

        // num <= -1 + M(1-n_rhs) + M(1-p_den)
        // num + Mn_rhs + Mp_den <= -1 + 2M
        let num_ub_explanation = new_explanation!(LinearLessOrEqual::new_expl(
            vec![
                numerator.flatten(),
                rhs.neg_aux(context.assignments).scaled(big_m),
                denominator.pos_aux(context.assignments).scaled(big_m),
            ]
            .non_zero_scale(),
            -1 + 2 * big_m,
            609,
        ));

        context.set_upper_bound(
            numerator,
            -1,
            (
                conjunction!([rhs <= -1] & [denominator >= 1]),
                num_ub_explanation,
            ),
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::test_solver::TestSolver;

    #[test]
    fn detects_conflicts() {
        let mut solver = TestSolver::default();
        let numerator = solver.new_variable(1, 1);
        let denominator = solver.new_variable(2, 2);
        let rhs = solver.new_variable(2, 2);

        let propagator =
            solver.new_propagator(DivisionPropagator::new(numerator, denominator, rhs));

        assert!(propagator.is_err());
    }
}
