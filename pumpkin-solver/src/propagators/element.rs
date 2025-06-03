use bitfield_struct::bitfield;
use itertools::Itertools;

use crate::basic_types::linear_less_or_equal::FilterNonZero;
use crate::basic_types::linear_less_or_equal::LinearLessOrEqual;
use crate::basic_types::PropagationReason;
use crate::basic_types::PropagationStatusCP;
use crate::conjunction;
use crate::engine::domain_events::DomainEvents;
use crate::engine::propagation::LocalId;
use crate::engine::propagation::PropagationContextMut;
use crate::engine::propagation::Propagator;
use crate::engine::propagation::PropagatorId;
use crate::engine::propagation::PropagatorInitialisationContext;
use crate::engine::propagation::ReadDomains;
use crate::engine::variables::IntegerVariable;
use crate::engine::Assignments;
use crate::engine::AuxiliaryVariable;
use crate::if_llg;
use crate::new_explanation;
use crate::predicate;
use crate::variables::DomainId;
use crate::variables::TransformableVariable;

/// Arc-consistent propagator for constraint `element([x_1, \ldots, x_n], i, e)`, where `x_j` are
///  variables, `i` is an integer variable, and `e` is a variable, which holds iff `x_i = e`
///
/// Note that this propagator is 0-indexed
#[derive(Clone, Debug)]
pub(crate) struct ElementPropagator<VX, VI, VE> {
    array: Box<[VX]>,
    index: VI,
    rhs: VE,
}

impl<VX, VI, VE> ElementPropagator<VX, VI, VE> {
    pub(crate) fn new(array: Box<[VX]>, index: VI, rhs: VE) -> Self {
        Self { array, index, rhs }
    }
}

const ID_INDEX: LocalId = LocalId::from(0);
const ID_RHS: LocalId = LocalId::from(1);

// local ids of array vars are shifted by ID_X_OFFSET
const ID_X_OFFSET: u32 = 2;

impl<VX, VI, VE> Propagator for ElementPropagator<VX, VI, VE>
where
    VX: IntegerVariable + 'static,
    VI: IntegerVariable + 'static,
    VE: IntegerVariable + 'static,
{
    fn priority(&self) -> u32 {
        2
    }

    fn name(&self) -> &str {
        "Element"
    }

    fn debug_propagate_from_scratch(
        &self,
        context: &mut PropagationContextMut,
    ) -> PropagationStatusCP {
        self.propagate_index_bounds_within_array(context)?;

        self.propagate_rhs_bounds_based_on_array(context)?;

        self.propagate_index_based_on_domain_intersection_with_rhs(context)?;

        if context.is_fixed(&self.index) {
            let idx = context.lower_bound(&self.index);
            self.propagate_equality(context, idx)?;
            let _ = if_llg!(self.propagate_auxiliaries(context, idx)?);
        }

        Ok(())
    }

    fn initialise_at_root(
        &mut self,
        context: &mut PropagatorInitialisationContext,
    ) -> Result<(), PropagationReason> {
        self.array.iter().enumerate().for_each(|(i, x_i)| {
            let _ = context.register(
                x_i.clone(),
                DomainEvents::ANY_INT,
                LocalId::from(i as u32 + ID_X_OFFSET),
            );
        });
        let _ = context.register(self.index.clone(), DomainEvents::ANY_INT, ID_INDEX);
        let _ = context.register(self.rhs.clone(), DomainEvents::ANY_INT, ID_RHS);
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

impl<VX, VI, VE> ElementPropagator<VX, VI, VE>
where
    VX: IntegerVariable + 'static,
    VI: IntegerVariable + 'static,
    VE: IntegerVariable + 'static,
{
    fn create_constraint_index_aux_lb(&self, auxes: &Vec<DomainId>) -> LinearLessOrEqual {
        let mut aux_idx_equal_gt_lhs = auxes
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx >= 1)
            .map(|(idx, aux)| aux.scaled(-(idx as i32)))
            .collect_vec();
        aux_idx_equal_gt_lhs.push(self.index.flatten());
        LinearLessOrEqual::new(aux_idx_equal_gt_lhs, 0)
    }

    fn create_constraint_index_aux_ub(&self, auxes: &Vec<DomainId>) -> LinearLessOrEqual {
        let mut aux_idx_equal_lt_lhs = auxes
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx >= 1)
            .map(|(idx, aux)| aux.scaled(idx as i32))
            .collect_vec();
        aux_idx_equal_lt_lhs.push(self.index.flatten().scaled(-1));
        LinearLessOrEqual::new(aux_idx_equal_lt_lhs, 0)
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

            // Constraint 1: maximum one aux is true
            // sum aux_i <= 1
            let one_aux_lt_lhs = array_index_auxes
                .iter()
                .map(|aux| aux.flatten())
                .collect_vec();
            let one_aux_lt = LinearLessOrEqual::new(one_aux_lt_lhs, 1);

            // Constraint 2: at least one aux is true
            // sum aux_i => 1, or -sum aux <= -1
            let one_aux_gt_lhs = array_index_auxes
                .iter()
                .map(|aux| aux.flatten().scaled(-1))
                .collect_vec();
            let one_aux_gt = LinearLessOrEqual::new(one_aux_gt_lhs, -1);

            // Constraint 3: index = aux that is true, low side
            // sum i * aux_i <= idx, or sum i * aux_i - idx <= 0
            let aux_idx_equal_lt = self.create_constraint_index_aux_ub(&array_index_auxes);

            // Constraint 4: index = aux that is true, high side
            // sum i * aux_i => idx, or -sum (i * aux_i) + idx <= 0
            let aux_idx_equal_gt = self.create_constraint_index_aux_lb(&array_index_auxes);

            assignments
                .new_auxiliaries
                .push(AuxiliaryVariable::Unlinked(
                    array_index_auxes,
                    vec![one_aux_lt, one_aux_gt, aux_idx_equal_lt, aux_idx_equal_gt],
                ))
        }

        assignments
            .unlinked_aux_variables_for_prop(propagator_id)
            .unwrap()
    }

    fn create_explanation(
        &self,
        assignments: &mut Assignments,
        propagator_id: PropagatorId,
        var_idx: usize,
        increase_lower_bound_rhs: bool,
    ) -> LinearLessOrEqual {
        // Now that we have the auxiliaries, we can create an explanation. Copied from v1:
        // We know that rhs = a_i if i = idx. We can represent that using rhs >= a_i and rhs <= a_i,
        // given that i = idx.
        let p_i = self.create_unlinked_auxes(assignments, propagator_id)[var_idx];

        let var = &self.array[var_idx];

        let rhs_lb = self.rhs.lower_bound_initial(assignments);
        let rhs_ub = self.rhs.upper_bound_initial(assignments);
        let var_lb = var.lower_bound_initial(assignments);
        let var_ub = var.upper_bound_initial(assignments);

        // Option 1: rhs <= a_i if p_i, or
        // rhs - a_i <= M(1-p_i), or
        // rhs - a_i + Mp_i <= M
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m_opt_1 = (rhs_ub - var_lb).max(0);
        let opt_1 = LinearLessOrEqual::new_expl(
            vec![
                self.rhs.flatten().scaled(1),
                var.flatten().scaled(-1),
                p_i.scaled(big_m_opt_1),
            ]
            .non_zero_scale(),
            big_m_opt_1,
            700,
        );

        // Option 2: rhs >= a_i if p_i, or
        // rhs - a_i >= -M(1-p_i), or
        // -rhs + a_i + Mp_i <= M
        // If M <= 0, this condition always holds so we do not need any auxiliary.
        let big_m_opt_2 = (-rhs_lb + var_ub).max(0);
        let opt_2 = LinearLessOrEqual::new_expl(
            vec![
                self.rhs.flatten().scaled(-1),
                var.flatten().scaled(1),
                p_i.scaled(big_m_opt_2),
            ]
            .non_zero_scale(),
            big_m_opt_2,
            701,
        );

        if increase_lower_bound_rhs {
            opt_2
        } else {
            opt_1
        }
    }

    /// Propagate the bounds of `self.index` to be in the range `[0, self.array.len())`.
    fn propagate_index_bounds_within_array(
        &self,
        context: &mut PropagationContextMut<'_>,
    ) -> PropagationStatusCP {
        context.set_lower_bound(&self.index, 0, conjunction!())?;
        context.set_upper_bound(&self.index, self.array.len() as i32 - 1, conjunction!())?;
        Ok(())
    }

    /// The lower bound (resp. upper bound) of the right-hand side can be the minimum lower
    /// bound (res. maximum upper bound) of the elements.
    fn propagate_rhs_bounds_based_on_array(
        &self,
        context: &mut PropagationContextMut<'_>,
    ) -> PropagationStatusCP {
        let mut rhs_lb = i32::MAX;
        let mut idx_lb = 0;

        let mut rhs_ub = i32::MIN;
        let mut idx_ub = 0;

        self.array
            .iter()
            .enumerate()
            .filter(|(idx, _)| self.index.contains(context.assignments, *idx as i32))
            .for_each(|(idx, element)| {
                let lb = context.lower_bound(element);
                if lb < rhs_lb {
                    (rhs_lb, idx_lb) = (lb, idx)
                }

                let ub = context.upper_bound(element);
                if ub > rhs_ub {
                    (rhs_ub, idx_ub) = (ub, idx)
                }
            });

        let lb_conjunction = self
            .array
            .iter()
            .enumerate()
            .map(|(idx, var)| {
                if self.index.contains(context.assignments, idx as i32) {
                    predicate![var >= rhs_lb]
                } else {
                    predicate![self.index != idx as i32]
                }
            })
            .collect();

        let lb_linleq = new_explanation!(self.create_explanation(
            context.assignments,
            context.propagator_id,
            idx_lb,
            true
        ));

        let ub_conjunction = self
            .array
            .iter()
            .enumerate()
            .map(|(idx, var)| {
                if self.index.contains(context.assignments, idx as i32) {
                    predicate![var <= rhs_ub]
                } else {
                    predicate![self.index != idx as i32]
                }
            })
            .collect();

        let ub_linleq = new_explanation!(self.create_explanation(
            context.assignments,
            context.propagator_id,
            idx_ub,
            false
        ));

        context.set_lower_bound(&self.rhs, rhs_lb, (lb_conjunction, lb_linleq))?;
        context.set_upper_bound(&self.rhs, rhs_ub, (ub_conjunction, ub_linleq))?;

        Ok(())
    }

    /// Go through the array. For every element for which the domain does not intersect with the
    /// right-hand side, remove it from index.
    fn propagate_index_based_on_domain_intersection_with_rhs(
        &self,
        context: &mut PropagationContextMut<'_>,
    ) -> PropagationStatusCP {
        let rhs_lb = context.lower_bound(&self.rhs);
        let rhs_ub = context.upper_bound(&self.rhs);
        let mut to_remove = vec![];

        let domain = context.iterate_domain(&self.index).collect_vec();
        for idx in domain {
            let element = &self.array[idx as usize];

            let element_ub = context.upper_bound(element);
            let element_lb = context.lower_bound(element);

            // TODO linear explanation commented out as it is hard to get it to work with the new
            // aux variables, and it will very likely never cause a propagation.
            // let reason_linleq = self.index_remove_explanation(context, idx, element);

            let reason = if rhs_lb > element_ub {
                conjunction!([element <= rhs_lb - 1] & [self.rhs >= rhs_lb])
            } else if rhs_ub < element_lb {
                conjunction!([element >= rhs_ub + 1] & [self.rhs <= rhs_ub])
            } else {
                continue;
            };

            to_remove.push((idx, reason));
        }

        for (idx, reason) in to_remove.drain(..) {
            context.remove(&self.index, idx, reason)?;
        }

        Ok(())
    }

    /// Propagate equality between lhs and rhs. This assumes the bounds of rhs have already been
    /// tightened to the bounds of lhs, through a previous propagation rule.
    fn propagate_equality(
        &self,
        context: &mut PropagationContextMut<'_>,
        index: i32,
    ) -> PropagationStatusCP {
        let rhs_lb = context.lower_bound(&self.rhs);
        let rhs_ub = context.upper_bound(&self.rhs);
        let lhs = &self.array[index as usize];

        let lb_linleq = new_explanation!(self.create_explanation(
            context.assignments,
            context.propagator_id,
            index as usize,
            true,
        ));
        context.set_lower_bound(
            lhs,
            rhs_lb,
            (
                conjunction!([self.rhs >= rhs_lb] & [self.index == index]),
                lb_linleq,
            ),
        )?;

        let ub_linleq = new_explanation!(self.create_explanation(
            context.assignments,
            context.propagator_id,
            index as usize,
            false,
        ));
        context.set_upper_bound(
            lhs,
            rhs_ub,
            (
                conjunction!([self.rhs <= rhs_ub] & [self.index == index]),
                ub_linleq,
            ),
        )?;
        Ok(())
    }

    /// Propagate the auxiliaries if index is fixed.
    fn propagate_auxiliaries(
        &self,
        context: &mut PropagationContextMut<'_>,
        index: i32,
    ) -> PropagationStatusCP {
        let auxes = self
            .create_unlinked_auxes(context.assignments, context.propagator_id)
            .clone();

        let aux_idx_equal_lb = self.create_constraint_index_aux_lb(&auxes);
        let aux_idx_equal_ub = self.create_constraint_index_aux_ub(&auxes);

        // For each aux, fix it in case this is not done yet
        // Skip i == 0, as this is not part of the lb/ub inequalities
        auxes
            .iter()
            .enumerate()
            .filter(|(aux_i, _)| *aux_i >= 1)
            .try_for_each(|(aux_i, aux)| {
                if !context.is_fixed(aux) {
                    if aux_i == index as usize {
                        context.set_lower_bound(
                            aux,
                            1,
                            (
                                conjunction!([self.index == index]),
                                aux_idx_equal_lb.clone(),
                            ),
                        )?;
                    } else {
                        context.set_upper_bound(
                            aux,
                            0,
                            (
                                conjunction!([self.index == index]),
                                aux_idx_equal_ub.clone(),
                            ),
                        )?;
                    }
                }

                return Ok(());
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Bound {
    Lower = 0,
    Upper = 1,
}

impl Bound {
    const fn into_bits(self) -> u8 {
        self as _
    }

    const fn from_bits(value: u8) -> Self {
        match value {
            0 => Bound::Lower,
            _ => Bound::Upper,
        }
    }
}

#[bitfield(u64)]
struct RightHandSideReason {
    #[bits(32, from = Bound::from_bits)]
    bound: Bound,
    value: i32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conjunction;
    use crate::engine::test_solver::TestSolver;
    use crate::predicate;

    #[test]
    fn elements_from_array_with_disjoint_domains_to_rhs_are_filtered_from_index() {
        let mut solver = TestSolver::default();

        let x_0 = solver.new_variable(4, 6);
        let x_1 = solver.new_variable(2, 3);
        let x_2 = solver.new_variable(7, 9);
        let x_3 = solver.new_variable(14, 15);

        let index = solver.new_variable(0, 3);
        let rhs = solver.new_variable(6, 9);

        let _ = solver
            .new_propagator(ElementPropagator::new(
                vec![x_0, x_1, x_2, x_3].into(),
                index,
                rhs,
            ))
            .expect("no empty domains");

        solver.assert_bounds(index, 0, 2);

        assert_eq!(
            solver.get_reason_int(predicate![index != 3]),
            conjunction!([x_3 >= 10] & [rhs <= 9])
        );

        assert_eq!(
            solver.get_reason_int(predicate![index != 1]),
            conjunction!([x_1 <= 5] & [rhs >= 6])
        );
    }

    #[test]
    fn bounds_of_rhs_are_min_and_max_of_lower_and_upper_in_array() {
        let mut solver = TestSolver::default();

        let x_0 = solver.new_variable(3, 10);
        let x_1 = solver.new_variable(2, 3);
        let x_2 = solver.new_variable(7, 9);
        let x_3 = solver.new_variable(14, 15);

        let index = solver.new_variable(0, 3);
        let rhs = solver.new_variable(0, 20);

        let _ = solver
            .new_propagator(ElementPropagator::new(
                vec![x_0, x_1, x_2, x_3].into(),
                index,
                rhs,
            ))
            .expect("no empty domains");

        solver.assert_bounds(rhs, 2, 15);

        assert_eq!(
            solver.get_reason_int(predicate![rhs >= 2]),
            conjunction!([x_0 >= 2] & [x_1 >= 2] & [x_2 >= 2] & [x_3 >= 2])
        );

        assert_eq!(
            solver.get_reason_int(predicate![rhs <= 15]),
            conjunction!([x_0 <= 15] & [x_1 <= 15] & [x_2 <= 15] & [x_3 <= 15])
        );
    }

    #[test]
    fn fixed_index_propagates_bounds_on_element() {
        let mut solver = TestSolver::default();

        let x_0 = solver.new_variable(3, 10);
        let x_1 = solver.new_variable(0, 15);
        let x_2 = solver.new_variable(7, 9);
        let x_3 = solver.new_variable(14, 15);

        let index = solver.new_variable(1, 1);
        let rhs = solver.new_variable(6, 9);

        let _ = solver
            .new_propagator(ElementPropagator::new(
                vec![x_0, x_1, x_2, x_3].into(),
                index,
                rhs,
            ))
            .expect("no empty domains");

        solver.assert_bounds(x_1, 6, 9);

        assert_eq!(
            solver.get_reason_int(predicate![x_1 >= 6]),
            conjunction!([index == 1] & [rhs >= 6])
        );

        assert_eq!(
            solver.get_reason_int(predicate![x_1 <= 9]),
            conjunction!([index == 1] & [rhs <= 9])
        );
    }

    #[test]
    fn index_hole_propagates_bounds_on_rhs() {
        let mut solver = TestSolver::default();

        let x_0 = solver.new_variable(3, 10);
        let x_1 = solver.new_variable(0, 15);
        let x_2 = solver.new_variable(7, 9);
        let x_3 = solver.new_variable(14, 15);

        let index = solver.new_variable(0, 3);
        solver.remove(index, 1).expect("Value can be removed");

        let rhs = solver.new_variable(-10, 30);

        let _ = solver
            .new_propagator(ElementPropagator::new(
                vec![x_0, x_1, x_2, x_3].into(),
                index,
                rhs,
            ))
            .expect("no empty domains");

        solver.assert_bounds(rhs, 3, 15);

        assert_eq!(
            solver.get_reason_int(predicate![rhs >= 3]),
            conjunction!([x_0 >= 3] & [x_2 >= 3] & [x_3 >= 3] & [index != 1])
        );

        assert_eq!(
            solver.get_reason_int(predicate![rhs <= 15]),
            conjunction!([x_0 <= 15] & [x_2 <= 15] & [x_3 <= 15] & [index != 1])
        );
    }
}
