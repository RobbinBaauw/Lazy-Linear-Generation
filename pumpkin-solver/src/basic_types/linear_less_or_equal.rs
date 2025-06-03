use std::fmt::Display;
use std::fmt::Formatter;
use std::rc::Rc;
use std::slice::Iter;
use std::slice::IterMut;

use itertools::Itertools;

use crate::basic_types::HashMap;
use crate::engine::Assignments;
use crate::pumpkin_assert_simple;
use crate::variables::AffineView;
use crate::variables::DomainId;
use crate::variables::IntegerVariable;
use crate::variables::TransformableVariable;

#[derive(Default, Debug, Clone, Eq, Hash)]
pub struct LinearLessOrEqualLhs(pub Vec<AffineView<DomainId>>);

impl LinearLessOrEqualLhs {
    pub(crate) fn contains_variable(&self, variable: DomainId) -> bool {
        self.iter()
            .find(|var| var.domain_id() == variable)
            .is_some()
    }

    pub(crate) fn find_variable_scale(&self, variable: DomainId) -> Option<i32> {
        self.iter()
            .find(|var| var.domain_id() == variable)
            .map(|var| var.scale)
    }

    fn lb_overflows(&self, assignments: &Assignments, trail_position: usize) -> bool {
        let elements_overflow = self.iter().any(|var| {
            let bound = if var.scale < 0 {
                var.domain_id()
                    .upper_bound_at_trail_position_i64(assignments, trail_position)
            } else {
                var.domain_id()
                    .lower_bound_at_trail_position_i64(assignments, trail_position)
            };

            let Ok(bound) = i32::try_from(bound) else {
                return true;
            };

            var.scale.checked_mul(bound).is_none()
        });
        if elements_overflow {
            return true;
        }

        i32::try_from(self.lb(assignments, trail_position, false)).is_err()
    }

    pub(crate) fn lb(
        &self,
        assignments: &Assignments,
        trail_position: usize,
        eval_linleqs: bool,
    ) -> i64 {
        self.iter()
            .map(|var| {
                var.lower_bound_at_trail_position_eval(assignments, trail_position, eval_linleqs)
            })
            .sum::<i64>()
    }

    pub(crate) fn lb_initial(&self, assignments: &Assignments) -> i64 {
        self.iter()
            .map(|var| var.lower_bound_initial(assignments) as i64)
            .sum::<i64>()
    }

    pub(crate) fn ub(
        &self,
        assignments: &Assignments,
        trail_position: usize,
        eval_linleqs: bool,
    ) -> i64 {
        self.iter()
            .map(|var| {
                var.upper_bound_at_trail_position_eval(assignments, trail_position, eval_linleqs)
            })
            .sum::<i64>()
    }

    pub(crate) fn ub_initial(&self, assignments: &Assignments) -> i64 {
        self.iter()
            .map(|var| var.upper_bound_initial(assignments) as i64)
            .sum::<i64>()
    }

    pub(crate) fn iter(&self) -> Iter<'_, AffineView<DomainId>> {
        self.0.iter()
    }

    pub(crate) fn iter_mut(&mut self) -> IterMut<'_, AffineView<DomainId>> {
        self.0.iter_mut()
    }
}

impl From<Vec<AffineView<DomainId>>> for LinearLessOrEqualLhs {
    fn from(value: Vec<AffineView<DomainId>>) -> Self {
        LinearLessOrEqualLhs(value)
    }
}

impl From<Vec<(DomainId, i32)>> for LinearLessOrEqualLhs {
    fn from(value: Vec<(DomainId, i32)>) -> Self {
        LinearLessOrEqualLhs(value.iter().map(|(id, scale)| id.scaled(*scale)).collect())
    }
}

impl<Var> From<&Rc<[Var]>> for LinearLessOrEqualLhs
where
    Var: IntegerVariable + 'static,
{
    fn from(value: &Rc<[Var]>) -> Self {
        LinearLessOrEqualLhs(value.iter().map(|var| var.flatten()).collect())
    }
}

impl<Var> From<&Box<[Var]>> for LinearLessOrEqualLhs
where
    Var: IntegerVariable + 'static,
{
    fn from(value: &Box<[Var]>) -> Self {
        LinearLessOrEqualLhs(value.iter().map(|var| var.flatten()).collect())
    }
}

impl PartialEq for LinearLessOrEqualLhs {
    fn eq(&self, other: &Self) -> bool {
        let self_sorted = self.iter().sorted_by_key(|var| var.domain_id().id);
        let other_sorted = other.iter().sorted_by_key(|var| var.domain_id().id);
        self_sorted.eq(other_sorted)
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct LinearLessOrEqual {
    pub lhs: LinearLessOrEqualLhs,
    pub rhs: i32,

    pub explanation_id: Option<u32>,
}

impl LinearLessOrEqual {
    pub fn new<L: Into<LinearLessOrEqualLhs>>(lhs: L, rhs: i32) -> Self {
        let mut lhs_with_offset = lhs.into();

        let var_offsets = lhs_with_offset.0.iter().map(|var| var.offset).sum::<i32>();
        let rhs = rhs - var_offsets;

        lhs_with_offset.iter_mut().for_each(|var| var.offset = 0);

        // Remove duplicate entries: turn 6x + 5x + 3z into 11x + 3z
        let mut domain_entries: HashMap<DomainId, i32> = HashMap::default();

        lhs_with_offset.iter().for_each(|var| {
            let curr_scale = domain_entries.entry(var.domain_id()).or_insert_with(|| 0);
            *curr_scale += var.scale
        });

        let dedup_lhs = domain_entries
            .iter()
            .filter_map(|(domain_id, scale)| {
                if *scale != 0 {
                    Some(domain_id.scaled(*scale))
                } else {
                    None
                }
            })
            .collect_vec()
            .into();

        Self {
            lhs: dedup_lhs,
            rhs,
            explanation_id: None,
        }
    }

    pub(crate) fn new_expl<L: Into<LinearLessOrEqualLhs>>(
        lhs: L,
        rhs: i32,
        explanation_id: u32,
    ) -> Self {
        let lhs = lhs.into();

        // Check that an explanation doesn't contain the 0 term
        pumpkin_assert_simple!(lhs.iter().all(|var| var.scale != 0));

        let mut new = Self::new(lhs, rhs);
        new.explanation_id = Some(explanation_id);
        new
    }

    pub(crate) fn evaluate_at_trail_position(
        &self,
        assignments: &Assignments,
        trail_position: usize,
        eval_linleqs: bool,
    ) -> Option<bool> {
        let ub_lhs = self.lhs.ub(assignments, trail_position, eval_linleqs);
        let lb_lhs = self.lhs.lb(assignments, trail_position, eval_linleqs);

        if ub_lhs <= self.rhs as i64 {
            Some(true)
        } else if lb_lhs > self.rhs as i64 {
            Some(false)
        } else {
            None
        }
    }

    pub(crate) fn slack(
        &self,
        assignments: &Assignments,
        trail_position: usize,
        eval_linleqs: bool,
    ) -> i64 {
        (self.rhs as i64) - self.lhs.lb(assignments, trail_position, eval_linleqs)
    }

    pub(crate) fn is_conflicting(
        &self,
        assignments: &Assignments,
        trail_position: usize,
        eval_linleqs: bool,
    ) -> bool {
        self.slack(assignments, trail_position, eval_linleqs) < 0
    }

    pub(crate) fn variables_propagating(
        &self,
        assignments: &Assignments,
        trail_position: usize,
        eval_linleqs: bool,
    ) -> Vec<(AffineView<DomainId>, i32)> {
        let lb_lhs = self.lhs.lb(assignments, trail_position, eval_linleqs);

        self.lhs
            .0
            .iter()
            .filter_map(|var| {
                let var_lower_bound = var.lower_bound_at_trail_position_eval(
                    assignments,
                    trail_position,
                    eval_linleqs,
                );
                let var_upper_bound = var.upper_bound_at_trail_position_eval(
                    assignments,
                    trail_position,
                    eval_linleqs,
                );

                let bound = (self.rhs as i64) - (lb_lhs - var_lower_bound);
                if var_upper_bound > bound {
                    Some((*var, bound as i32))
                } else {
                    None
                }
            })
            .collect()
    }

    pub(crate) fn is_propagating(
        &self,
        assignments: &Assignments,
        trail_position: usize,
        eval_linleqs: bool,
    ) -> bool {
        self.variables_propagating(assignments, trail_position, eval_linleqs)
            .len()
            > 0
    }

    pub(crate) fn overflows(&self, assignments: &Assignments, trail_position: usize) -> bool {
        if self.lhs.lb_overflows(assignments, trail_position) {
            return true;
        }

        let slack = self.slack(assignments, trail_position, false);
        for var in self.lhs.iter() {
            let var_lb = var.lower_bound_at_trail_position_i64(assignments, trail_position);
            if i32::try_from(var_lb).is_err() {
                return true;
            };

            if i32::try_from(slack + var_lb).is_err() {
                return true;
            };
        }

        false
    }

    pub(crate) fn invert(&self) -> Self {
        // Turn Ax <= b into
        // * Ax > b...
        // * Ax >= b + 1...
        // * -Ax <= -b - 1
        let new_lhs = self.lhs.iter().map(|var| var.scaled(-1)).collect_vec();

        Self {
            lhs: LinearLessOrEqualLhs::from(new_lhs),
            rhs: -self.rhs - 1,
            explanation_id: self.explanation_id,
        }
    }
}

impl Display for LinearLessOrEqual {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let lhs_mapped = self
            .lhs
            .0
            .iter()
            .sorted_by_key(|var| var.domain_id().id)
            .filter_map(|var| {
                let s = var.scale;
                let v = var.domain_id().id;

                return if s == 0 {
                    None
                } else if s == 1 {
                    Some(format!("x{v}"))
                } else if s == -1 {
                    Some(format!("-x{v}"))
                } else {
                    Some(format!("{s}x{v}"))
                };
            })
            .join(" + ");
        let mut res = format!("{lhs_mapped} <= {:?}", self.rhs);
        if res.len() > 10000000 {
            res.truncate(300);
            write!(f, "{}...", res)
        } else {
            write!(f, "{}", res)
        }
    }
}

impl AffineView<DomainId> {
    fn lower_bound_at_trail_position_eval(
        &self,
        assignments: &Assignments,
        trail_position: usize,
        eval_linleqs: bool,
    ) -> i64 {
        self.aux_eval(assignments, trail_position, eval_linleqs)
            .unwrap_or_else(|| self.lower_bound_at_trail_position_i64(assignments, trail_position))
    }

    fn upper_bound_at_trail_position_eval(
        &self,
        assignments: &Assignments,
        trail_position: usize,
        eval_linleqs: bool,
    ) -> i64 {
        self.aux_eval(assignments, trail_position, eval_linleqs)
            .unwrap_or_else(|| self.upper_bound_at_trail_position_i64(assignments, trail_position))
    }

    fn aux_eval(
        &self,
        assignments: &Assignments,
        trail_position: usize,
        eval_linleqs: bool,
    ) -> Option<i64> {
        if !eval_linleqs {
            return None;
        }

        if let Some(aux_linleq) = assignments.get_auxiliary_linleq(self.domain_id()) {
            // Prevent infinite recursion by only evaluating 1 level deep
            if let Some(is_true) =
                aux_linleq.evaluate_at_trail_position(assignments, trail_position, false)
            {
                // Observation: if you can definitely say something about the aux variable,
                // you don't need to bother with lower/upper bound, as they are equal.
                return Some(self.map_i64(is_true.into()));
            }
        }

        None
    }
}

pub(crate) trait FilterNonZero {
    fn non_zero_scale(self) -> LinearLessOrEqualLhs;
}

impl<L: Into<LinearLessOrEqualLhs>> FilterNonZero for L {
    fn non_zero_scale(self) -> LinearLessOrEqualLhs {
        let mut lhs = self.into();
        lhs.0.retain(|v| v.scale != 0);
        lhs
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use itertools::Itertools;

    use crate::basic_types::linear_less_or_equal::LinearLessOrEqualLhs;
    use crate::basic_types::LinearLessOrEqual;
    use crate::engine::Assignments;
    use crate::variables::AffineView;
    use crate::variables::DomainId;
    use crate::variables::IntegerVariable;
    use crate::variables::TransformableVariable;

    #[test]
    fn test_contains_variable() {
        let domains = vec![
            DomainId::new(0).scaled(5),
            DomainId::new(1).scaled(2),
            DomainId::new(2).scaled(-5),
        ];

        let lhs = LinearLessOrEqualLhs(vec![domains[0], domains[1]]);
        assert!(lhs.contains_variable(domains[0].domain_id()));
        assert!(!lhs.contains_variable(domains[2].domain_id()))
    }

    #[test]
    fn test_find_variable_scale() {
        let domains = vec![
            DomainId::new(0).scaled(5),
            DomainId::new(1).scaled(2),
            DomainId::new(2).scaled(-5),
        ];

        let lhs = LinearLessOrEqualLhs(vec![domains[0], domains[1]]);
        assert_eq!(lhs.find_variable_scale(domains[0].domain_id()), Some(5));
        assert_eq!(lhs.find_variable_scale(domains[1].domain_id()), Some(2));
        assert_eq!(lhs.find_variable_scale(domains[2].domain_id()), None);
    }

    #[test]
    fn test_lb_overflows_any_element_high() {
        let mut assignments = Assignments::default();
        let d1 = assignments.grow(i32::MAX - 10, i32::MAX - 10);
        let lhs = LinearLessOrEqualLhs(vec![d1.scaled(10)]);

        assert!(lhs.lb_overflows(&assignments, assignments.num_trail_entries() - 1));
    }

    #[test]
    fn test_lb_overflows_any_element_low() {
        let mut assignments = Assignments::default();
        let d1 = assignments.grow(i32::MAX - 10, i32::MAX - 10);
        let lhs = LinearLessOrEqualLhs(vec![d1.scaled(-10)]);

        assert!(lhs.lb_overflows(&assignments, assignments.num_trail_entries() - 1));
    }

    #[test]
    fn test_lb_overflows_sum() {
        let mut assignments = Assignments::default();
        let d1 = assignments.grow(i32::MAX / 3, i32::MAX / 3);
        let d2 = assignments.grow(i32::MAX / 3, i32::MAX / 3);
        let lhs = LinearLessOrEqualLhs(vec![d1.scaled(2), d2.scaled(2)]);

        assert!(lhs.lb_overflows(&assignments, assignments.num_trail_entries() - 1));
    }

    #[test]
    fn test_lb() {
        let mut assignments = Assignments::default();
        let d1 = assignments.grow(-20, 20);
        let d2 = assignments.grow(50, 80);

        let curr_trail_level = assignments.num_trail_entries() - 1;
        d1.set_upper_bound(&mut assignments, 10, None).unwrap();

        let lhs = LinearLessOrEqualLhs(vec![d1.scaled(-3), d2.scaled(10)]);

        assert_eq!(lhs.lb(&assignments, curr_trail_level, false), 440);
    }

    #[test]
    fn test_lb_eval_linleq() {
        let mut assignments = Assignments::default();
        let d1 = assignments.grow(-20, 20);
        let d2 = assignments.new_linked_aux_variable(LinearLessOrEqual::new(vec![d1.scaled(1)], 0));
        let lhs = LinearLessOrEqualLhs(vec![d1.scaled(-3), d2.scaled(10)]);

        let curr_trail_level = assignments.num_trail_entries() - 1;
        d1.set_upper_bound(&mut assignments, 5, None).unwrap();

        assert_eq!(lhs.lb(&assignments, curr_trail_level, true), -60);

        d1.set_upper_bound(&mut assignments, 0, None).unwrap();
        assert_eq!(
            lhs.lb(&assignments, assignments.num_trail_entries() - 1, true),
            10
        );
    }

    #[test]
    fn test_lb_initial() {
        let mut assignments = Assignments::default();
        let d1 = assignments.grow(-20, 20);
        let d2 = assignments.grow(50, 80);
        let lhs = LinearLessOrEqualLhs(vec![d1.scaled(-3), d2.scaled(10)]);

        assert_eq!(lhs.lb_initial(&assignments), 440);
    }

    #[test]
    fn test_ub() {
        let mut assignments = Assignments::default();
        let d1 = assignments.grow(-20, 20);
        let d2 = assignments.grow(50, 80);
        let lhs = LinearLessOrEqualLhs(vec![d1.scaled(-3), d2.scaled(10)]);

        let curr_trail_level = assignments.num_trail_entries() - 1;
        d1.set_lower_bound(&mut assignments, -10, None).unwrap();

        assert_eq!(lhs.ub(&assignments, curr_trail_level, false), 860);
    }

    #[test]
    fn test_ub_eval_linleq() {
        let mut assignments = Assignments::default();
        let d1 = assignments.grow(-20, 20);
        let d2 = assignments.new_linked_aux_variable(LinearLessOrEqual::new(vec![d1.scaled(1)], 0));
        let lhs = LinearLessOrEqualLhs(vec![d1.scaled(-3), d2.scaled(10)]);

        let curr_trail_level = assignments.num_trail_entries() - 1;
        d1.set_lower_bound(&mut assignments, 0, None).unwrap();

        assert_eq!(lhs.ub(&assignments, curr_trail_level, true), 70);

        d1.set_lower_bound(&mut assignments, 1, None).unwrap();
        assert_eq!(
            lhs.ub(&assignments, assignments.num_trail_entries() - 1, true),
            -3
        );
    }

    #[test]
    fn test_ub_initial() {
        let mut assignments = Assignments::default();
        let d1 = assignments.grow(-20, 20);
        let d2 = assignments.grow(50, 80);
        let lhs = LinearLessOrEqualLhs(vec![d1.scaled(-3), d2.scaled(10)]);

        assert_eq!(lhs.ub_initial(&assignments), 860);
    }

    #[test]
    fn test_iter() {
        let mut assignments = Assignments::default();
        let d1 = assignments.grow(-20, 20);
        let d2 = assignments.grow(50, 80);
        let lhs = LinearLessOrEqualLhs(vec![d1.scaled(-3), d2.scaled(10)]);

        let iter_vec = lhs.iter().collect_vec();

        assert_eq!(iter_vec, vec![&d1.scaled(-3), &d2.scaled(10)]);
    }

    #[test]
    fn test_iter_mut() {
        let mut assignments = Assignments::default();
        let d1 = assignments.grow(-20, 20);
        let d2 = assignments.grow(50, 80);
        let mut lhs = LinearLessOrEqualLhs(vec![d1.scaled(-3), d2.scaled(10)]);

        lhs.iter_mut().for_each(|var| *var = var.scaled(2));

        assert_eq!(
            lhs.lb(&assignments, assignments.num_trail_entries() - 1, false),
            880
        );
    }

    #[test]
    fn test_lhs_from_domain_scale() {
        let d1 = DomainId::new(0);
        let d2 = DomainId::new(1);

        let lhs_1 = LinearLessOrEqualLhs::from(vec![(d1, -3), (d2, 10)]);
        let lhs_2 = LinearLessOrEqualLhs(vec![d1.scaled(-3), d2.scaled(10)]);

        assert_eq!(lhs_1, lhs_2);
    }

    #[test]
    fn test_lhs_from_vars_box() {
        let d1 = DomainId::new(0);
        let d2 = DomainId::new(1);

        let lhs_1 = LinearLessOrEqualLhs::from(
            &vec![
                AffineView::new(d1.scaled(-3), 2, 3),
                AffineView::new(d2.scaled(10), 2, 5),
            ]
            .into_boxed_slice(),
        );
        let lhs_2 = LinearLessOrEqualLhs(vec![
            d1.scaled(-3).scaled(2).offset(3),
            d2.scaled(10).scaled(2).offset(5),
        ]);

        assert_eq!(lhs_1, lhs_2);
    }

    #[test]
    fn test_lhs_from_vars_rc() {
        let d1 = DomainId::new(0);
        let d2 = DomainId::new(1);

        let lhs_rc: Rc<[AffineView<AffineView<DomainId>>]> = vec![
            AffineView::new(d1.scaled(-3), 2, 3),
            AffineView::new(d2.scaled(10), 2, 5),
        ]
        .into();

        let lhs_1 = LinearLessOrEqualLhs::from(&lhs_rc);
        let lhs_2 = LinearLessOrEqualLhs(vec![
            d1.scaled(-3).scaled(2).offset(3),
            d2.scaled(10).scaled(2).offset(5),
        ]);

        assert_eq!(lhs_1, lhs_2);
    }

    #[test]
    fn test_lhs_eq() {
        let d1_s1 = DomainId::new(1).scaled(1);
        let d1_s2 = DomainId::new(1).scaled(2);
        let d2 = DomainId::new(2).scaled(10);
        let d3 = DomainId::new(3).scaled(-10);
        let d4 = DomainId::new(4).scaled(1);

        let lhs_1 = LinearLessOrEqualLhs::from(vec![d1_s1, d2, d3, d4]);
        let lhs_2 = LinearLessOrEqualLhs::from(vec![d1_s2, d2, d3, d4]);
        assert_ne!(lhs_1, lhs_2);

        let lhs_1 = LinearLessOrEqualLhs::from(vec![d1_s1, d2, d3, d4]);
        let lhs_2 = LinearLessOrEqualLhs::from(vec![d2, d1_s1, d3, d4]);
        assert_eq!(lhs_1, lhs_2);

        let lhs_1 = LinearLessOrEqualLhs::from(vec![d1_s1, d2, d3, d4]);
        let lhs_2 = LinearLessOrEqualLhs::from(vec![d1_s1, d2, d3, d4]);
        assert_eq!(lhs_1, lhs_2);
    }

    #[test]
    fn test_new_less_equal_offsets() {
        let d1 = DomainId::new(1);
        let d2 = DomainId::new(2);
        let d3 = DomainId::new(3);
        let less_equal = LinearLessOrEqual::new(
            vec![
                d1.scaled(1).offset(-10),
                d2.scaled(2).offset(1),
                d3.scaled(10).offset(0),
            ],
            5,
        );

        assert_eq!(
            less_equal.lhs,
            vec![d1.scaled(1), d2.scaled(2), d3.scaled(10),].into()
        );
        assert_eq!(less_equal.rhs, 14)
    }

    #[test]
    fn test_new_less_equal_dedup() {
        let d1 = DomainId::new(1);
        let d2 = DomainId::new(2);
        let less_equal = LinearLessOrEqual::new(
            vec![
                d1.scaled(1).offset(-10),
                d1.scaled(2).offset(1),
                d2.scaled(10).offset(0),
            ],
            5,
        );

        assert_eq!(less_equal.lhs, vec![d1.scaled(3), d2.scaled(10),].into());
        assert_eq!(less_equal.rhs, 14)
    }

    #[test]
    fn test_evaluate_at_trail() {
        let mut assignments = Assignments::default();
        let d1 = assignments.grow(-5, 5);
        let d2 = assignments.grow(-5, 5);
        let less_equal = LinearLessOrEqual::new(vec![d1.scaled(1), d2.scaled(1)], 0);

        let before_trail_level = assignments.num_trail_entries() - 1;
        d1.set_upper_bound(&mut assignments, -5, None).unwrap();
        let after_trail_level = assignments.num_trail_entries() - 1;

        assert_eq!(
            less_equal.evaluate_at_trail_position(&assignments, before_trail_level, false),
            None
        );
        assert_eq!(
            less_equal.evaluate_at_trail_position(&assignments, after_trail_level, false),
            Some(true)
        );
    }

    #[test]
    fn test_slack() {
        let mut assignments = Assignments::default();
        let d1 = assignments.grow(-5, 5);
        let d2 = assignments.grow(-5, 5);
        let less_equal = LinearLessOrEqual::new(vec![d1.scaled(1), d2.scaled(1)], 0);

        let before_trail_level = assignments.num_trail_entries() - 1;
        d1.set_lower_bound(&mut assignments, 0, None).unwrap();

        assert_eq!(
            less_equal.slack(&assignments, before_trail_level, false),
            10
        );
    }

    #[test]
    fn test_is_conflicting() {
        let mut assignments = Assignments::default();
        let d1 = assignments.grow(-5, 5);
        let d2 = assignments.grow(-5, 5);
        let less_equal = LinearLessOrEqual::new(vec![d1.scaled(1), d2.scaled(1)], 0);

        let before_trail_level = assignments.num_trail_entries() - 1;
        d1.set_lower_bound(&mut assignments, 5, None).unwrap();
        d2.set_lower_bound(&mut assignments, -4, None).unwrap();
        let after_trail_level = assignments.num_trail_entries() - 1;

        assert!(!less_equal.is_conflicting(&assignments, before_trail_level, false));
        assert!(less_equal.is_conflicting(&assignments, after_trail_level, false));
    }

    #[test]
    fn test_variables_propagating() {
        let mut assignments = Assignments::default();
        let d1 = assignments.grow(-5, 5);
        let d2 = assignments.grow(-5, 5);
        let less_equal = LinearLessOrEqual::new(vec![d1.scaled(1), d2.scaled(1)], 0);

        let before_trail_level = assignments.num_trail_entries() - 1;
        d1.set_lower_bound(&mut assignments, 0, None).unwrap();
        let after_trail_level = assignments.num_trail_entries() - 1;

        assert_eq!(
            less_equal.variables_propagating(&assignments, before_trail_level, false),
            vec![]
        );
        assert!(!less_equal.is_propagating(&assignments, before_trail_level, false));

        assert_eq!(
            less_equal.variables_propagating(&assignments, after_trail_level, false),
            vec![(d2.scaled(1), 0)]
        );
        assert!(less_equal.is_propagating(&assignments, after_trail_level, false));
    }

    #[test]
    fn test_overflows() {
        let mut assignments = Assignments::default();
        let d1 = assignments.grow(-100, 0);
        let d2 = assignments.grow(-100, 0);
        let less_equal = LinearLessOrEqual::new(vec![d1.scaled(1), d2.scaled(1)], i32::MAX - 50);

        let before_trail_level = assignments.num_trail_entries() - 1;
        d1.set_lower_bound(&mut assignments, 0, None).unwrap();
        d2.set_lower_bound(&mut assignments, 0, None).unwrap();
        let after_trail_level = assignments.num_trail_entries() - 1;

        assert!(less_equal.overflows(&assignments, before_trail_level));
        assert!(!less_equal.overflows(&assignments, after_trail_level));
    }

    #[test]
    fn test_invert() {
        let d1 = DomainId::new(1);
        let d2 = DomainId::new(2);
        let less_equal = LinearLessOrEqual::new(vec![d1.scaled(1), d2.scaled(2)], 5);

        assert_eq!(
            less_equal.invert(),
            LinearLessOrEqual::new(vec![d1.scaled(-1), d2.scaled(-2)], -6,)
        )
    }
}
