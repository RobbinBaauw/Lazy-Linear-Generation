use std::fmt::Debug;

use super::propagation::store::PropagatorStore;
use super::propagation::ExplanationContext;
use super::propagation::PropagatorId;
use crate::basic_types::linear_less_or_equal::LinearLessOrEqual;
use crate::basic_types::PropagationReason;
use crate::basic_types::PropositionalConjunction;
use crate::basic_types::Trail;
use crate::predicates::Predicate;
use crate::pumpkin_assert_simple;

/// The reason store holds a reason for each change made by a CP propagator on a trail.
#[derive(Default, Debug)]
pub(crate) struct ReasonStore {
    trail: Trail<(PropagatorId, Reason)>,
    pub helper: PropositionalConjunction,
}

impl ReasonStore {
    pub(crate) fn push(&mut self, propagator: PropagatorId, reason: Reason) -> ReasonRef {
        let index = self.trail.len();
        self.trail.push((propagator, reason));
        pumpkin_assert_simple!(
            index < (1 << 30),
            "ReasonRef in reason store should fit in ContraintReference, \
             which has 30 bits available at most"
        );
        ReasonRef(index as u32)
    }

    pub(crate) fn get_or_compute<'this>(
        &'this self,
        reference: ReasonRef,
        context: ExplanationContext<'_>,
        propagators: &'this mut PropagatorStore,
    ) -> Option<PropagationReasonRef<'this>> {
        self.trail
            .get(reference.0 as usize)
            .map(|reason| reason.1.compute(context, reason.0, propagators))
    }

    pub(crate) fn get_lazy_code(&self, reference: ReasonRef) -> Option<&u64> {
        match self.trail.get(reference.0 as usize) {
            Some(reason) => match &reason.1 {
                Reason::Eager(_) => None,
                Reason::DynamicLazy(code) => Some(code),
            },
            None => None,
        }
    }

    pub(crate) fn increase_decision_level(&mut self) {
        self.trail.increase_decision_level()
    }

    pub(crate) fn synchronise(&mut self, level: usize) {
        let _ = self.trail.synchronise(level);
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.trail.len()
    }

    /// Get the propagator which generated the given reason.
    pub(crate) fn get_propagator(&self, reason_ref: ReasonRef) -> PropagatorId {
        self.trail.get(reason_ref.0 as usize).unwrap().0
    }
}

/// A reference to a reason
#[derive(Default, Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct ReasonRef(pub(crate) u32);

pub(crate) struct PropagationReasonRef<'a>(
    pub(crate) &'a [Predicate],
    pub(crate) &'a Option<LinearLessOrEqual>,
);

impl<'a> From<&'a [Predicate]> for PropagationReasonRef<'a> {
    fn from(value: &'a [Predicate]) -> Self {
        PropagationReasonRef(value, &None)
    }
}

/// A reason for CP propagator to make a change
#[derive(Debug)]
pub(crate) enum Reason {
    /// An eager reason contains the propositional conjunction with the reason, without the
    ///   propagated predicate.
    Eager(PropagationReason),
    /// A lazy reason, which is computed on-demand rather than up-front. This is also referred to
    /// as a 'backward' reason.
    ///
    /// A lazy reason contains a payload that propagators can use to identify what type of
    /// propagation the reason is for. The payload should be enough for the propagator to construct
    /// an explanation based on its internal state.
    DynamicLazy(u64),
}

impl Reason {
    pub(crate) fn compute<'a>(
        &'a self,
        context: ExplanationContext<'_>,
        propagator_id: PropagatorId,
        propagators: &'a mut PropagatorStore,
    ) -> PropagationReasonRef<'a> {
        match self {
            // We do not replace the reason with an eager explanation for dynamic lazy explanations.
            //
            // Benchmarking will have to show whether this should change or not.
            Reason::DynamicLazy(code) => {
                propagators[propagator_id].lazy_explanation(*code, context)
            }
            Reason::Eager(PropagationReason(conjunction, constraint)) => {
                PropagationReasonRef(conjunction.as_slice(), constraint)
            }
        }
    }
}

impl From<PropositionalConjunction> for Reason {
    fn from(value: PropositionalConjunction) -> Self {
        Reason::Eager(PropagationReason(value, None))
    }
}

impl From<(PropositionalConjunction, LinearLessOrEqual)> for Reason {
    fn from(value: (PropositionalConjunction, LinearLessOrEqual)) -> Self {
        Reason::Eager(PropagationReason(value.0, Some(value.1)))
    }
}

impl From<(PropositionalConjunction, Option<LinearLessOrEqual>)> for Reason {
    fn from(value: (PropositionalConjunction, Option<LinearLessOrEqual>)) -> Self {
        Reason::Eager(PropagationReason(value.0, value.1))
    }
}

impl From<PropagationReason> for Reason {
    fn from(value: PropagationReason) -> Self {
        Reason::Eager(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conjunction;
    use crate::engine::variables::DomainId;
    use crate::engine::Assignments;

    #[test]
    fn computing_an_eager_reason_returns_a_reference_to_the_conjunction() {
        let integers = Assignments::default();

        let x = DomainId::new(0);
        let y = DomainId::new(1);

        let conjunction = conjunction!([x == 1] & [y == 2]);
        let reason = Reason::Eager(conjunction.clone().into());

        assert_eq!(
            conjunction.as_slice(),
            reason
                .compute(
                    ExplanationContext::from(&integers),
                    PropagatorId(0),
                    &mut PropagatorStore::default()
                )
                .0
        );
    }

    #[test]
    fn pushing_a_reason_gives_a_reason_ref_that_can_be_computed() {
        let mut reason_store = ReasonStore::default();
        let integers = Assignments::default();

        let x = DomainId::new(0);
        let y = DomainId::new(1);

        let conjunction = conjunction!([x == 1] & [y == 2]);
        let reason_ref =
            reason_store.push(PropagatorId(0), Reason::Eager(conjunction.clone().into()));

        assert_eq!(ReasonRef(0), reason_ref);

        assert_eq!(
            Some(conjunction.as_slice()),
            reason_store
                .get_or_compute(
                    reason_ref,
                    ExplanationContext::from(&integers),
                    &mut PropagatorStore::default()
                )
                .map(|v| v.0)
        );
    }
}
