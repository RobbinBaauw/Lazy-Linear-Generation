use downcast_rs::impl_downcast;
use downcast_rs::Downcast;

use super::explanation_context::ExplanationContext;
use super::propagator_initialisation_context::PropagatorInitialisationContext;
#[cfg(doc)]
use crate::basic_types::Inconsistency;
use crate::basic_types::PropagationReason;
use crate::basic_types::PropagationStatusCP;
#[cfg(doc)]
use crate::create_statistics_struct;
use crate::engine::opaque_domain_event::OpaqueDomainEvent;
use crate::engine::propagation::local_id::LocalId;
use crate::engine::propagation::propagation_context::PropagationContext;
use crate::engine::propagation::propagation_context::PropagationContextMut;
use crate::engine::propagation::PropagatorId;
use crate::engine::reason::PropagationReasonRef;
use crate::engine::Assignments;
#[cfg(doc)]
use crate::engine::ConstraintSatisfactionSolver;
#[cfg(doc)]
use crate::pumpkin_asserts::PUMPKIN_ASSERT_ADVANCED;
#[cfg(doc)]
use crate::pumpkin_asserts::PUMPKIN_ASSERT_EXTREME;
use crate::statistics::statistic_logger::StatisticLogger;
use crate::variables::DomainId;

// We need to use this to cast from `Box<dyn Propagator>` to `NogoodPropagator`; rust inherently
// does not allow downcasting from the trait definition to its concrete type.
impl_downcast!(Propagator);

/// All propagators implement the [`Propagator`] trait, with the exception of the
/// clausal propagator. Structs implementing the trait defines the main propagator logic with
/// regards to propagation, detecting conflicts, and providing explanations.
///
/// The only required functions are [`Propagator::name`],
/// [`Propagator::initialise_at_root`], and [`Propagator::debug_propagate_from_scratch`]; all other
/// functions have default implementations. For initial development, the required functions are
/// enough, but a more mature implementation considers all functions in most cases.
///
/// See the [`crate::engine::cp::propagation`] documentation for more details.
pub(crate) trait Propagator: Downcast {
    /// Return the name of the propagator, this is a convenience method that is used for printing.
    fn name(&self) -> &str;

    /// A propagation method that is used to help debugging.
    ///
    /// This method propagates without relying on internal data structures, hence the immutable
    /// &self parameter. It is usually best to implement this propagation method in the simplest
    /// but correct way. When the assert level is set to [`PUMPKIN_ASSERT_ADVANCED`] or
    /// [`PUMPKIN_ASSERT_EXTREME`] (see [`crate::pumpkin_asserts`]) this method will be called
    /// to double check the reasons for failures and propagations that have been reported by
    /// this propagator.
    ///
    /// Propagators are not required to propagate until a fixed point. It will be called again by
    /// the solver until no further propagations happen.
    fn debug_propagate_from_scratch(
        &self,
        context: &mut PropagationContextMut,
    ) -> PropagationStatusCP;

    /// Propagate method that will be called during search (e.g. in
    /// [`ConstraintSatisfactionSolver::solve`]).
    ///
    /// This method extends the current partial
    /// assignments with inferred domain changes found by the
    /// [`Propagator`]. In case no conflict has been detected it should return
    /// [`Result::Ok`], otherwise it should return a [`Result::Err`] with an [`Inconsistency`] which
    /// contains the reason for the failure; either because a propagation caused an
    /// an empty domain ([`Inconsistency::EmptyDomain`]) or because the logic of the propagator
    /// found the current state to be inconsistent ([`Inconsistency::Conflict`]).
    ///
    /// Note that the failure (explanation) is given as a conjunction of predicates that lead to the
    /// failure
    ///
    /// Propagators are not required to propagate until a fixed point. It will be called
    /// again by the solver until no further propagations happen.
    ///
    /// By default, this function calls [`Propagator::debug_propagate_from_scratch`].
    fn propagate(&mut self, context: &mut PropagationContextMut) -> PropagationStatusCP {
        self.debug_propagate_from_scratch(context)
    }

    /// Called when an event happens to one of the variables the propagator is subscribed to. It
    /// indicates whether the provided event should cause the propagator to be enqueued.
    ///
    /// This can be used to incrementally maintain data structures or perform propagations, and
    /// should only be used for computationally cheap logic. Expensive computation should be
    /// performed in the [`Propagator::propagate()`] method.
    ///
    /// By default the propagator is always enqueued for every event. Not all propagators will
    /// benefit from implementing this, so it is not required to do so.
    ///
    /// Note that the variables and events to which the propagator is subscribed to are determined
    /// upon propagator initialisation via [`Propagator::initialise_at_root`] by calling
    /// [`PropagatorInitialisationContext::register()`].
    fn notify(
        &mut self,
        _context: PropagationContext,
        _local_id: LocalId,
        _event: OpaqueDomainEvent,
    ) -> EnqueueDecision {
        EnqueueDecision::Enqueue
    }

    /// Called when an event happens to one of the variables the propagator is subscribed to. This
    /// method is called during backtrack when the domain of a variable has been undone.
    ///
    /// This can be used to incrementally maintain data structures or perform propagations, and
    /// should only be used for computationally cheap logic. Expensive computation should be
    /// performed in the [`Propagator::propagate`] method.
    ///
    /// By default the propagator does nothing when this method is called. Not all propagators will
    /// benefit from implementing this, so it is not required to do so.
    ///
    /// Note that the variables and events to which the propagator is subscribed to are determined
    /// upon propagator initialisation via [`Propagator::initialise_at_root`] by calling
    /// [`PropagatorInitialisationContext::register()`].
    fn notify_backtrack(
        &mut self,
        _context: PropagationContext,
        _local_id: LocalId,
        _event: OpaqueDomainEvent,
    ) -> EnqueueDecision {
        EnqueueDecision::Skip
    }

    /// Called each time the [`ConstraintSatisfactionSolver`] backtracks, the propagator can then
    /// update its internal data structures given the new variable domains.
    ///
    /// By default this function does nothing.
    fn synchronise(&mut self, _context: PropagationContext) -> EnqueueDecision {
        EnqueueDecision::Skip
    }

    /// Returns the priority of the propagator represented as an integer. Lower values mean higher
    /// priority and the priority determines the order in which propagators will be asked to
    /// propagate. It is custom for simpler propagators to have lower priority values.
    ///
    /// By default the priority is set to 3. It is expected that propagator implementations would
    /// set this value to some appropriate value.
    fn priority(&self) -> u32 {
        // setting an arbitrary priority by default
        3
    }

    /// Initialises the propagator without performing propagation. This method is called only once
    /// by the [`ConstraintSatisfactionSolver`] when the propagator is added using
    /// [`ConstraintSatisfactionSolver::add_propagator`].
    ///
    /// The method can be used to detect root-level inconsistencies and to register variables used
    /// for notifications (see [`Propagator::notify`]) by calling
    /// [`PropagatorInitialisationContext::register`].
    ///
    /// The solver will call this before any call to [`Propagator::propagate`] is made.
    fn initialise_at_root(
        &mut self,
        _: &mut PropagatorInitialisationContext,
    ) -> Result<(), PropagationReason>;

    fn initialise_at_non_root(
        &mut self,
        ctx: &mut PropagatorInitialisationContext,
    ) -> Result<(), PropagationReason> {
        self.initialise_at_root(ctx)
    }

    /// A check whether this propagator can detect an inconsistency.
    ///
    /// By implementing this function, if the propagator is reified, it can propagate the
    /// reification literal based on the detected inconsistency. Yet, an implementation is not
    /// needed for correctness, as [`Propagator::propagate`] should still check for
    /// inconsistency as well.
    fn detect_inconsistency(&self, _context: PropagationContext) -> Option<PropagationReason> {
        None
    }

    fn detect_inconsistency_mut(
        &self,
        context: &mut PropagationContextMut,
    ) -> Option<PropagationReason> {
        self.detect_inconsistency(context.as_readonly())
    }

    /// Hook which is called when a propagation was done with a lazy reason.
    ///
    /// The code which was attached to the propagation through [`Reason::DynamicLazy`] is given, as
    /// well as a context object which defines what can be inspected from the solver to build the
    /// explanation.
    fn lazy_explanation(
        &mut self,
        _code: u64,
        _context: ExplanationContext,
    ) -> PropagationReasonRef {
        panic!(
            "{}",
            format!(
                "Propagator {} does not support lazy explanations.",
                self.name()
            )
        );
    }

    /// Logs statistics of the propagator using the provided [`StatisticLogger`].
    ///
    /// It is recommended to create a struct through the [`create_statistics_struct!`] macro!
    fn log_statistics(&self, _statistic_logger: StatisticLogger) {}

    /// Creates unlinked auxiliaries, used for verifying explanations
    fn create_unlinked_auxiliaries(
        &self,
        _assignments: &mut Assignments,
        _propagator_id: PropagatorId,
    ) -> Option<Vec<DomainId>> {
        None
    }
}

/// Indicator of what to do when a propagator is notified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EnqueueDecision {
    /// The propagator should be enqueued.
    Enqueue,
    /// The propagator should not be enqueued.
    Skip,
}
