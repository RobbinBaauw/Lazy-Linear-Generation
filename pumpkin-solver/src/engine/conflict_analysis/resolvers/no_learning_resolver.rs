use super::ConflictResolver;
use crate::engine::conflict_analysis::ConflictAnalysisContext;
use crate::engine::conflict_analysis::ConflictResolveResult;
use crate::pumpkin_assert_simple;

/// Resolve conflicts by backtracking one decision level trying the opposite of the last decision.
#[derive(Default, Debug, Clone, Copy)]
pub(crate) struct NoLearningResolver;

impl ConflictResolver for NoLearningResolver {
    fn resolve_conflict(
        &mut self,
        _context: &mut ConflictAnalysisContext,
    ) -> Option<ConflictResolveResult> {
        None
    }

    fn process(
        &mut self,
        context: &mut ConflictAnalysisContext,
        resolve_result: &Option<ConflictResolveResult>,
    ) -> Result<(), ()> {
        pumpkin_assert_simple!(resolve_result.is_none());

        if let Some(last_decision) = context.find_last_decision() {
            context.backtrack(context.assignments.get_decision_level() - 1);
            context.enqueue_propagated_predicate(!last_decision);
            Ok(())
        } else {
            Err(())
        }
    }
}
