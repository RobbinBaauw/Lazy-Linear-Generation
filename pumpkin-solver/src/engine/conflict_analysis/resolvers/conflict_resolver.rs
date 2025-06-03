use std::fmt::Debug;

use crate::engine::conflict_analysis::ConflictAnalysisContext;
use crate::engine::conflict_analysis::ConflictResolveResult;

pub(crate) trait ConflictResolver: Debug {
    fn resolve_conflict(
        &mut self,
        context: &mut ConflictAnalysisContext,
    ) -> Option<ConflictResolveResult>;

    #[allow(clippy::result_unit_err, reason = "unknown, this should be refactored")]
    fn process(
        &mut self,
        context: &mut ConflictAnalysisContext,
        learned_nogood: &Option<ConflictResolveResult>,
    ) -> Result<(), ()>;
}
