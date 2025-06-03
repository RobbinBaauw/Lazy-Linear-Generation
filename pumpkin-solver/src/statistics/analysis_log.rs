use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::io::Write;

use itertools::Itertools;

use crate::basic_types::linear_less_or_equal::LinearLessOrEqual;
use crate::basic_types::HashMap;
use crate::basic_types::PropositionalConjunction;
use crate::engine::conflict_analysis::FallbackReason;
use crate::engine::propagation::PropagatorId;
use crate::engine::Assignments;
use crate::propagators::nogoods::NogoodId;

// Limit file size to 10GB uncompressed
const MAX_BYTES_WRITTEN: u64 = 10_000_000_000;

#[derive(Debug, Clone)]
pub(crate) struct ExplanationStats {
    pub(crate) explanation_id: Option<u32>,
    pub(crate) nr_vars_lhs: usize,
    pub(crate) slack: i64,
}

impl LinearLessOrEqual {
    pub(crate) fn to_explanation_stats(&self, assignments: &Assignments) -> ExplanationStats {
        let slack = self.slack(assignments, assignments.num_trail_entries() - 1, true);
        let nr_vars_lhs = self.lhs.0.len();

        ExplanationStats {
            explanation_id: self.explanation_id,
            nr_vars_lhs,
            slack,
        }
    }
}

impl Display for ExplanationStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "|{}|{}|{}",
            self.explanation_id.unwrap_or(0),
            self.nr_vars_lhs,
            self.slack
        )
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LearnedInequalityStats {
    pub(crate) backtrack_distance: usize,
    pub(crate) slack: i64,
    pub(crate) used_explanations: Vec<ExplanationStats>,
}

#[derive(Debug, Clone)]
pub(crate) struct LearnedNogoodStats {
    pub(crate) backtrack_distance: usize,
    pub(crate) lbd: u32,
}

#[derive(Debug, Clone, Default)]
struct LearnedPropagationErrorStats {
    propagations_at_conflicts: HashMap<u32, u32>,
    total_propagations: u32,
    matched_propagations: u32,

    errors_at_conflicts: HashMap<u32, u32>,
    total_errors: u32,
    matched_errors: u32,
}

#[derive(Debug, Clone)]
enum AnalysisLogItem {
    ConflictAnalysisFailed {
        fallback_id: LearnedItemId,
        fallback_reason: FallbackReason,
        used_explanations: Vec<ExplanationStats>,
    },
    ConflictAnalysisSuccess {
        learned_id: LearnedItemId,
        inequality_stats: LearnedInequalityStats,
        nogood_stats: LearnedNogoodStats,
    },
    LearnedPropagations {
        learned_id: LearnedItemId,
        stats: LearnedPropagationErrorStats,
    },
}

#[derive(Debug, Clone, Eq, PartialEq, Copy, Hash)]
pub(crate) enum LearnedItemId {
    Inequality(PropagatorId),
    Nogood(NogoodId),
    PermanentNogood,
}

impl Display for LearnedItemId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            LearnedItemId::Inequality(prop_id) => write!(f, "P{}", prop_id.0),
            LearnedItemId::Nogood(nogood_id) => write!(f, "N{}", nogood_id.id),
            LearnedItemId::PermanentNogood => write!(f, "NP"),
        }
    }
}

impl Display for AnalysisLogItem {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match &self {
            AnalysisLogItem::ConflictAnalysisFailed {
                fallback_id,
                fallback_reason,
                used_explanations,
            } => {
                write!(f, "F|{fallback_id}|{}", *fallback_reason as u8)?;

                used_explanations
                    .iter()
                    .try_for_each(|expl_stats| write!(f, "{expl_stats}"))?;

                writeln!(f, "")
            }
            AnalysisLogItem::ConflictAnalysisSuccess {
                learned_id,
                inequality_stats,
                nogood_stats,
            } => {
                write!(
                    f,
                    "S|{learned_id}|{}|{}|{}|{}",
                    inequality_stats.backtrack_distance,
                    inequality_stats.slack,
                    nogood_stats.backtrack_distance,
                    nogood_stats.lbd
                )?;

                inequality_stats
                    .used_explanations
                    .iter()
                    .try_for_each(|expl_stats| write!(f, "{expl_stats}"))?;

                writeln!(f, "")
            }
            AnalysisLogItem::LearnedPropagations { learned_id, stats } => {
                let to_str = |map: &HashMap<u32, u32>| {
                    map.iter()
                        .map(|(confl, count)| {
                            if *count == 1 {
                                format!("{confl}")
                            } else {
                                format!("{confl}-{count}")
                            }
                        })
                        .join(" ")
                };

                let prop_at_confl_str = to_str(&stats.propagations_at_conflicts);
                let err_at_confl_str = to_str(&stats.errors_at_conflicts);

                writeln!(
                    f,
                    "P|{learned_id}|{}|{}|{}|{}|{prop_at_confl_str}|{err_at_confl_str}",
                    stats.total_propagations,
                    stats.matched_propagations,
                    stats.total_errors,
                    stats.matched_errors
                )
            }
        }
    }
}

#[derive(Debug, Clone)]
enum QueuedAnalysisResult {
    Success {
        nogood: PropositionalConjunction,
        nogood_stats: LearnedNogoodStats,
        inequality: LinearLessOrEqual,
        inequality_stats: LearnedInequalityStats,
    },
    Fallback {
        fallback_reason: FallbackReason,
        used_explanations: Vec<ExplanationStats>,
    },
}

pub struct AnalysisLog {
    curr_num_conflicts: u64,

    learned_items: HashMap<
        LearnedItemId,
        (
            Option<(PropositionalConjunction, LinearLessOrEqual)>,
            LearnedPropagationErrorStats,
        ),
    >,

    queued_analysis_result: Option<QueuedAnalysisResult>,

    bytes_written: u64,
    writer: Box<dyn Write + Send + Sync>,
}

impl AnalysisLog {
    pub fn new(writer: Box<dyn Write + Send + Sync>) -> Self {
        Self {
            curr_num_conflicts: 0,

            learned_items: Default::default(),
            queued_analysis_result: Default::default(),

            bytes_written: 0,
            writer,
        }
    }

    pub(crate) fn has_reached_file_limit(&self) -> bool {
        let has_exceeded_limit = self.bytes_written > MAX_BYTES_WRITTEN;
        if has_exceeded_limit {
            eprintln!("Exec log: limit of {MAX_BYTES_WRITTEN} reached, stop logging");
        }
        has_exceeded_limit
    }

    pub(crate) fn update_conflicts(&mut self, num_conflicts: u64) {
        self.curr_num_conflicts = num_conflicts;
    }

    fn log_item(&mut self, item: AnalysisLogItem) {
        let item_str = format!("{}|{item}", self.curr_num_conflicts);

        self.bytes_written += item_str.len() as u64;

        self.writer
            .write_all(item_str.as_bytes())
            .expect("Should be able to write!");
    }
}

impl AnalysisLog {
    pub fn flush(&mut self) {
        let items = self.learned_items.drain().collect_vec();
        items.into_iter().for_each(|(learned_id, (_, stats))| {
            self.log_item(AnalysisLogItem::LearnedPropagations { learned_id, stats })
        });
        self.writer.flush().unwrap();
    }

    pub(crate) fn log_analysis_success(
        &mut self,
        nogood: PropositionalConjunction,
        nogood_stats: LearnedNogoodStats,
        inequality: LinearLessOrEqual,
        inequality_stats: LearnedInequalityStats,
    ) {
        assert!(self.queued_analysis_result.is_none());
        self.queued_analysis_result = Some(QueuedAnalysisResult::Success {
            nogood,
            nogood_stats,
            inequality,
            inequality_stats,
        });
    }

    pub(crate) fn log_analysis_fallback(
        &mut self,
        fallback_reason: FallbackReason,
        used_explanations: Vec<ExplanationStats>,
    ) {
        assert!(self.queued_analysis_result.is_none());
        self.queued_analysis_result = Some(QueuedAnalysisResult::Fallback {
            fallback_reason,
            used_explanations,
        });
    }

    pub(crate) fn log_new_learned(&mut self, learned_id: LearnedItemId) {
        // In this case, we re-use a nogood that we already used before
        // Then, we can write out the stats for that item
        let _ = self.learned_items.remove(&learned_id).map(|(_, stats)| {
            self.log_item(AnalysisLogItem::LearnedPropagations { learned_id, stats })
        });

        match self.queued_analysis_result.take().unwrap() {
            QueuedAnalysisResult::Success {
                nogood,
                nogood_stats,
                inequality,
                inequality_stats,
            } => {
                let _ = self
                    .learned_items
                    .insert(learned_id, (Some((nogood, inequality)), Default::default()));

                self.log_item(AnalysisLogItem::ConflictAnalysisSuccess {
                    learned_id,
                    inequality_stats,
                    nogood_stats,
                });
            }
            QueuedAnalysisResult::Fallback {
                fallback_reason,
                used_explanations,
            } => {
                let _ = self
                    .learned_items
                    .insert(learned_id, (None, Default::default()));

                self.log_item(AnalysisLogItem::ConflictAnalysisFailed {
                    fallback_id: learned_id,
                    fallback_reason,
                    used_explanations,
                });
            }
        }
    }

    pub(crate) fn log_propagation(&mut self, learned_id: LearnedItemId, assignments: &Assignments) {
        let (learned_opt, stats) = self.learned_items.get_mut(&learned_id).unwrap();

        stats.total_propagations += 1;

        match learned_id {
            LearnedItemId::Inequality(_) => {
                // Only for inequalities do we log their propagation pattern
                *stats
                    .propagations_at_conflicts
                    .entry(self.curr_num_conflicts as u32)
                    .or_insert_with(Default::default) += 1;

                let (nogood, _) = learned_opt.as_ref().unwrap();

                let satisfied_preds = nogood
                    .iter()
                    .filter(|pred| assignments.is_predicate_satisfied(**pred))
                    .count();
                let propagation_matched = satisfied_preds >= nogood.len() - 1;
                stats.matched_propagations += propagation_matched as u32
            }
            LearnedItemId::Nogood(_) | LearnedItemId::PermanentNogood => {
                // If we haven't been able to learn an inequality, we can skip
                let Some((_, inequality)) = learned_opt else {
                    return;
                };

                let propagation_matched = inequality.is_propagating(
                    assignments,
                    assignments.num_trail_entries() - 1,
                    true,
                );
                stats.matched_propagations += propagation_matched as u32;
            }
        }
    }

    pub(crate) fn log_error(&mut self, learned_id: LearnedItemId, assignments: &Assignments) {
        let (learned_opt, stats) = self.learned_items.get_mut(&learned_id).unwrap();

        stats.total_errors += 1;

        match learned_id {
            LearnedItemId::Inequality(_) => {
                // Only for inequalities do we log their error pattern
                *stats
                    .errors_at_conflicts
                    .entry(self.curr_num_conflicts as u32)
                    .or_insert_with(Default::default) += 1;

                let (nogood, _) = learned_opt.as_ref().unwrap();

                let satisfied_preds = nogood
                    .iter()
                    .filter(|pred| assignments.is_predicate_satisfied(**pred))
                    .count();
                let error_matched = satisfied_preds == nogood.len() - 1;
                stats.matched_errors += error_matched as u32
            }
            LearnedItemId::Nogood(_) | LearnedItemId::PermanentNogood => {
                // If we haven't been able to learn an inequality, we can skip
                let Some((_, inequality)) = learned_opt else {
                    return;
                };

                let error_matched = inequality.is_conflicting(
                    assignments,
                    assignments.num_trail_entries() - 1,
                    true,
                );
                stats.matched_errors += error_matched as u32;
            }
        }
    }
}

impl Debug for AnalysisLog {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecLog")
            .field("curr_num_conflicts", &self.curr_num_conflicts)
            .field("learned_items", &self.learned_items)
            .field("queued_learned_item", &self.queued_analysis_result)
            .finish_non_exhaustive()
    }
}
