use crate::basic_types::linear_less_or_equal::LinearLessOrEqual;
use crate::predicates::Predicate;

#[derive(Clone, Debug)]
pub(crate) struct LearnedConstraint {
    pub(crate) constraint: LinearLessOrEqual,
    pub(crate) backjump_level: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct LearnedNogood {
    pub(crate) predicates: Vec<Predicate>,
    pub(crate) backjump_level: usize,
}

#[derive(Clone, Debug)]
pub(crate) enum ConflictResolveResult {
    Nogood(LearnedNogood),
    Constraint(LearnedConstraint),
}
