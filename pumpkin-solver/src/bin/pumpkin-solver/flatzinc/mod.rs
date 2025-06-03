mod ast;
mod compiler;
pub(crate) mod error;
mod instance;
mod parser;

use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::sync::RwLock;
use std::time::Duration;

use pumpkin_solver::basic_types::LinearLessOrEqual;
use pumpkin_solver::branching::branchers::alternating_brancher::AlternatingBrancher;
use pumpkin_solver::branching::branchers::alternating_brancher::AlternatingStrategy;
use pumpkin_solver::branching::branchers::dynamic_brancher::DynamicBrancher;
use pumpkin_solver::constraints;
#[cfg(doc)]
use pumpkin_solver::constraints::cumulative;
use pumpkin_solver::constraints::Constraint;
use pumpkin_solver::constraints::NegatableConstraint;
use pumpkin_solver::options::CumulativeOptions;
use pumpkin_solver::propagators::LinearInequalityLiteralPropagator;
use pumpkin_solver::results::solution_iterator::IteratedSolution;
use pumpkin_solver::results::OptimisationResult;
use pumpkin_solver::results::ProblemSolution;
use pumpkin_solver::results::SatisfactionResult;
use pumpkin_solver::results::Solution;
use pumpkin_solver::termination::Combinator;
use pumpkin_solver::termination::OsSignal;
use pumpkin_solver::termination::TimeBudget;
use pumpkin_solver::variables::AffineView;
use pumpkin_solver::variables::DomainId;
use pumpkin_solver::variables::IntegerVariable;
use pumpkin_solver::variables::TransformableVariable;
use pumpkin_solver::Solver;
use serde::Deserialize;

use self::instance::FlatZincInstance;
use self::instance::FlatzincObjective;
use self::instance::Output;
use crate::flatzinc::error::FlatZincError;

const MSG_UNKNOWN: &str = "=====UNKNOWN=====";
const MSG_UNSATISFIABLE: &str = "=====UNSATISFIABLE=====";

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct FlatZincOptions {
    /// If `true`, the solver will not strictly keep to the search annotations in the flatzinc.
    pub free_search: bool,

    /// For satisfaction problems, print all solutions. For optimisation problems, this instructs
    /// the solver to print intermediate solutions.
    pub all_solutions: bool,

    /// Options used for the cumulative constraint (see [`cumulative`]).
    pub cumulative_options: CumulativeOptions,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct JSONLinearLessOrEquals {
    lhs: Vec<(u32, i32)>,
    rhs: i32,
}

impl JSONLinearLessOrEquals {
    fn to_affine_vec(&self) -> Vec<AffineView<DomainId>> {
        self.lhs
            .iter()
            .map(|(var_id, var_scale)| DomainId::new(*var_id).scaled(*var_scale))
            .collect()
    }

    fn remap(&mut self, aux_remap: &HashMap<u32, DomainId>) {
        self.lhs.iter_mut().for_each(|(var_id, _)| {
            let remapped_key = aux_remap.get(var_id);
            if let Some(remapped_key) = remapped_key {
                *var_id = remapped_key.id;
            }
        });
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct AdditionalExplanation {
    explanation: JSONLinearLessOrEquals,
    linked_auxes: Vec<(u32, JSONLinearLessOrEquals)>,
    unlinked_auxes: Vec<(u32, Vec<u32>, Vec<JSONLinearLessOrEquals>)>,
}

pub(crate) fn solve(
    mut solver: Solver,
    instance: impl AsRef<Path>,
    time_limit: Option<Duration>,
    options: FlatZincOptions,
    additional_explanation: Option<AdditionalExplanation>,
    output_writer: &'static RwLock<Box<dyn Write + Send + Sync>>,
) -> Result<Solver, FlatZincError> {
    let instance = File::open(instance)?;

    let unlock_writer = || output_writer.write().unwrap();

    let mut termination = Combinator::new(
        OsSignal::install(),
        time_limit.map(TimeBudget::starting_now),
    );

    let instance = parse_and_compile(&mut solver, instance, options)?;

    if let Some(mut additional_explanation) = additional_explanation {
        let mut aux_remap = HashMap::<u32, DomainId>::new();

        for (aux_id, _) in &additional_explanation.linked_auxes {
            let new_aux_id = solver.new_bounded_integer(0, 1);
            let _ = aux_remap.insert(*aux_id, new_aux_id);
        }

        for (prop_id, aux_ids, _) in &additional_explanation.unlinked_auxes {
            let auxes = &solver
                .get_satisfaction_solver_mut()
                .create_unlinked_auxiliaries(*prop_id)
                .unwrap();

            aux_ids.iter().zip(auxes).for_each(|(aux, new_aux)| {
                let _ = aux_remap.insert(*aux, *new_aux);
            });
        }

        for (aux_id, aux_constr) in &mut additional_explanation.linked_auxes {
            aux_constr.remap(&aux_remap);

            LinearInequalityLiteralPropagator::new(
                LinearLessOrEqual::new(aux_constr.to_affine_vec(), aux_constr.rhs),
                aux_remap[&aux_id],
            )
            .post(&mut solver, None)
            .unwrap();
        }

        additional_explanation.explanation.remap(&aux_remap);

        let ok_at_root = constraints::less_than_or_equals(
            additional_explanation.explanation.to_affine_vec(),
            additional_explanation.explanation.rhs,
        )
        .negation()
        .post(&mut solver, None)
        .is_ok();

        if !ok_at_root {
            println!("UNSAT at root!!");
            return Ok(solver);
        }
    }

    let outputs = instance.outputs.clone();

    solver.with_solution_callback(move |solution_callback_arguments| {
        if options.all_solutions {
            print_solution_from_solver(
                solution_callback_arguments.solution,
                &outputs,
                &mut unlock_writer(),
            );
        }
    });

    let mut brancher = if options.free_search {
        // The free search flag is active, we just use the default brancher
        DynamicBrancher::new(vec![Box::new(AlternatingBrancher::new(
            &solver,
            instance.search.expect("Expected a search to be defined"),
            AlternatingStrategy::SwitchToDefaultAfterFirstSolution,
        ))])
    } else {
        instance.search.expect("Expected a search to be defined")
    };

    let value = if let Some(objective_function) = &instance.objective_function {
        let result = match objective_function {
            FlatzincObjective::Maximize(domain_id) => {
                solver.maximise(&mut brancher, &mut termination, *domain_id)
            }
            FlatzincObjective::Minimize(domain_id) => {
                solver.minimise(&mut brancher, &mut termination, *domain_id)
            }
        };

        match result {
            OptimisationResult::Optimal(optimal_solution) => {
                let optimal_objective_value =
                    optimal_solution.get_integer_value(*objective_function.get_domain());

                if !options.all_solutions {
                    print_solution_from_solver(
                        &optimal_solution,
                        &instance.outputs,
                        &mut unlock_writer(),
                    )
                }
                writeln!(&mut unlock_writer(), "==========")?;

                Some(optimal_objective_value)
            }
            OptimisationResult::Satisfiable(solution) => {
                let best_found_objective_value =
                    solution.get_integer_value(*objective_function.get_domain());

                print_solution_from_solver(&solution, &instance.outputs, &mut unlock_writer());
                writeln!(&mut unlock_writer(), "==========")?;

                Some(best_found_objective_value)
            }
            OptimisationResult::Unsatisfiable => {
                writeln!(&mut unlock_writer(), "{MSG_UNSATISFIABLE}")?;
                None
            }
            OptimisationResult::Unknown => {
                writeln!(&mut unlock_writer(), "{MSG_UNKNOWN}")?;
                None
            }
        }
    } else {
        if options.all_solutions {
            let output_domains = instance
                .outputs
                .iter()
                .flat_map(|output| match output {
                    Output::Bool(bool) => vec![bool.variable.domain_id()],
                    Output::Int(int) => vec![int.variable],
                    Output::ArrayOfBool(bool_arr) => bool_arr
                        .contents
                        .iter()
                        .map(|bool| bool.domain_id())
                        .collect(),
                    Output::ArrayOfInt(int_arr) => {
                        int_arr.contents.iter().map(|int| *int).collect()
                    }
                })
                .collect::<HashSet<DomainId>>();
            let var_in_outputs = |var| output_domains.contains(&var);

            let mut solution_iterator = solver.get_solution_iterator(
                &mut brancher,
                &mut termination,
                Some(&var_in_outputs),
            );
            loop {
                match solution_iterator.next_solution() {
                    IteratedSolution::Solution(_) => {}
                    IteratedSolution::Finished => {
                        writeln!(&mut unlock_writer(), "==========")?;
                        break;
                    }
                    IteratedSolution::Unknown => {
                        break;
                    }
                    IteratedSolution::Unsatisfiable => {
                        writeln!(&mut unlock_writer(), "{MSG_UNSATISFIABLE}")?;
                        break;
                    }
                }
            }
        } else {
            match solver.satisfy(&mut brancher, &mut termination) {
                SatisfactionResult::Satisfiable(solution) => {
                    print_solution_from_solver(&solution, &instance.outputs, &mut unlock_writer());
                    writeln!(&mut unlock_writer(), "==========")?;
                }
                SatisfactionResult::Unsatisfiable => {
                    writeln!(&mut unlock_writer(), "{MSG_UNSATISFIABLE}")?;
                }
                SatisfactionResult::Unknown => {
                    writeln!(&mut unlock_writer(), "{MSG_UNKNOWN}")?;
                }
            }
        }

        None
    };

    if let Some(value) = value {
        solver.log_statistics_with_objective(value as i64)
    } else {
        solver.log_statistics()
    }

    Ok(solver)
}

fn parse_and_compile(
    solver: &mut Solver,
    instance: impl Read,
    options: FlatZincOptions,
) -> Result<FlatZincInstance, FlatZincError> {
    let ast = parser::parse(instance)?;
    compiler::compile(ast, solver, options)
}

/// Prints the current solution.
fn print_solution_from_solver(
    solution: &Solution,
    outputs: &[Output],
    output_writer: &mut Box<dyn Write + Send + Sync>,
) {
    for output_specification in outputs {
        match output_specification {
            Output::Bool(output) => output.print_value(
                |literal| solution.get_literal_value(*literal),
                output_writer,
            ),

            Output::Int(output) => output.print_value(
                |domain_id| solution.get_integer_value(*domain_id),
                output_writer,
            ),

            Output::ArrayOfBool(output) => output.print_value(
                |literal| solution.get_literal_value(*literal),
                output_writer,
            ),

            Output::ArrayOfInt(output) => output.print_value(
                |domain_id| solution.get_integer_value(*domain_id),
                output_writer,
            ),
        }
    }

    writeln!(output_writer, "----------").expect("Cannot write to writer");
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: The following tests rely on observing the interal state of the solver. This is not good
    // design, and these tests should be re-done.
    //
    // #[test]
    // fn single_bool_gets_compiled_to_literal() {
    //     let model = r#"
    //         var bool: SomeVar;
    //         solve satisfy;
    //     "#;

    //     let mut solver = ConstraintSatisfactionSolver::default();

    //     let starting_variables = solver
    //         .get_propositional_assignments()
    //         .num_propositional_variables();

    //     let _ =
    //         parse_and_compile(&mut solver, model.as_bytes()).expect("compilation should
    // succeed");

    //     let final_variables = solver
    //         .get_propositional_assignments()
    //         .num_propositional_variables();

    //     assert_eq!(1, final_variables - starting_variables);
    // }

    // #[test]
    // fn output_annotation_is_interpreted_on_bools() {
    //     let model = r#"
    //         var bool: SomeVar ::output_var;
    //         solve satisfy;
    //     "#;

    //     let mut solver = ConstraintSatisfactionSolver::default();

    //     let instance =
    //         parse_and_compile(&mut solver, model.as_bytes()).expect("compilation should
    // succeed");

    //     let literal = Literal::new(
    //         PropositionalVariable::new(
    //             solver
    //                 .get_propositional_assignments()
    //                 .num_propositional_variables()
    //                 - 1,
    //         ),
    //         true,
    //     );

    //     let outputs = instance.outputs().collect::<Vec<_>>();
    //     assert_eq!(1, outputs.len());

    //     let output = outputs[0].clone();
    //     assert_eq!(output, Output::bool("SomeVar".into(), literal));
    // }

    // #[test]
    // fn equivalent_bools_refer_to_the_same_literal() {
    //     let model = r#"
    //         var bool: SomeVar;
    //         var bool: OtherVar = SomeVar;
    //         solve satisfy;
    //     "#;

    //     let mut solver = ConstraintSatisfactionSolver::default();

    //     let starting_variables = solver
    //         .get_propositional_assignments()
    //         .num_propositional_variables();

    //     let _ =
    //         parse_and_compile(&mut solver, model.as_bytes()).expect("compilation should
    // succeed");

    //     let final_variables = solver
    //         .get_propositional_assignments()
    //         .num_propositional_variables();

    //     assert_eq!(1, final_variables - starting_variables);
    // }

    // #[test]
    // fn bool_equivalent_to_true_uses_builtin_true_literal() {
    //     let model = r#"
    //         var bool: SomeVar = true;
    //         solve satisfy;
    //     "#;

    //     let mut solver = ConstraintSatisfactionSolver::default();

    //     let starting_variables = solver
    //         .get_propositional_assignments()
    //         .num_propositional_variables();

    //     let _ =
    //         parse_and_compile(&mut solver, model.as_bytes()).expect("compilation should
    // succeed");

    //     let final_variables = solver
    //         .get_propositional_assignments()
    //         .num_propositional_variables();

    //     assert_eq!(0, final_variables - starting_variables);
    // }

    // #[test]
    // fn single_variable_gets_compiled_to_domain_id() {
    //     let instance = "var 1..5: SomeVar;\nsolve satisfy;";
    //     let mut solver = ConstraintSatisfactionSolver::default();

    //     let _ = parse_and_compile(&mut solver, instance.as_bytes())
    //         .expect("compilation should succeed");

    //     let domains = solver
    //         .get_integer_assignments()
    //         .get_domains()
    //         .collect::<Vec<DomainId>>();

    //     assert_eq!(1, domains.len());

    //     let domain = domains[0];
    //     assert_eq!(1, solver.get_integer_assignments().get_lower_bound(domain));
    //     assert_eq!(5, solver.get_integer_assignments().get_upper_bound(domain));
    // }

    // #[test]
    // fn equal_integer_variables_use_one_domain_id() {
    //     let instance = r#"
    //          var 1..10: SomeVar;
    //          var 0..11: OtherVar = SomeVar;
    //          solve satisfy;
    //      "#;
    //     let mut solver = ConstraintSatisfactionSolver::default();

    //     let _ = parse_and_compile(&mut solver, instance.as_bytes())
    //         .expect("compilation should succeed");

    //     let domains = solver
    //         .get_integer_assignments()
    //         .get_domains()
    //         .collect::<Vec<DomainId>>();

    //     assert_eq!(1, domains.len());

    //     let domain = domains[0];
    //     assert_eq!(1, solver.get_integer_assignments().get_lower_bound(domain));
    //     assert_eq!(10, solver.get_integer_assignments().get_upper_bound(domain));
    // }

    // #[test]
    // fn var_equal_to_constant_reuse_domain_id() {
    //     let instance = r#"
    //          var 1..10: SomeVar = 5;
    //          var 0..11: OtherVar = 5;
    //          solve satisfy;
    //      "#;
    //     let mut solver = ConstraintSatisfactionSolver::default();

    //     let _ = parse_and_compile(&mut solver, instance.as_bytes())
    //         .expect("compilation should succeed");

    //     let domains = solver
    //         .get_integer_assignments()
    //         .get_domains()
    //         .collect::<Vec<DomainId>>();

    //     assert_eq!(1, domains.len());

    //     let domain = domains[0];
    //     assert_eq!(5, solver.get_integer_assignments().get_lower_bound(domain));
    //     assert_eq!(5, solver.get_integer_assignments().get_upper_bound(domain));
    // }

    #[test]
    fn array_1d_of_boolean_variables() {
        let instance = r#"
            var bool: x1;
            var bool: x2;
            array [1..2] of var bool: xs :: output_array([1..2]) = [x1,x2];
            solve satisfy;
        "#;
        let mut solver = Solver::default();

        let instance =
            parse_and_compile(&mut solver, instance.as_bytes(), FlatZincOptions::default())
                .expect("compilation should succeed");

        let outputs = instance.outputs().collect::<Vec<_>>();
        assert_eq!(1, outputs.len());

        assert!(matches!(outputs[0], Output::ArrayOfBool(_)));
    }

    #[test]
    fn array_2d_of_boolean_variables() {
        let instance = r#"
            var bool: x1;
            var bool: x2;
            var bool: x3;
            var bool: x4;
            array [1..4] of var bool: xs :: output_array([1..2, 1..2]) = [x1,x2,x3,x4];
            solve satisfy;
        "#;
        let mut solver = Solver::default();

        let instance =
            parse_and_compile(&mut solver, instance.as_bytes(), FlatZincOptions::default())
                .expect("compilation should succeed");

        let outputs = instance.outputs().collect::<Vec<_>>();
        assert_eq!(1, outputs.len());
    }

    #[test]
    fn array_1d_of_integer_variables() {
        let instance = r#"
            var 1..10: x1;
            var 1..10: x2;
            array [1..2] of var int: xs :: output_array([1..2]) = [x1,x2];
            solve satisfy;
        "#;
        let mut solver = Solver::default();

        let instance =
            parse_and_compile(&mut solver, instance.as_bytes(), FlatZincOptions::default())
                .expect("compilation should succeed");

        let outputs = instance.outputs().collect::<Vec<_>>();
        assert_eq!(1, outputs.len());

        assert!(matches!(outputs[0], Output::ArrayOfInt(_)));
    }

    #[test]
    fn array_2d_of_integer_variables() {
        let instance = r#"
            var 1..10: x1;
            var 1..10: x2;
            var 1..10: x3;
            var 1..10: x4;
            array [1..4] of var 1..10: xs :: output_array([1..2, 1..2]) = [x1,x2,x3,x4];
            solve satisfy;
        "#;
        let mut solver = Solver::default();

        let instance =
            parse_and_compile(&mut solver, instance.as_bytes(), FlatZincOptions::default())
                .expect("compilation should succeed");

        let outputs = instance.outputs().collect::<Vec<_>>();
        assert_eq!(1, outputs.len());
    }
}
