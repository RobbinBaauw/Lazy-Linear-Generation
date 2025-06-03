use std::error::Error;
use std::fmt::Debug;
use std::fs;
use std::fs::OpenOptions;
use std::io::stdout;
use std::io::Write;
mod flatzinc;

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::RwLock;
use std::time::Duration;

use clap::Parser;
use itertools::Itertools;
use log::warn;
use log::LevelFilter;
use pumpkin_solver::options::ConflictResolver::LLG;
use pumpkin_solver::options::ConflictResolver::UIP;
use pumpkin_solver::options::ConflictingCheckBeforeCutStrategy;
use pumpkin_solver::options::CumulativeOptions;
use pumpkin_solver::options::LearningOptions;
use pumpkin_solver::options::RestartOptions;
use pumpkin_solver::options::SolverOptions;
use pumpkin_solver::proof::ProofLog;
use pumpkin_solver::statistics::analysis_log::AnalysisLog;
use pumpkin_solver::statistics::configure_statistic_logging;
use pumpkin_solver::Solver;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::Serialize;
use serde_json::from_str;

use crate::flatzinc::AdditionalExplanation;
use crate::flatzinc::FlatZincOptions;

#[derive(Debug, Parser, Serialize)]
#[command(arg_required_else_help = true)]
struct Args {
    #[clap(verbatim_doc_comment)]
    instance_path: PathBuf,

    // Housekeeping args
    #[arg(long = "verbose")]
    verbose: bool,

    #[arg(long = "log-to-files")]
    log_to_files: bool,

    #[arg(long = "time-limit")]
    time_limit: Option<u64>,

    // Solver options
    #[arg(
        long = "learning-max-num-clauses",
        default_value_t = 4000,
        verbatim_doc_comment
    )]
    learning_max_num_clauses: usize,

    // LLG options
    #[arg(long = "use-llg")]
    use_llg: bool,

    #[arg(long = "llg-skip-nogood-learning")]
    llg_skip_nogood_learning: bool,

    #[arg(long = "llg-conflicting-check-before-cut", value_enum, default_value_t)]
    llg_conflicting_check_before_cut: ConflictingCheckBeforeCutStrategy,

    #[arg(long = "llg-skip-conflicting-check-after-cut")]
    llg_skip_conflicting_check_after_cut: bool,

    #[arg(long = "llg-skip-early-backjump-check")]
    llg_skip_early_backjump_check: bool,

    #[arg(long = "llg-continue-on-nogoods")]
    llg_continue_on_nogoods: bool,

    #[arg(long = "llg-only-learn-nogoods")]
    llg_only_learn_nogoods: bool,

    #[arg(long = "llg-skip-high-slack")]
    llg_skip_high_slack: bool,

    #[arg(long = "llg-clause-to-inequality")]
    llg_clause_to_inequality: bool,

    #[arg(long = "additional-explanation")]
    is_additional_explanation: bool,

    // Logging options
    #[arg(long = "all-solutions")]
    all_solutions: bool,

    #[arg(long = "analysis-log")]
    analysis_log: bool,
}

fn open_file(name: &str) -> Box<dyn Write + Send + Sync> {
    let f = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(name)
        .expect("Cannot open file");

    Box::new(f)
}

static OUTPUT_LOGGER: OnceLock<RwLock<Box<dyn Write + Send + Sync>>> = OnceLock::new();

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let mut general_logger = if args.log_to_files {
        open_file("run_info")
    } else {
        Box::new(stdout())
    };
    let stats_logger = if args.log_to_files {
        open_file("run_stats")
    } else {
        Box::new(stdout())
    };
    let output_logger = OUTPUT_LOGGER.get_or_init(|| {
        RwLock::from(if args.log_to_files {
            open_file("run_outputs")
        } else {
            Box::new(stdout())
        })
    });

    writeln!(&mut general_logger, "Version: 7")?;
    writeln!(&mut general_logger, "Git hash: {}", env!("GIT_HASH"))?;

    let args_json = serde_json::to_string(&args)?;
    writeln!(&mut general_logger, "Args: {}", args_json)?;

    // Configure logging
    configure_statistic_logging("$stat$", None, None, Some(stats_logger));

    let level_filter = if args.verbose {
        LevelFilter::Debug
    } else {
        LevelFilter::Warn
    };

    #[cfg(feature = "mult-exact-bounds")]
    warn!("Enabled exact-bound explanation for multiplication!");

    env_logger::Builder::new()
        .format(move |buf, record| writeln!(buf, "{}", record.args()))
        .filter_level(level_filter)
        .target(env_logger::Target::Stdout)
        .init();

    if pumpkin_solver::asserts::PUMPKIN_ASSERT_LEVEL_DEFINITION
        >= pumpkin_solver::asserts::PUMPKIN_ASSERT_MODERATE
    {
        warn!("Potential performance degradation: the Pumpkin assert level is set to {}, meaning many debug asserts are active which may result in performance degradation.", pumpkin_solver::asserts::PUMPKIN_ASSERT_LEVEL_DEFINITION);
    };

    let mut learning_options = LearningOptions::default();
    learning_options.limit_num_high_lbd_nogoods = args.learning_max_num_clauses;

    learning_options.llg_skip_nogood_learning = args.llg_skip_nogood_learning;
    learning_options.llg_conflicting_check_before_cut = args.llg_conflicting_check_before_cut;
    learning_options.llg_skip_conflicting_check_after_cut =
        args.llg_skip_conflicting_check_after_cut;
    learning_options.llg_skip_early_backjump_check = args.llg_skip_early_backjump_check;
    learning_options.llg_continue_on_nogoods = args.llg_continue_on_nogoods;
    learning_options.llg_only_learn_nogoods = args.llg_only_learn_nogoods;
    learning_options.llg_clause_to_inequality = args.llg_clause_to_inequality;
    learning_options.llg_skip_high_slack = args.llg_skip_high_slack;

    let (instance_path, additional_explanation) = if args.is_additional_explanation {
        let instance_path = args.instance_path.to_str().expect("Invalid path");
        let content = fs::read_to_string(instance_path).expect("Can't read file");
        let content_lines = content.lines().collect_vec();

        let act_path = content_lines[0].to_string();
        let additional_explanation: AdditionalExplanation =
            from_str(content_lines[1]).expect("Additional constraints should be valid JSON");

        (act_path, Some(additional_explanation))
    } else {
        (
            args.instance_path
                .to_str()
                .expect("Invalid path")
                .to_string(),
            None,
        )
    };

    let solver_options = SolverOptions {
        restart_options: RestartOptions::default(),
        learning_clause_minimisation: true,
        random_generator: SmallRng::seed_from_u64(42),
        proof_log: ProofLog::default(),
        conflict_resolver: if args.use_llg { LLG } else { UIP },
        learning_options,
        analysis_log: if args.analysis_log {
            let writer: Box<dyn Write + Send + Sync> = if args.log_to_files {
                open_file("analysis_log")
            } else {
                Box::new(stdout())
            };

            Some(AnalysisLog::new(writer))
        } else {
            None
        },
    };

    let time_limit = args.time_limit.map(Duration::from_millis);

    let mut solver = flatzinc::solve(
        Solver::with_options(solver_options),
        instance_path,
        time_limit,
        FlatZincOptions {
            free_search: false,
            all_solutions: args.all_solutions,
            cumulative_options: CumulativeOptions::default(),
        },
        additional_explanation,
        output_logger,
    )?;

    if let Some(analysis_log) = solver.get_satisfaction_solver_mut().get_analysis_lot_mut() {
        analysis_log.flush();
    }

    Ok(())
}
