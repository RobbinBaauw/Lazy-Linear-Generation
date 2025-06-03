#![cfg(test)]

mod helpers;

use std::collections::HashSet;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::panic;
use std::path::PathBuf;
use std::process::Command;
use std::str::from_utf8;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use indicatif::MultiProgress;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use itertools::Itertools;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use regex::Regex;

use crate::helpers::flatzinc::Solutions;

fn run_minizinc_with_ozn(
    output_file: PathBuf,
    ozn_file: PathBuf,
) -> Result<HashSet<String>, Box<dyn Error>> {
    let output = Command::new("minizinc")
        .arg("--ozn-file")
        .arg(ozn_file.into_os_string())
        .stdin(File::open(output_file)?)
        .output()?;

    Ok(from_utf8(&output.stdout)?
        .trim()
        .trim_end_matches("==========")
        .split("\n----------\n")
        .map(|str| String::from(str.trim()))
        .collect())
}

fn process_pumpkin_outputs_dir(
    pumpkin_outputs_dir_path: &PathBuf,
    problem_set_path: &PathBuf,
    gecode_results_path: &PathBuf,
    gecode_program_name: &str,
    progress_bar: &ProgressBar,
) -> Result<String, Box<dyn Error>> {
    let stderr = fs::read_to_string(pumpkin_outputs_dir_path.join("stderr"))?;
    if stderr.trim().len() > 0 {
        return Ok(format!(
            "ERR: Stderr in {pumpkin_outputs_dir_path:?} not empty {stderr}"
        ));
    }

    let run_info = fs::read_to_string(pumpkin_outputs_dir_path.join("run_info"))?;
    let problem_path_str = run_info
        .lines()
        .nth(1)
        .ok_or("Didn't find the 'file' line in 'run_info'")?
        .split(": ")
        .nth(1)
        .ok_or("Didn't find the path in file line")?;
    let problem_path = PathBuf::from(problem_path_str);

    let pumpkin_problem_file = problem_path
        .file_name()
        .ok_or("Couldn't retrieve file name")?
        .to_str()
        .ok_or("Couldn't convert file name")?
        .replace("\"", "");
    let gecode_problem_file = pumpkin_problem_file.replace("pumpkin.fzn", "gecode.fzn");

    if pumpkin_problem_file == "steiner-triples.07.pumpkin.fzn" {
        return Ok("ERR: Skipping Steiner Triples due to incorrect result in Gecode".to_string());
    }

    progress_bar.set_message(format!(
        "Problem files {pumpkin_problem_file} & {gecode_problem_file}, finding Gecode outputs"
    ));

    let gecode_outputs_dir = gecode_results_path
        .read_dir()?
        .map(|gecode_run_dir| {
            let gecode_run_dir_path = gecode_run_dir?.path().join(gecode_program_name);

            let stderr = fs::read_to_string(gecode_run_dir_path.join("stderr"))?;
            if !stderr.contains(gecode_problem_file.as_str()) {
                return Ok(None);
            }

            Ok(Some(gecode_run_dir_path))
        })
        .collect::<Result<Vec<Option<PathBuf>>, Box<dyn Error>>>()?
        .into_iter()
        .find(|output| output.is_some())
        .flatten();
    let Some(gecode_outputs_dir) = gecode_outputs_dir else {
        return Ok(format!(
            "ERR: Gecode outputs not matched for {pumpkin_problem_file}"
        ));
    };

    progress_bar.set_message("Finding OZNs");

    let ozns = problem_set_path
        .read_dir()?
        .map(|run_dir| {
            let run_dir_path = run_dir?.path();

            let pumpkin_ozn_file = run_dir_path.join(pumpkin_problem_file.replace("fzn", "ozn"));
            let gecode_ozn_file = run_dir_path.join(gecode_problem_file.replace("fzn", "ozn"));

            if pumpkin_ozn_file.exists() && gecode_ozn_file.exists() {
                return Ok(Some((pumpkin_ozn_file, gecode_ozn_file)));
            }

            Ok(None)
        })
        .collect::<Result<Vec<Option<(PathBuf, PathBuf)>>, Box<dyn Error>>>()?
        .into_iter()
        .find(|output| output.is_some())
        .flatten();
    let Some((pumpkin_ozn, gecode_ozn)) = ozns else {
        return Ok(format!("ERR: OZNs not matched for {pumpkin_problem_file}"));
    };

    progress_bar.set_message("Reading metrics");

    let time_regex = Regex::new(r"secs = 36\d\d").unwrap();

    let pumpkin_metrics = fs::read_to_string(pumpkin_outputs_dir_path.join("metrics"))?;
    if !pumpkin_metrics.contains("type = \"Done\"") {
        return Ok(format!("ERR: metric not done for {pumpkin_problem_file}"));
    } else if time_regex.is_match(&pumpkin_metrics) {
        return Ok(format!("ERR: timeout for {pumpkin_problem_file}"));
    }

    let gecode_metrics = fs::read_to_string(gecode_outputs_dir.join("metrics"))?;
    if !gecode_metrics.contains("type = \"Done\"") {
        return Ok(format!("ERR: metric not done for {gecode_problem_file}"));
    } else if time_regex.is_match(&gecode_metrics) {
        return Ok(format!("ERR: timeout for {gecode_problem_file}"));
    }

    progress_bar.set_message("Reading output scripts");

    let pumpkin_outputs = pumpkin_outputs_dir_path.join("run_outputs");
    if !pumpkin_outputs.exists() {
        return Ok(format!(
            "ERR: outputs not present for {pumpkin_problem_file}, file too large?"
        ));
    }

    let gecode_outputs = gecode_outputs_dir.join("run_outputs.dzn");
    if !gecode_outputs.exists() {
        return Ok(format!(
            "ERR: outputs not present for {gecode_problem_file}, file too large?"
        ));
    }

    let pumpkin_outputs_str = fs::read_to_string(&pumpkin_outputs)?;
    let gecode_outputs_str = fs::read_to_string(&gecode_outputs)?;

    if pumpkin_outputs_str.contains("=====UNSATISFIABLE=====")
        || gecode_outputs_str.contains("=====UNSATISFIABLE=====")
    {
        assert_eq!(
            pumpkin_outputs_str.contains("=====UNSATISFIABLE====="),
            gecode_outputs_str.contains("=====UNSATISFIABLE====="),
            "({pumpkin_problem_file}) Both pumpkin & gecode should be UNSATISFIABLE"
        );
    }

    // If the OZNs are identical, we can simply parse the old way!
    if fs::read_to_string(&pumpkin_ozn)? == fs::read_to_string(&gecode_ozn)? {
        progress_bar.set_message(format!(
            "Default parser reading {}",
            pumpkin_outputs
                .as_os_str()
                .to_str()
                .ok_or("Couldn't convert file name")?
        ));

        let actual_solutions = pumpkin_outputs_str.parse::<Solutions<false>>();

        progress_bar.set_message(format!(
            "Default parser reading {}",
            gecode_outputs
                .as_os_str()
                .to_str()
                .ok_or("Couldn't convert file name")?
        ));

        let expected_solutions = gecode_outputs_str.parse::<Solutions<false>>();

        let actual_solutions = match actual_solutions {
            Ok(actual_solutions) => actual_solutions,
            Err(err) => {
                return Ok(format!(
                    "ERR: parse act {pumpkin_problem_file} failed {err}"
                ));
            }
        };

        let expected_solutions = match expected_solutions {
            Ok(expected_solutions) => expected_solutions,
            Err(err) => {
                return Ok(format!(
                    "ERR: parse exp {pumpkin_problem_file} failed {err}"
                ));
            }
        };

        progress_bar.set_message("Checking equality of outputs");

        assert_eq!(actual_solutions, expected_solutions,
                   "({pumpkin_problem_file}) Did not find the elements {:?} in the expected solution and the expected solution contained {:?} while the actual solution did not.",
                   actual_solutions.assignments.iter().filter(|solution| !expected_solutions.assignments.contains(solution)).collect::<Vec<_>>(),
                   expected_solutions.assignments.iter().filter(|solution| !actual_solutions.assignments.contains(solution)).collect::<Vec<_>>()
        );
    } else {
        progress_bar.set_message(format!(
            "Minizinc parser reading {}",
            pumpkin_outputs
                .as_os_str()
                .to_str()
                .ok_or("Couldn't convert file name")?
        ));

        let actual_solutions = run_minizinc_with_ozn(pumpkin_outputs, pumpkin_ozn)?;

        progress_bar.set_message(format!(
            "Minizinc parser reading {}",
            gecode_outputs
                .as_os_str()
                .to_str()
                .ok_or("Couldn't convert file name")?
        ));

        let expected_solutions = run_minizinc_with_ozn(gecode_outputs, gecode_ozn)?;

        progress_bar.set_message("Checking equality of outputs");

        assert_eq!(actual_solutions, expected_solutions,
               "({pumpkin_problem_file}) Did not find the elements {:?} in the expected solution and the expected solution contained {:?} while the actual solution did not.",
               actual_solutions.iter().filter(|solution| !expected_solutions.contains(*solution)).collect::<Vec<_>>(),
               expected_solutions.iter().filter(|solution| !actual_solutions.contains(*solution)).collect::<Vec<_>>()
        );
    }

    Ok("".into())
}

fn main() -> Result<(), Box<dyn Error>> {
    let problem_set_path = &PathBuf::from(format!(
        "{}/benches/benchmarks-set-satisfy",
        env!("CARGO_MANIFEST_DIR")
    ));

    let pumpkin_results_path = &PathBuf::from(format!(
        "{}/benches/results/../..",
        env!("CARGO_MANIFEST_DIR")
    ));
    let pumpkin_program_name = "pumpkin-results";

    let gecode_results_path = &PathBuf::from(format!(
        "{}/benches/results/../..",
        env!("CARGO_MANIFEST_DIR")
    ));
    let gecode_program_name = "gecode-results";

    let pool = ThreadPoolBuilder::new().num_threads(12).build()?;

    let pumpkin_result_paths = pumpkin_results_path.read_dir()?.collect_vec();
    let pumpkin_result_paths_len = pumpkin_result_paths.len();

    let curr_i = AtomicU32::new(1);
    let total_ok = AtomicU32::new(0);
    let multi_progress = Arc::new(MultiProgress::new());

    pool.install(|| {
        let _ = pumpkin_result_paths
            .into_par_iter()
            .map(|pumpkin_outputs_dir| {
                let i = curr_i.fetch_add(1, Ordering::Relaxed);

                let progress_bar = multi_progress.add(ProgressBar::new(pumpkin_result_paths_len as u64));
                progress_bar.set_position(i as u64);
                progress_bar.set_style(
                    ProgressStyle::default_bar()
                        .template("{spinner:.green} [{elapsed_precise:.bold.blue}] ({pos:.bold.green}/{len:.bold.green}) {msg}")
                        .unwrap(),
                );
                progress_bar.enable_steady_tick(Duration::from_millis(250));

                let pumpkin_outputs_dir_path =
                    &pumpkin_outputs_dir.map_err(|err| format!("{err}"))?.path().join(pumpkin_program_name);

                progress_bar.set_message(format!("Checking {pumpkin_outputs_dir_path:?}..."));

                if let Err(err) = panic::catch_unwind(|| {
                    let logs = process_pumpkin_outputs_dir(
                        pumpkin_outputs_dir_path,
                        problem_set_path,
                        gecode_results_path,
                        gecode_program_name,
                        &progress_bar,
                    )
                        .map_err(|e| format!("{e}")).expect("Unexpected error");

                    if logs.len() > 0 {
                        if logs.contains("timeout for") ||
                            logs.contains("outputs not present for") ||
                            logs.contains("metric not done for") ||
                            logs.contains("should end with") {
                            progress_bar.finish_and_clear();
                        } else {
                            progress_bar.abandon_with_message(logs.replace("\n", " "));
                        }
                    } else {
                        let _ = total_ok.fetch_add(1, Ordering::Relaxed);
                        progress_bar.finish_and_clear();
                    }
                }) {
                    if let Some(s) = err.downcast_ref::<String>() {
                        progress_bar.println(format!("Panic msg (string): {}", s));
                    } else if let Some(s) = err.downcast_ref::<&str>() {
                        progress_bar.println(format!("Panic msg (str): {}", s));
                    } else {
                        progress_bar.println(format!("Unknown panic type: {:?}", err.type_id()))
                    }
                }

                Ok(())
            })
            .collect::<Result<Vec<()>, String>>()
            .expect("Error in any process");
    });

    println!("Total ok: {}", total_ok.load(Ordering::Relaxed));

    Ok(())
}
