import gzip
import json
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import total_ordering
from pathlib import Path
from typing import Optional, Dict, List, Tuple, BinaryIO, Generator, Union, TypeVar, NamedTuple


@dataclass
class LinearLeq:
    num_executions: int
    num_propagations: int
    num_pb_vars: int


@dataclass
class Stats:
    objective: Optional[int]
    solver_time: float

    num_decisions: int
    num_conflicts: int
    num_restarts: int
    num_propagations: int

    llg_learned_constraints: int
    llg_learned_constraints_avg_length: float
    llg_learned_constraints_avg_coeff: float
    llg_fallback_used: int

    llg_linked_auxiliaries: int
    llg_unlinked_auxiliaries: int
    llg_unlinked_auxiliaries_constraints: int

    llg_time_spent_analysis_ns: int
    llg_time_spent_cutting_ns: int
    llg_time_spent_checking_backjump_ns: int
    llg_time_spent_checking_overflow_ns: int
    llg_time_spent_checking_conflicting_ns: int
    llg_time_spent_creating_explanations_ns: int

    linear_leqs: Dict[int, LinearLeq]


class Result(Enum):
    UNSAT = "unsat"
    UNKNOWN = "unknown"
    SUCCESS = "success"


class Program(Enum):
    INTSAT = "IntSat"
    LLG = "LLG"
    RESOLUTION = "Resolution"

    def is_llg(self):
        return self == Program.LLG

    def is_intsat(self):
        return self == Program.INTSAT

    def is_resolution(self):
        return self == Program.RESOLUTION


@dataclass
class Outputs:
    result: Result
    outputs: Optional[List[List[str]]]


ConflictId = int
VarId = int
VarScale = int
VarBounds = Tuple[int, int]
PropagatorId = int


@dataclass(eq=True, frozen=True)
class ExplanationStats:
    explanation_id: int
    nr_vars_lhs: int
    slack: int


@dataclass(eq=True, frozen=True)
class LearnedInequalityStats:
    backtrack_distance: int
    slack: int
    used_explanations: List[ExplanationStats]


@dataclass(eq=True, frozen=True)
class LearnedNogoodStats:
    backtrack_distance: int
    lbd: int


@dataclass(eq=True, frozen=True)
class LearnedPropagationErrorStats:
    propagations_at_conflicts: Dict[int, int]
    total_propagations: int
    matched_propagations: int

    errors_at_conflicts: Dict[int, int]
    total_errors: int
    matched_errors: int


@total_ordering
class ConstraintType(IntEnum):
    NOGOOD = 0
    INEQUALITY = 1

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class FallbackReason(IntEnum):
    PROOF_COMPLETION = 0
    NOT_CONFLICTING = 1
    OVERFLOW = 2
    NOGOOD_EXPLANATION = 3
    NOGOOD_CONFLICT = 4
    DECISION_REACHED = 5
    NOTHING_LEARNED = 6


@dataclass
class FallbackConstraint:
    fallback_at_conflict: int
    fallback_reason: FallbackReason

    used_explanations: List[ExplanationStats]

    propagation_error_stats: LearnedPropagationErrorStats


@dataclass
class LearnedConstraint:
    learned_at_conflict: int
    constraint_type: ConstraintType

    inequality_stats: LearnedInequalityStats
    nogood_stats: LearnedNogoodStats

    propagation_error_stats: LearnedPropagationErrorStats


LearnedConstraints = Dict[str, Union[LearnedConstraint, FallbackConstraint]]


@dataclass
class RunData:
    stats: Stats
    outputs: Outputs
    learned_constraints: Optional[LearnedConstraints]


@dataclass
class RunResult:
    exit_code: int
    wall_time: float

    bench_version: int
    git_hash: str

    fzn_file_name: str
    fzn_file_path: Path

    program: Program
    skip_nogood_learning: bool

    stderr: Optional[str]
    stdout: Optional[str]

    run_data: Optional[RunData]

    def short_result(self):
        if self.stderr is not None:
            return "E"

        if self.wall_time > 3600:
            return "T"

        if self.exit_code == -9:
            return "OOM"

        if self.run_data is None or self.run_data.outputs.result == Result.UNKNOWN:
            return "?"

        if self.run_data.outputs.result == Result.SUCCESS:
            return "S"

        if self.run_data.outputs.result == Result.UNSAT:
            return "U"

    def timed_out(self):
        return self.wall_time > 3600

    def failed_print_reason(self):
        if self.exit_code != 0:
            print(" ==> Exit code")
            return True

        if self.timed_out():
            print(" ==> Timed out")
            return True

        if (self.run_data is None) or (self.run_data.outputs.result == Result.UNKNOWN):
            print(" ==> Unknown reason")
            return True

        return False

    def failed(self):
        return ((self.exit_code != 0) or
                (self.timed_out()) or
                (self.run_data is None) or
                (self.run_data.outputs.result == Result.UNKNOWN))


ProgramResults = Dict[Tuple[Program, str], RunResult]
InstanceResults = Tuple[str, ProgramResults]


def parse_metrics(metrics_path: Path):
    with open(metrics_path) as metrics_file:
        metrics_lines = metrics_file.read().split("\n")

    wall_time = float(metrics_lines[0].split(":")[1])
    exit_code = int(metrics_lines[1].split(":")[1])

    return exit_code, wall_time


def parse_run_info(info_path: Path):
    with open(info_path) as info_file:
        info_lines = info_file.read().split("\n")

    version = int(info_lines[0].split(": ")[1])
    git_hash = info_lines[1].split(": ")[1]

    args = json.loads(info_lines[2].split(": ")[1])

    file_path = Path(args["instance_path"].replace("\"", ""))
    file_name = file_path.stem

    use_llg = args["use_llg"]
    skip_nogood_learning = args["llg_skip_nogood_learning"]

    if use_llg:
        program = Program.LLG
    else:
        program = Program.RESOLUTION

    return version, git_hash, file_path, file_name, program, skip_nogood_learning


def parse_stderr(stderr_path: Path):
    with open(stderr_path) as stderr_file:
        stderr = stderr_file.read()

    return stderr if len(stderr) > 0 else None


def parse_stdout(stdout_path: Path):
    with open(stdout_path) as stdout_file:
        stdout = stdout_file.read()

    return stdout if len(stdout) > 0 else None


def parse_stat_file(stat_path: Path):
    with open(stat_path) as stat_file:
        stats = stat_file.read().strip()

    # Invalid
    if len(stats) == 0:
        return None

    linear_leq_id_values = defaultdict(lambda: LinearLeq(0, 0, 0))

    for (linear_leq_id, linear_leq_field, linear_leq_value) in re.findall("LinearLeq_number_(\d+)_(.+)=(.+)", stats):
        match linear_leq_field:
            case "number_of_executions":
                linear_leq_id_values[linear_leq_id].num_executions = int(linear_leq_value)
            case "number_of_propagations":
                linear_leq_id_values[linear_leq_id].num_propagations = int(linear_leq_value)
            case "number_of_pb_vars":
                linear_leq_id_values[linear_leq_id].num_pb_vars = int(linear_leq_value)

    def get_stat_value(stat: str, stats: str):
        stat_res = re.search(fr"\$stat\$ {stat}=(.+)", stats)
        if stat_res is not None:
            return stat_res.group(1)

        return None

    return Stats(
        objective=get_stat_value("objective", stats),
        solver_time=int(get_stat_value("_engine_statistics_time_spent_in_solver", stats)) / 1000.0,

        num_decisions=int(get_stat_value("_engine_statistics_num_decisions", stats)),
        num_conflicts=int(get_stat_value("_engine_statistics_num_conflicts", stats)),
        num_restarts=int(get_stat_value("_engine_statistics_num_restarts", stats)),
        num_propagations=int(get_stat_value("_engine_statistics_num_propagations", stats)),

        llg_learned_constraints=int(get_stat_value("_llg_statistics_llg_learned_constraints", stats)),
        llg_learned_constraints_avg_length=float(get_stat_value("_llg_statistics_llg_learned_constraints_avg_length", stats)),
        llg_learned_constraints_avg_coeff=float(get_stat_value("_llg_statistics_llg_constraint_avg_lhs_coeff", stats)),
        llg_fallback_used=int(get_stat_value("_llg_statistics_llg_fallback_used", stats)),

        llg_linked_auxiliaries=int(get_stat_value("_llg_statistics_llg_linked_auxiliaries", stats)),
        llg_unlinked_auxiliaries=int(get_stat_value("_llg_statistics_llg_unlinked_auxilaries", stats)),
        llg_unlinked_auxiliaries_constraints=int(get_stat_value("_llg_statistics_llg_unlinked_auxiliaries_constraints", stats)),

        llg_time_spent_analysis_ns=int(get_stat_value("_llg_statistics_llg_time_spent_analysis_ns", stats)),
        llg_time_spent_cutting_ns=int(get_stat_value("_llg_statistics_llg_time_spent_cutting_ns", stats)),
        llg_time_spent_checking_backjump_ns=int(get_stat_value("_llg_statistics_llg_time_spent_checking_backjump_ns", stats)),
        llg_time_spent_checking_overflow_ns=int(get_stat_value("_llg_statistics_llg_time_spent_checking_overflow_ns", stats)),
        llg_time_spent_checking_conflicting_ns=int(get_stat_value("_llg_statistics_llg_time_spent_checking_conflicting_ns", stats)),
        llg_time_spent_creating_explanations_ns=int(get_stat_value("_llg_statistics_llg_time_spent_creating_explanations_ns", stats)),

        linear_leqs=dict(linear_leq_id_values)
    )


def parse_outputs(outputs_path: Path):
    with open(outputs_path) as outputs_file:
        outputs_lines = outputs_file.read().split("\n")

    if "=====UNKNOWN=====" in outputs_lines[0]:
        result = Result.UNKNOWN
        outputs = None
    elif "=====UNSATISFIABLE=====" in outputs_lines[0]:
        result = Result.UNSAT
        outputs = None
    else:
        result = Result.SUCCESS

        outputs = [[]]

        outputs_line_i = 0
        while "==========" not in outputs_lines[outputs_line_i]:
            if "----------" in outputs_lines[outputs_line_i]:
                outputs.append([])

            outputs[-1].append(outputs_lines[outputs_line_i])
            outputs_line_i += 1

    return Outputs(result, outputs)


T = TypeVar("T")


def chunk(arr: List[T], chunk_size: int) -> List[List[T]]:
    return [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]


class LearnedConstraintsParser:
    def __init__(self):
        self.queued_new_constraint_line: Optional[Tuple[int, str]] = None
        self.learned_constraints: LearnedConstraints = {}
        self.key_version = defaultdict(lambda: 0)

    @staticmethod
    def parse_explanation_stats(stat_fields: List[str]) -> List[ExplanationStats]:
        stat_fields = list(filter(lambda f: len(f) > 0, stat_fields))
        chunks = chunk(stat_fields, 3)
        return list(map(lambda e: ExplanationStats(int(e[0]), int(e[1]), int(e[2])), chunks))

    @staticmethod
    def parse_counts(count_str: str) -> Dict[int, int]:
        if len(count_str) == 0:
            return {}

        counts = {}
        for count_item in count_str.split(" "):
            if "-" in count_item:
                confl, count = count_item.split("-")
                counts[int(confl)] = int(count)
            else:
                counts[int(count_item)] = 1

        return counts

    def parse_line(self, line: str):
        line = line.strip()
        if len(line) == 0:
            return

        conflict_nr, event, *fields = line.split("|")
        conflict_nr = int(conflict_nr)

        match event:
            case 'F':
                (iden, reason, *analysis_used_explanations) = fields

                iden_curr = f"{iden}_{self.key_version[iden]}"
                if iden_curr in self.learned_constraints:
                    self.key_version[iden] += 1

                self.learned_constraints[f"{iden}_{self.key_version[iden]}"] = FallbackConstraint(
                    conflict_nr,
                    FallbackReason(int(reason)),

                    self.parse_explanation_stats(analysis_used_explanations),

                    None
                )

            case 'S':
                (iden,
                 inequality_backtrack_levels, inequality_slack,
                 nogood_backtrack_levels, nogood_lbd,
                 *constraint_used_explanations) = fields

                iden_curr = f"{iden}_{self.key_version[iden]}"
                if iden_curr in self.learned_constraints:
                    self.key_version[iden] += 1

                inequality_stats = LearnedInequalityStats(
                    inequality_backtrack_levels,
                    inequality_slack,
                    self.parse_explanation_stats(constraint_used_explanations)
                )

                nogood_stats = LearnedNogoodStats(
                    nogood_backtrack_levels,
                    nogood_lbd
                )

                self.learned_constraints[f"{iden}_{self.key_version[iden]}"] = LearnedConstraint(
                    conflict_nr,
                    ConstraintType.NOGOOD if iden.startswith('N') else ConstraintType.INEQUALITY,

                    inequality_stats,
                    nogood_stats,

                    None
                )

            case 'P':
                iden, total_props, matched_props, total_errs, matched_errs, prop_count, err_count = fields

                self.learned_constraints[
                    f"{iden}_{self.key_version[iden]}"].propagation_error_stats = LearnedPropagationErrorStats(
                    self.parse_counts(prop_count),
                    int(total_props),
                    int(matched_props),
                    self.parse_counts(err_count),
                    int(total_errs),
                    int(matched_errs)
                )


def parse_learned_constraints(learned_constraints_path: Path) -> Optional[LearnedConstraints]:
    learned_constraints_path_gz = learned_constraints_path.with_suffix(".gz")
    if learned_constraints_path.exists():
        file_size = learned_constraints_path.stat().st_size
        print(f"==> Parse learned constraints ({file_size / 1_000_000}MB)")

        with open(learned_constraints_path) as learned_constraints_file:
            parser = LearnedConstraintsParser()
            while line := learned_constraints_file.readline():
                parser.parse_line(line)
            return parser.learned_constraints

    elif learned_constraints_path_gz.exists():
        file_size = learned_constraints_path_gz.stat().st_size
        print(f"==> Parse learned constraints ({file_size / 1_000_000}MB)")

        with gzip.open(learned_constraints_path_gz, "rt") as learned_constraints_file:
            parser = LearnedConstraintsParser()
            while line := learned_constraints_file.readline():
                parser.parse_line(line)
            return parser.learned_constraints

    else:
        return None


def parse_intsat(stderr_path: Path):
    with open(stderr_path) as stderr_file:
        output = stderr_file.read().split("\n")

    file_path = Path(output[1].split(":  ")[1])
    file_name = file_path.stem

    if "Internal error" in output[2]:
        return "intsat", file_name, file_path, "??", None, None

    for stats_line_i in range(5, len(output)):
        line = output[stats_line_i]

        if "Decisions:" in line:
            num_decisions = int(line.split(":")[1].strip())

        if "Conflicts:" in line:
            num_conflicts = int(line.split(":")[1].strip())

        if "Restarts:" in line:
            num_restarts = int(line.split(":")[1].strip())

        if "Total learned Constrs" in line:
            intsat_learned_constraints = int(line.split(":")[1].strip())

        if "Avg. size of learned Ctrs" in line:
            try:
                intsat_learned_constraints_avg_length = int(line.split(":")[1].strip())
            except ValueError:
                intsat_learned_constraints_avg_length = 0

    return -1, file_name, file_path, None, None, RunData(
        Stats(
            0,
            0.0,
            num_decisions,
            num_conflicts,
            num_restarts,
            0,
            intsat_learned_constraints,
            intsat_learned_constraints_avg_length,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            dict()
        ),
        Outputs(Result.SUCCESS, None),
        None,
    )


def parse_experiment_to_file(results_dir: Path, results_parsed_pkl: BinaryIO, skip_learned_constraints=False):
    input_dirs = list(filter(lambda d: d.is_dir(), results_dir.iterdir()))

    if len(input_dirs) > 1:
        print("Multiple input dirs found, instances might be overriden")
        exit(1)

    for instance_dir in input_dirs[0].iterdir():
        instance_results = parse_experiment_instance(instance_dir, skip_learned_constraints)
        if instance_results is not None:
            pickle.dump(instance_results, results_parsed_pkl, pickle.HIGHEST_PROTOCOL)


def parse_experiment_generator(results_dir: Path, skip_learned_constraints=False) -> Generator[Tuple[str, InstanceResults], None, None]:
    for input_dir in results_dir.iterdir():
        if not input_dir.is_dir():
            continue

        for instance_dir in input_dir.iterdir():
            instance_results = parse_experiment_instance(instance_dir, skip_learned_constraints)
            if instance_results is not None:
                yield input_dir.name, instance_results


class InstanceRunData(NamedTuple):
    input_name: str
    instance: str
    program_name: str
    run_result: RunResult
    run_data: RunData


def parse_instance_run_data_generator(results_dir: Path) -> Generator[InstanceRunData, None, None]:
    for (input_name, instance_results) in parse_experiment_generator(results_dir):
        if instance_results is None:
            continue

        (instance, program_results) = instance_results
        for ((_, prog_name), prog_result) in program_results.items():
            if (prog_result is None) or (prog_result.run_data is None):
                continue

            yield InstanceRunData(input_name, instance, prog_name, prog_result, prog_result.run_data)


def parse_experiment_instance(instance_dir: Path, skip_learned_constraints=False) -> Optional[InstanceResults]:
    if not instance_dir.is_dir():
        return None

    program_results: ProgramResults = {}

    for program_dir in instance_dir.iterdir():
        if not program_dir.is_dir():
            continue

        # Skip unfinished runs
        if not (program_dir / "metrics").exists():
            print(f"Skipping {program_dir}, still empty")
            continue

        print(f"Parsing {program_dir}")
        exit_code, wall_time = parse_metrics(program_dir / "metrics")

        if (program_dir / "intsat.pid.txt").exists():
            version, file_name, file_path, stderr, stdout, run_data = parse_intsat(program_dir / "stderr")
            run_result = RunResult(exit_code, wall_time, version, "",
                                   file_name, file_path,
                                   Program.INTSAT, False,
                                   stderr, stdout, run_data)
        else:
            stderr = parse_stderr(program_dir / "stderr")
            stdout = parse_stdout(program_dir / "stdout")

            version, git_hash, file_path, file_name, program, skip_nogood_learning = parse_run_info(
                program_dir / "run_info")

            if stderr is None:
                stats = parse_stat_file(program_dir / "run_stats")
                if stats is None:
                    run_data = None
                else:
                    outputs = parse_outputs(program_dir / "run_outputs")

                    if skip_learned_constraints:
                        learned_constraints = None
                    else:
                        learned_constraints = parse_learned_constraints(program_dir / "analysis_log")

                    run_data = RunData(stats, outputs, learned_constraints)
            else:
                run_data = None

            run_result = RunResult(exit_code, wall_time, version, git_hash,
                                   file_name, file_path,
                                   program, skip_nogood_learning,
                                   stderr, stdout, run_data)

        program_results[(run_result.program, program_dir.name)] = run_result

    if len(program_results) > 0:
        return run_result.fzn_file_name, program_results
    else:
        return None


def generate_results(results_parsed_pkl: BinaryIO) -> Generator[T, None, None]:
    try:
        while True:
            instance_results: T = pickle.load(results_parsed_pkl)
            yield instance_results
    except EOFError:
        results_parsed_pkl.seek(0)
        pass


def all_results(results_parsed_pkl: BinaryIO) -> Dict[str, ProgramResults]:
    instances_results = {}

    try:
        while True:
            (instance, program_results) = pickle.load(results_parsed_pkl)
            instances_results[instance] = program_results
    except EOFError:
        results_parsed_pkl.seek(0)
        pass

    return instances_results
