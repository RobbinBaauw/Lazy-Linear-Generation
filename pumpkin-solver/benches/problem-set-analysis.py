import dataclasses
import pickle
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

PAPER_SET_DIR = (Path(__file__).parent / "datasets" / "paper-set").resolve()


@dataclass
class PropagatorCounts:
    maximum: int
    element: int
    not_equal: int
    less_equal: int
    reified: int
    times: int
    div: int
    abs: int

    @staticmethod
    def from_dict(dict) -> "PropagatorCounts":
        return PropagatorCounts(
            dict["maximum"],
            dict["element"],
            dict["not_equal"],
            dict["less_equal"],
            dict["reified"],
            dict["times"],
            dict["div"],
            dict["abs"]
        )
    
    def sum(self, other: "PropagatorCounts") -> "PropagatorCounts":
        return PropagatorCounts(
            self.maximum + other.maximum,
            self.element + other.element,
            self.not_equal + other.not_equal,
            self.less_equal + other.less_equal,
            self.reified + other.reified,
            self.times + other.times,
            self.div + other.div,
            self.abs + other.abs,
        )


def analyze_dataset():
    counts = {}
    instances = list(PAPER_SET_DIR.rglob("*.fzn"))
    for (i, fzn) in enumerate(instances):
        print(f"{i}/{len(instances)} reading")
        fzn_contents = fzn.open().read()

        cnt = PropagatorCounts(0, 0, 0, 0, 0, 0, 0, 0)

        cnt.maximum += fzn_contents.count("array_int_maximum")

        cnt.maximum += fzn_contents.count("array_int_minimum")

        cnt.maximum += fzn_contents.count("int_max")

        cnt.maximum += fzn_contents.count("int_min")

        cnt.element += fzn_contents.count("array_int_element")

        cnt.element += fzn_contents.count("array_var_int_element")

        cnt.not_equal += fzn_contents.count("int_lin_ne")

        # int_lin_ne_reif becomes a:
        # - 1x reified not equals
        # - 2x reified less than equal
        cnt.not_equal += fzn_contents.count("int_lin_ne_reif")
        cnt.less_equal += 2 * fzn_contents.count("int_lin_ne_reif")
        cnt.reified += 3 * fzn_contents.count("int_lin_ne_reif")

        cnt.less_equal += fzn_contents.count("int_lin_le")

        # int_lin_le_reif becomes 2x reified less than equal
        cnt.less_equal += 2 * fzn_contents.count("int_lin_le_reif")
        cnt.reified += 2 * fzn_contents.count("int_lin_le_reif")

        # int_lin_eq becomes 2x less than equal
        cnt.less_equal += 2 * fzn_contents.count("int_lin_eq")

        # int_lin_eq_reif becomes a:
        # - 2x reified less than equals
        # - 1x reified not equals
        cnt.not_equal += fzn_contents.count("int_lin_eq_reif")
        cnt.less_equal += 2 * fzn_contents.count("int_lin_eq_reif")
        cnt.reified += 3 * fzn_contents.count("int_lin_eq_reif")

        cnt.not_equal += fzn_contents.count("int_ne")

        # int_ne_reif becomes a:
        # - 1x reified not equals
        # - 2x reified less than equal
        cnt.not_equal += fzn_contents.count("int_ne_reif")
        cnt.less_equal += 2 * fzn_contents.count("int_ne_reif")
        cnt.reified += 3 * fzn_contents.count("int_ne_reif")

        # int_eq becomes 2x less than equal
        cnt.less_equal += 2 * fzn_contents.count("int_eq")

        # int_eq_reif becomes a:
        # - 2x reified less than equals
        # - 1x reified not equals
        cnt.not_equal += fzn_contents.count("int_eq_reif")
        cnt.less_equal += 2 * fzn_contents.count("int_eq_reif")
        cnt.reified += 3 * fzn_contents.count("int_eq_reif")

        cnt.less_equal += fzn_contents.count("int_le")

        # int_le_reif becomes 2x reified less than equal
        cnt.less_equal += 2 * fzn_contents.count("int_le_reif")
        cnt.reified += 2 * fzn_contents.count("int_le_reif")

        cnt.less_equal += fzn_contents.count("int_lt")

        # int_lt_reif becomes 2x reified less than equal
        cnt.less_equal += 2 * fzn_contents.count("int_lt_reif")
        cnt.reified += 2 * fzn_contents.count("int_lt_reif")

        # int_plus becomes 2x less than equal
        cnt.less_equal += 2 * fzn_contents.count("int_plus")

        cnt.times += fzn_contents.count("int_times")

        cnt.div += fzn_contents.count("int_div")

        cnt.abs += fzn_contents.count("int_abs")

        # pumpkin_all_different becomes n(n+1)/2 not equals (with n = len-1)
        # we don't know the length, so comment out
        # not_equal += fzn_contents.count("pumpkin_all_different")

        # array_bool_and becomes 2x less than equal
        cnt.less_equal += 2 * fzn_contents.count("array_bool_and")

        cnt.element += fzn_contents.count("array_bool_element")

        cnt.element += fzn_contents.count("array_var_bool_element")

        # array_bool_or becomes 2x less than equal
        cnt.less_equal += 2 * fzn_contents.count("array_bool_or")

        # pumpkin_bool_xor becomes 2x less than equal
        cnt.less_equal += 2 * fzn_contents.count("pumpkin_bool_xor")

        # pumpkin_bool_xor_reif becomes a:
        # - 2x reified less than equals
        # - 1x reified not equals
        cnt.not_equal += fzn_contents.count("pumpkin_bool_xor_reif")
        cnt.less_equal += 2 * fzn_contents.count("pumpkin_bool_xor_reif")
        cnt.reified += 3 * fzn_contents.count("pumpkin_bool_xor_reif")

        # bool2int becomes 2x less than equal
        cnt.less_equal += 2 * fzn_contents.count("bool2int")

        cnt.less_equal += fzn_contents.count("bool_lin_le")

        # bool_and becomes 2x less than equal
        cnt.less_equal += 2 * fzn_contents.count("bool_and")

        # bool_clause becomes 2x less than equal
        cnt.less_equal += 2 * fzn_contents.count("bool_clause")

        # bool_eq becomes 2x less than equal
        cnt.less_equal += 2 * fzn_contents.count("bool_eq")

        # bool_eq_reif becomes a:
        # - 2x reified less than equals
        # - 1x reified not equals
        cnt.not_equal += fzn_contents.count("bool_eq_reif")
        cnt.less_equal += 2 * fzn_contents.count("bool_eq_reif")
        cnt.reified += 3 * fzn_contents.count("bool_eq_reif")

        cnt.not_equal += fzn_contents.count("bool_not")

        # set_in_reif becomes a 4x reified less than equals
        cnt.less_equal += 4 * fzn_contents.count("set_in_reif")
        cnt.reified += 4 * fzn_contents.count("set_in_reif")

        counts[fzn] = cnt

    counts_dict = {p: dataclasses.asdict(v) for (p, v) in counts.items()}
    with open("./paper-set-counts.pkl", "wb") as f:
        pickle.dump(counts_dict, f)


def print_counts():
    with open("./paper-set-counts.pkl", "rb") as f:
        counts = pickle.load(f)

    print(counts)

    count_values = map(lambda v: PropagatorCounts.from_dict(v), counts.values())
    print(reduce(lambda a, b: a.sum(b), count_values))


if __name__ == "__main__":
    analyze_dataset()
    print_counts()
