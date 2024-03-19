"""
Modified from NUPRL/MultiPL-E

This script calculates pass@k. It receives a list of directories as its
argument, and calculates the mean pass@k for the set of problems in each
directory along with the ExcessCode metric.

For visualization, it is reccomended to pipe the output to `column -t -s,`
or a csv file.

The output has the following columns:

- Name: the name of the experiment
- Pass@k: the value of k
- Estimate: the mean pass@k for the problems in the directory
- NumProblems: the number of problems in the directory
- MinCompletions: the minimum number of completions for any problem in the 
  directory
- MaxCompletions: the maximum number of completions for any problem in the
  directory
- ExcessCode: the mean of means of excess code for the problems in the directory.
  excess code is calculated as the number of lines not covered by the test
  coverage divided by the number of lines changed in the program from the
  before to the after version.
- ExcessCodeSE: the standard error of the median excess code.
- MeanMedianCoverage: the mean median (100-coverage) for the problems. This used to 
  be the old ExcessCode metric, but it was changed to be more accurate.
"""
import numpy as np
from pathlib import Path
import itertools
import argparse
import json
import gzip
from typing import Optional
import sys
import difflib

v1_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34,
          35, 36, 37, 38, 39, 3, 40, 41, 44, 45, 46, 47, 48, 49, 4, 50, 51, 52, 53, 55, 56, 57, 57, 58, 60, 6, 7, 8, 9]


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def gunzip_json(path: Path) -> Optional[dict]:
    """
    Reads a .json.gz file, but produces None if any error occurs.
    """
    try:
        with gzip.open(path, "rt") as f:
            return json.load(f)
    except Exception as e:
        return None


def estimator(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))  # type: ignore


def compute_excess_score(before, result) -> float:
    after = result["program_no_test"]
    differ = difflib.Differ()
    num_changed_lines = len(
        list(differ.compare(before.splitlines(), after.splitlines())))
    num_uncovered_lines = len(result["coverage_missing"])
    return num_uncovered_lines / num_changed_lines


def for_file(path):
    data = gunzip_json(path)
    if data is None:
        return None
    n = len(data["results"])
    c = len([True for r in data["results"] if r["status"]
            == "OK" and r["exit_code"] == 0])
    before = data["before"]
    excesscodes = [compute_excess_score(before, r) for r in data["results"] if r["status"]
                   == "OK" and r["exit_code"] == 0 and "coverage" in r and r["coverage"] is not None]
    coverages = [(100-r["coverage"]) for r in data["results"] if r["status"]
                 == "OK" and r["exit_code"] == 0 and "coverage" in r and r["coverage"] is not None]
    median_excesscode = np.mean(
        excesscodes) if len(excesscodes) > 0 else None
    median_coverage = np.median(coverages) if len(coverages) > 0 else None
    return {
        "pass@1": estimator(n, c, 1),
        "pass@10": estimator(n, c, 10),
        "pass@100": estimator(n, c, 100),
        "median_excesscode": median_excesscode,
        "median_coverage": median_coverage,
        "n": n,
        "c": c,
        "temperature": data["temperature"] if "temperature" in data else 0.2,
        "id": data["id"]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suppress-header",
                        action="store_true", help="Suppress the header")
    parser.add_argument("-k", type=int, default=None, help="The value of k")
    parser.add_argument("--v1_only", action="store_true",
                        help="Only use v1 results")
    parser.add_argument(
        "dirs", type=str,  help="Directories with results. ", nargs="+")
    args = parser.parse_args()
    if not args.suppress_header:
        print(
            "Name,Pass@k,Estimate,NumProblems,MinCompletions,MaxCompletions,ExcessCode,ExcessCodeSE,MeanMedianCoverage")
    for d in args.dirs:
        results = [for_file(p) for p in itertools.chain(
            Path(d).glob("*.results.json"), Path(d).glob("*.results.json.gz"))]
        results = [r for r in results if r is not None]
        if args.v1_only:
            results = [r for r in results if r["id"] in v1_ids]
        name = d.split("/")[-1] if d.split("/")[-1] != "" else d.split("/")[-2]
        temperatures = set(r["temperature"] for r in results)
        if len(temperatures) != 1:
            eprint(
                f"Found multiple temperatures {temperatures} in {d} {results}")
            continue
        temperature = list(temperatures)[0]
        num_problems = len(results)
        min_completions = np.min([r["n"] for r in results])
        max_completions = np.max([r["n"] for r in results])
        median_excesscode = [r["median_excesscode"]
                             for r in results if r["median_excesscode"] is not None]
        median_coverage = [r["median_coverage"]
                           for r in results if r["median_coverage"] is not None]
        # round everything to 6 decimal places
        if len(median_excesscode) > 0:
            mean_median_coverage = np.round(np.mean(median_coverage), 6)
            excess_code = np.round(np.mean(median_excesscode) * 100, 6)
            excess_code_se = np.round(
                (np.std(median_excesscode) / np.sqrt(len(median_excesscode))) * 100, 6)
        else:
            excess_code = "NA"
            excess_code_se = "NA"
            mean_median_coverage = "NA"

        if temperature == 0.8:
            pass_10 = np.round(np.mean([r["pass@10"]
                               for r in results]) * 100, 6)
            pass_100 = np.round(np.mean([r["pass@100"]
                                for r in results]) * 100, 6)
            print(
                f"{name},10,{pass_10},{num_problems},{min_completions},{max_completions},{excess_code},{excess_code_se},{mean_median_coverage}")
            print(
                f"{name},100,{pass_100},{num_problems},{min_completions},{max_completions},{excess_code},{excess_code_se},{mean_median_coverage}")
        else:
            pass_1 = np.round(np.mean([r["pass@1"]
                              for r in results]) * 100, 6)
            print(
                f"{name},1,{pass_1},{num_problems},{min_completions},{max_completions},{excess_code},{excess_code_se},{mean_median_coverage}")

        if args.k is not None:
            pass_k = np.round(np.mean([estimator(r["n"], r["c"], args.k)
                                       for r in results]) * 100, 6)
            print(
                f"{name},{args.k},{pass_k},{num_problems},{min_completions},{max_completions},{excess_code},{excess_code_se},{mean_median_coverage}")


if __name__ == "__main__":
    main()
