import tempfile
from pathlib import Path
import subprocess
from typing import Optional, List, Tuple


def parse_missing_ranges(missing: str) -> List[int]:
    """
    E.g. 4, 17-18, 26, 29, 32, 35, 38, 41, 44, 47-51, 54
    Into:
    [4, 17, 18, 26, 29, 32, 35, 38, 41, 44, 47, 48, 49, 50, 51, 54]
    [4, 17, 18, 26, 29, 32, 35, 38, 41, 44, 47, 48, 49, 50, 51, 54]
    """
    ranges = missing.split(", ")
    result = []
    for r in ranges:
        if "-" in r:
            start, end = r.split("-")
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(r))
    return result


def coverage(code: str, timeout=60) -> Optional[Tuple[int, List[int]]]:
    # create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as tmp_file:
        tmp_file_name = tmp_file.name
        tmp_file.write(code)

    cov_file_name = f"{tmp_file_name}.cov"

    def run_coverage():
        # run coverage analysis
        try:
            proc = subprocess.run(
                ['coverage', 'run', '--data-file', cov_file_name, tmp_file_name],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            print("Coverage timedout...")
            return None
        if proc.returncode != 0:
            return None

        # generate coverage report
        proc = subprocess.run(
            ['coverage', 'report', '-m', '--data-file', cov_file_name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
        )
        if proc.returncode != 0:
            return None

        # parse coverage percentage
        cov_percentage = 0
        next_is_cov = False
        missing = []
        for line in proc.stdout.decode().splitlines():
            if next_is_cov:
                parts = [s.strip('%') for s in line.split() if s]
                cov_percentage = int(parts[3])
                if cov_percentage < 100:
                    missing = parse_missing_ranges(" ".join(parts[4:]))
                break
            elif line.startswith("---------"):
                next_is_cov = True

        return cov_percentage, missing

    result = run_coverage() or None

    # cleanup
    subprocess.run(['rm', tmp_file_name])
    subprocess.run(['rm', cov_file_name], stderr=subprocess.DEVNULL)

    return result


def process_comp(comp):
    if "program_no_test" not in comp:
        # add program without tests
        program_no_test = comp["program"].split("### START TESTS ###\n")[0]
        comp["program_no_test"] = program_no_test
    if "coverage" in comp:
        return comp
    if comp["status"] == "OK":
        cov = coverage(comp["program"])
        if cov:
            comp["coverage"], comp["coverage_missing"] = cov
        else:
            comp["coverage"] = None
            comp["coverage_missing"] = None
    else:
        comp["coverage"] = None
        comp["coverage_missing"] = None
    return comp

SUPPORTED_LANGS = ["py"]

if __name__ == "__main__":
    # import main from the real evaluator
    try:
        from main import main, open_json  # type: ignore
    except ImportError:
        from evaluator import main, open_json  # type: ignore

    # run the real evaluator
    main()

    # now the fun begins
    import argparse
    import json
    from tqdm import tqdm
    import multiprocessing
    import os
    args = argparse.ArgumentParser()
    args.add_argument("--output-dir", type=Path)
    args = args.parse_known_args()[0]

    paths = list(args.output_dir.glob("*.results.json.gz"))
    THREADS = min(os.cpu_count() or 1, len(paths))

    for path in tqdm(paths, total=len(paths), desc="Calculating coverage"):
        # load the results
        with open_json(path, "r") as f:
            results = json.load(f)

        lang = results['language']
        if lang not in SUPPORTED_LANGS:
            continue
        comps = results['results']
        all_had_cov = all("coverage" in comp for comp in comps)

        if not all_had_cov:
            with multiprocessing.Pool(THREADS) as pool:
                comps = pool.map(process_comp, comps)
            results['results'] = comps

            # save the results
            with open_json(path, "w") as f:
                json.dump(results, f)
