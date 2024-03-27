# CanItEdit Benchmark

This directory contains programs to generate CanItEdit completions and evaluate them.

## Generating completions

To generate completions, you can use the `generate_completions.py` script.
It takes a `--model` parameter that specifies the model to use, and a `--model-type` parameter that specifies the type of the model, for example `direct` or `starcoder`.
The `--model-type` parameter mostly control the kind of prompt that is used to generate the completions.
To evaluate the EditCoder model, the `--model-type` parameter should be set to `direct`.
There is also `--output-dir` parameter, which points to the output directory where the
completions will be saved.
Other parameters can be found by running `python generate_completions.py --help`.

Here is an example of how to evaluate EditCoder, with 20 completions per program, 5 batch
size, temperature 0.2, top-p 0.95, and 4096 max tokens:

```bash
python generate_completions.py --model-type direct --model nuprl/EditCoder-6.7b-v1 --output-dir outputs --completion-limit 20 --batch-size 5 --temperature 0.2 --top-p 0.95 --max-tokens 4096
```

## Executing Tests on The Completions

When the completions have been generated, you can execute the tests to evaluate the completions.
To do this, you first need to install the Docker image that contains the test runner,
which can be done by running `make build-docker` in this directory.
Then, you can run the tests using the `evaluate_completions.sh` script,
pointing it to the directory where the completions are saved.

For example, to evaluate the completions generated in the previous step, which
were saved in the `outputs` directory, you can run:

```bash
./evaluate_completions.sh ./outputs
```

## Retrieving the Results

Finally, you can retrieve the results by running the `pass_k.py` script pointing it to the directory where the completions are saved.

For example, to retrieve the results from the previous step, you can run:

```bash
python pass_k.py ./outputs
```

You will be provided with a CSV-formatted table with the results, including `pass@1` and `ExcessCode` metrics.
You can provide a `-k` parameter to the script to change the value of `k` for the `pass@k` metric.

```
python pass_k.py -k 5 ./outputs
```

### Separating Results Based on Taxonomy

We provide two scripts that allow you to separate the results based on the taxonomy of the programs.

- `./separate_results.sh` simply separates the results based on the instruction kind (descriptive and lazy), as done in the paper.
- `./separate_results_on_change_kind.sh` separates the results based on the change kind (corrective, perfective, and adaptive).
  This scripts requires `jq` to be installed on your system.
