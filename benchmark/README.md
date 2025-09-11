# CanItEdit Benchmark

This directory contains programs to generate CanItEdit completions and evaluate them.

## Generating Completions

For a base model:

```bash
./generate_completions.py --model-type direct --model openai/Qwen/Qwen3-4B-Base --output-dir out
```

For a chat model:

```bash
./generate_completions.py --model-type chat --model openai/Qwen/Qwen3-4B-Instruct-2507 --output-dir out
 ```

The script has several flags. The default values are those used in the CanItEdit paper.
For prototyping, you can use `--completion-limit 1` to generate only one completion per prompt.

## Executing Completions

We have a prebuilt x86-64 Docker/Podman container. (This is necessary because
the benchmark has very particular Python dependencies.))

```bash
podman pull ghcr.io/nuprl/canitedit
```

To run tests:

```bash
podman run --rm --network none --volume ./out:/data:rw ghcr.io/nuprl/canitedit --dir /data --output-dir /data
```

## Calculating Scores

Finally, you can retrieve the results by running the `pass_k.py` script pointing
it to the directory where the completions are saved.

For example, to retrieve the results from the previous step, you can run:

```bash
./pass_k.py ./out
```

You will be provided with a CSV-formatted table with the results, including `pass@1` and `ExcessCode` metrics.
You can provide a `-k` parameter to the script to change the value of `k` for the `pass@k` metric.

```
python pass_k.py -k 5 ./outputs
```