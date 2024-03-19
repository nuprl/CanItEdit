# EditPackFT

This directory contains the code for reproducing EditPackFT from CommitPackFT.
The `filter.py` script filters and cleans the code from CommitPackFT,
and the `format.py` script adds the training prompt to each example.

A pre-built EditPackFT dataset can be found at: https://huggingface.co/datasets/nuprl/EditPackFT

The code and dataset for Commits2023FT will be released soon.

## Near-Deduplication

An additional step is used to remove near-duplicate examples from the dataset.
Once the dataset is built and formatted, we utilize the [code-dedup](https://github.com/cassanof/code-dedup)
framework to remove near-duplicate examples; this deduplication step utilizes
MinHash and Locality Sensitive Hashing (LSH) to identify near-duplicate examples and
discard them from the dataset.

The submodule `./code-dedup` contains a `minhash_dedup_hf.sh` script that can be used
to easily deduplicate a dataset; the script just requires you to specify the dataset
path on HuggingFace and the output path for the deduplicated dataset.
