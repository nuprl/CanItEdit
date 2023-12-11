# Can It Edit? Evaluating the Ability of Large Language Models to Follow Code Editing Instructions

CanItEdit is a benchmark for evaluating LLMs on instructional code editing, the task of
updating a program given a natural language instruction. The benchmark contains 54
hand-crafted Python programs with `before` and `after` code blocks,
two types of natural language instructions (descriptive and lazy), and a hidden test suite.

This repository provides code for evaluating models on the benchmark, and the code to reproduce
EditPackFT and EditCoder, a dataset and a LLM built for instructional code editing.

The CanItEdit benchmark dataset, EditCoder model, and EditPackFT dataset can be found on HuggingFace:

- CanItEdit: https://huggingface.co/nuprl/CanItEdit
- EditCoder: https://huggingface.co/nuprl/EditCoder-6.7b-v1
- EditPackFT: https://huggingface.co/datasets/nuprl/EditPackFT

## Structure

- `./benchmark` contains the CanItEdit benchmark dataset and code for generating and evaluating completions
- `./editcoder` contains code to train an EditCoder model
- `./editpackft` contains code to reproduce the EditPackFT dataset
- `./requirements.txt` contains the requirements for running the code of this repository


# TODOs

add script for editeval
write tutorial for running CanItEdit
