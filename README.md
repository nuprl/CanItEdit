# Can It Edit? Evaluating the Ability of Large Language Models to Follow Code Editing Instructions

CanItEdit is a benchmark for evaluating LLMs on instructional code editing, the task of
updating a program given a natural language instruction. The benchmark contains 54
hand-crafted Python programs with `before` and `after` code blocks,
two types of natural language instructions (descriptive and lazy), and a hidden test suite.

See [our paper](https://federico.codes/assets/papers/canitedit.pdf) for more.

This repository provides code for evaluating models on the benchmark, and the code to reproduce
EditPackFT and EditCoder, a dataset and a LLM built for instructional code editing.

The CanItEdit benchmark dataset, EditCoder model, and EditPackFT dataset can be found on HuggingFace:

- CanItEdit: https://huggingface.co/datasets/nuprl/CanItEdit
- EditCoder: https://huggingface.co/nuprl/EditCoder-6.7b-v1
- EditPackFT: https://huggingface.co/datasets/nuprl/EditPackFT

## Cloning the repository
It is very important to clone this repository and initialize all submodule recursively.
This can be done with the following command:

```bash
git clone --recurse-submodules https://github.com/nuprl/CanItEdit
```

## Structure

- `./benchmark` contains the CanItEdit benchmark dataset and code for generating and evaluating completions
- `./editcoder` contains code to train an EditCoder model
- `./editpackft` contains code to reproduce the EditPackFT dataset
- `./requirements.txt` contains the requirements for running the code of this repository

## Citation
If you use this code or the CanItEdit benchmark, please cite our paper:

```
@misc{canitedit,
  title = {Can It Edit? Evaluating the Ability of Large Language Models to Follow Code Editing Instructions},
  author = {Federico Cassano and Luisa Li and Akul Sethi and Noah Shinn and Abby Brennan-Jones and Anton Lozhkov and Carolyn Anderson and Arjun Guha},
  year = {2023},
  month = dec,
}
```
