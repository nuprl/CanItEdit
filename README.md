# Can It Edit? Evaluating the Ability of Large Language Models to Follow Code Editing Instructions

CanItEdit is a benchmark for evaluating LLMs on instructional code editing, the task of 
updating a program given a natural language instruction. The benchmark contains 54
hand-crafted Python programs with `before` and `after` code blocks, 
two types of natural language instructions (descriptive and lazy), and a hidden test suite. 

This repository provides code for evaluating models on the benchmark, and the code to reproduce
EditPackFT and EditCoder, a dataset and a LLM built for instructional code editing.

The CanItEdit benchmark dataset, EditCoder model, and EditPackFT dataset can be found on HuggingFace:
- CanItEdit: link
- EditCoder: link
- EditPackFT: link

# TODOs
fix docker
add multiple
add training code for EditCoder
add requirements.txt
add script for editeval
write tutorial for running CanItEdit
