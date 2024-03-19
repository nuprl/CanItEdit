# EditCoder

This directory contains code to fine-tune EditCoder, a DeepSeek-Coder-Base-6.7b model fine-tuned on
EditPackFT for instructional code editing.
The code to train the model on Commits2023FT will be released soon.

To run the training, a `./run.sh` script is provided; it requires a deepspeed config as an argument,
we provide two DeepSpeed configs: `./deepspeed.json` and `./deepspeed_offload.json`. The former
runs the training with all weights on the GPUs, while the latter offloads some weights to the CPU
to reduce GPU memory usage. The latter is recommended for training on a single GPU.

The training script assumes you have 8 GPUs available, and will use all of them. If you have fewer
GPUs, you can modify the `--nproc_per_node` argument in the `./run.sh` script. However, it is
important to note that to reproduce the results in the paper, you must have an effective batch size
of 32, which means that `num_gpus * batch_size * grad_accumulation_steps = 32`; we recommend
adjusting the grad_accumulation_steps argument in the `./run.sh` script to achieve this.

Another assumption is that your GPU supports bf16 training. If it does not, you can remove the
`--fp16` argument from the `./run.sh` script and the deepspeed config.

The `finetuning-harness` submodule needs to be initialized and updated to run the training script,
this can be done with `git submodule update --init`. You should also install the requirements
with `pip install -r ./finetuning-harness/requirements.txt`.
