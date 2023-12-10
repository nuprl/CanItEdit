# need to give deepspeed config file as argument
if [ $# -eq 0 ]
  then
    echo "No arguments supplied. Please give deepspeed config file as argument"
    exit 1
fi
DS=$(realpath $1)
pushd ./finetuning-harness
python3 -m torch.distributed.launch \
        --nproc_per_node 8 \
        main.py \
        --deepspeed="$DS" \
        --model_path="deepseek-ai/deepseek-coder-6.7b-base" \
        --dataset_name="nuprl/EditPackFT" \
        --no_approx_tokens \
        --output_dir="./model_editcoder_7b" \
        --seq_length 4096 \
        --epochs 8 \
        --fa2 \
        --batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-5 \
        --num_warmup_steps 10 \
        --num_workers=$(expr $(nproc --all) - 4) \
        --no_fp16 \
        --bf16
popd
