if [ -z "$1" ]; then
    echo "Usage: evaluate_completions.sh <path to directory with completions>"
    exit 1
fi

docker run --rm --network none  --volume $1:/inputs:ro --volume $1:/outputs:rw edit-eval --dir /inputs --output-dir /outputs
