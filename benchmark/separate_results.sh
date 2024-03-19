#!/bin/bash
# separates the lazy and descriptive instruction results
# ./separate_results.sh <results_dir> <lazy_results_dir> <descriptive_results_dir>
# if <lazy_results_dir> or <descriptive_results_dir> is omitted, the default is
# <results_dir>_lazy and <results_dir>_descriptive
if [ $# -lt 1 ]; then
    echo "Usage: ./separate_results.sh <results_dir> <lazy_results_dir> <descriptive_results_dir>"
    exit 1
fi

results_dir=$1
lazy_results_dir=${2:-"$(realpath $results_dir)_lazy"}
descriptive_results_dir=${3:-"$(realpath $results_dir)_descriptive"}
echo "Separating results in $results_dir into:"
echo "  $lazy_results_dir"
echo "  $descriptive_results_dir"

# check if results_dir exists
if [ ! -d $results_dir ]; then
    echo "Error: $results_dir does not exist"
    exit 1
fi

mkdir -p $lazy_results_dir
mkdir -p $descriptive_results_dir

# separate the results
for file in $results_dir/*; do
  if [[ $file == *"instruction_descriptive"* ]]; then
    cp $file $descriptive_results_dir
  elif [[ $file == *"instruction_lazy"* ]]; then
    cp $file $lazy_results_dir
  fi
done
