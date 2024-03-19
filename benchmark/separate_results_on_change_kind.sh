#!/bin/bash
# separates the results based on the change kind ("adaptive", "corrective", "perfective")
# ./separate_results_on_change_kind.sh <results_dir>
if [ $# -lt 1 ]; then
    echo "Usage: ./separate_results_on_change_kind.sh <results_dir>"
    exit 1
fi

# will create folders <results_dir>_adaptive, <results_dir>_corrective, <results_dir>_perfective

dir=$1

ADAPTIVE=()
CORRECTIVE=()
PERFECTIVE=()

# check if jq is installed
if ! [ -x "$(command -v jq)" ]; then
  echo 'Error: jq is not installed.' >&2
  exit 1
fi

for file in $dir/*; do
  CHANGE_KIND=$(cat $file | gunzip | jq .taxonomy.change_kind) 
  # keep evolve/revise for backward compatibility
  if [ "$CHANGE_KIND" == "\"adaptive\"" ] || [ "$CHANGE_KIND" == "\"evolve\"" ]; then
    ADAPTIVE+=($file)
  elif [ "$CHANGE_KIND" == "\"corrective\"" ]; then
    CORRECTIVE+=($file)
  elif [ "$CHANGE_KIND" == "\"perfective\"" ] || [ "$CHANGE_KIND" == "\"revise\"" ]; then
    PERFECTIVE+=($file)
  else
    echo "Unknown change kind: $CHANGE_KIND"
  fi
done

# get dir name without trailing slash
dir=$(echo $dir | sed 's:/*$::')

echo "Adaptive: ${#ADAPTIVE[@]}"
echo "Corrective: ${#CORRECTIVE[@]}"
echo "Perfective: ${#PERFECTIVE[@]}"

if [ ${#ADAPTIVE[@]} -gt 0 ]; then
  OUT=$dir"_adaptive"
  echo "Moving to $OUT"
  mkdir -p $OUT
  for file in ${ADAPTIVE[@]}; do
    cp $file $OUT
  done
fi

if [ ${#CORRECTIVE[@]} -gt 0 ]; then
  OUT=$dir"_corrective"
  echo "Moving to $OUT"
  mkdir -p $OUT
  for file in ${CORRECTIVE[@]}; do
    cp $file $OUT
  done
fi

if [ ${#PERFECTIVE[@]} -gt 0 ]; then
  OUT=$dir"_perfective"
  echo "Moving to $OUT"
  mkdir -p $OUT
  for file in ${PERFECTIVE[@]}; do
    cp $file $OUT
  done
fi
