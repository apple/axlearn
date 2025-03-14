#!/bin/bash

export PATH=../bin/:$PATH
source ../jaxmoe/bin/activate

process_file() {
    set -ex
    local file="$1"
    local dir=$(dirname "$file")
    local filename=$(basename "$file")

    # Check if the filename matches the pattern and has execution000000
    # if [[ $filename =~ pjit__train_step-process000000-executable*-device000000-execution000000.inputs.decomposed_hlo_snapshot ]]; then
    echo " ============================== Working on $dir =============================="
    
    local new_dir="${dir}/${filename%.*}"
    if [ -d "$new_dir" ]; then
        rm -rf "$new_dir"
    fi
    mkdir -p "$new_dir"
    cd "$new_dir" || return
    
    python3 ../bin/unpack.py "../$filename"
    neuronx-cc compile --framework=XLA --target=trn2 --verbose=35 \
        --pipeline verify model.hlo \
        --internal-max-instruction-limit=20000000 \
        --target=trn2 --internal-num-neuroncores-per-sengine=2 \
        --model-type transformer --no-internal-hlo-remat \
        --enable-mixed-precision-accumulation -O1 \
        --tensorizer-options=--enable-hoist-fsdp-collectives \
        --internal-hlo2tensorizer-options=--remat-rope \
        --auto-cast=none \
        --policy=1 --enable-saturate-infinity --tolerance 1 1e-5
}

export -f process_file

set -x
cd $1
find . -type f -name "pjit__train_step-process000000-executable*-device000000-execution000000.inputs.decomposed_hlo_snapshot" | xargs -P 1 -I {} bash -c 'process_file "$@"' _ {}
