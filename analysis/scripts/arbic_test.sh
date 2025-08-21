#!/bin/bash

# List your five directories here
# dirs=("ARBic_data/3.six_type/row_const" "ARBic_data/3.six_type/scale" "ARBic_data/3.six_type/shift" "ARBic_data/3.six_type/shift_scale" "ARBic_data/3.six_type/trend")
dirs=("ARBic_data/3.six_type/scale" "ARBic_data/3.six_type/shift" "ARBic_data/3.six_type/shift_scale" "ARBic_data/3.six_type/trend")

# Path to ARBic executable
arbic="/mnt/c/Users/rober/downloads/arbic/ARBic-main/ARBic/ARBic"

# Loop through directories
for d in "${dirs[@]}"; do
    for f in 0 1 2 3; do
        filepath="$d/$f"
        if [[ -f "$filepath" ]]; then
            echo "Running ARBic on $filepath"
            "$arbic" -i "$filepath" -m 2 -o 6 -f 0
        else
            echo "File $filepath not found, skipping."
        fi
    done
done

