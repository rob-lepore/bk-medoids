arbic="/mnt/c/Users/rober/downloads/arbic/ARBic-main/ARBic/ARBic"

for f in 0 1 2; do
    filepath="ARBic_data/3.six_type/colconst/$f"
    if [[ -f "$filepath" ]]; then
        echo "Running ARBic on $filepath"
        "$arbic" -i "$filepath" -m 2 -o 6 -f 0
    else
        echo "File $filepath not found, skipping."
    fi
done