from bicgen import BicGen
import numpy as np
import json
import os

input_dir = "./gbic_output"
output_dir = "./datasets"

noise_levels = [0,0.1,0.2,0.3]
patterns = ["col_const", "row_const", "scale", "shift"]
n_ds = 25

for n_id, noise in enumerate(noise_levels):
    
    
    for pattern_fold in patterns:
        in_dir = f"{input_dir}/{pattern_fold}"
        out_dir = f"{output_dir}/{pattern_fold}/noise_{n_id}"
        os.makedirs(out_dir, exist_ok=True)

        output_meta_filename = f"{out_dir}/metadata.json"
        metadata_list = []

        for ds_id in range(n_ds):
                    
            input_filename = f"{in_dir}/ds_{ds_id}_bics.json"
            output_filename = f"{out_dir}/ds_{ds_id}.tsv"

            ds, rows, columns, _ = BicGen(gen_seed = ds_id, noise = noise*noise).from_gbic(input_filename)    
            
            np.savetxt(output_filename, ds, fmt="%f", delimiter="\t") 
            
            n_bics = len(rows)

            for i in range(n_bics):
                r_idx = np.where(rows[i] == True)[0]
                c_idx = np.where(columns[i] == True)[0]
                metadata_list.append({
                    "dataset_id": ds_id,
                    "bicluster_id": i,
                    "rows": list(int(r) for r in r_idx),
                    "columns": list(int(c) for c in c_idx),
                })
            
        with open(output_meta_filename, "w") as f:
            json.dump(metadata_list, f, indent=2)