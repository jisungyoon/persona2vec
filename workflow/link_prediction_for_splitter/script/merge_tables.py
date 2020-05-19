import sys

import pandas as pd

input_files = sys.argv[1:-1]

out_path = sys.argv[-1]

dfs = []
for input_file in input_files:
    temp_df = pd.read_csv(input_file, sep="\t", header=None)
    dfs.append(temp_df)

df = pd.concat(dfs)
df.to_csv(out_path, index=None, sep="\t")
