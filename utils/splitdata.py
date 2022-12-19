import os
import sys
import numpy as np
import pandas as pd


def create_csv(path: str, output_path: str, max_files: int = 1) -> None:

    """Function to split the files to be processed in order to run them
    on multiple machines without overlapping."""

    assert (
        type(max_files) is int and max_files > 0
    ), "You must enter a int from the 1 to the N"

    # delete old .csv files
    for file in os.listdir(path):
        if file.endswith(".csv"):
            print(f"[Data] Removing the file: {os.path.join(path, file)}")
            os.remove(os.path.join(path, file))

    # check processed features
    if os.path.exists(output_path):
        proc_v = [v.split(".")[0] for v in os.listdir(output_path)]
        if len(proc_v) > 0:
            print(f"[Data] Already {len(proc_v)} files have been processed")
        entries = os.listdir(path)
        entries = [
            v.split(".")[0] for v in entries if v.split(".")[0] not in proc_v
        ]
    else:
        entries = os.listdir(path)
        entries = [v.split(".")[0] for v in entries]

    df = pd.DataFrame(entries)
    if max_files == 1:
        path_csv = path + "/videos_list.csv"
        df.to_csv(path_csv, index=False, header=False)
        print(f"[Data] Successfully created: {path_csv}")
    else:
        indices = np.array_split(df.index, max_files)

        output = {}
        for i in range(max_files):
            os.path.join
            path_csv = path + f"/videos_list_{i+1}.csv"
            data_csv = df.loc[indices[i]]
            output[i] = data_csv
            data_csv.to_csv(path_csv, index=False, header=False)
            print(f"[Data] Successfully created: {path_csv}")
    print("[Data] Execution completed")


if __name__ == "__main__":
    path_input = (
        sys.argv[1]
        if not sys.argv[1] is None
        else "/data/imeza/ActivityNet/Activity_Net_videos_rescale"
    )
    path_output = (
        sys.argv[2]
        if not sys.argv[2] is None
        else "/data/imeza/ActivityNet/AN_Features/x3d_s"
    )
    max_files = int(sys.argv[3]) if not sys.argv[3] is None else 1
    create_csv(path_input, path_output, max_files)
