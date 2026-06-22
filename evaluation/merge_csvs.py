import glob
import os

import pandas as pd

RESULTS_DIR = "evaluation/results"


def merge_by_prefix(prefix: str) -> None:
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{prefix}_*.csv")))
    files = [path for path in files if not path.endswith(f"{prefix}_final.csv")]
    if not files:
        print(f"No {prefix} files found.")
        return

    merged = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    final_path = os.path.join(RESULTS_DIR, f"{prefix}_final.csv")
    merged.to_csv(final_path, index=False)
    print(f"Merged {prefix} saved to {final_path}")


if __name__ == "__main__":
    merge_by_prefix("summary")
    merge_by_prefix("detail")
    merge_by_prefix("human_eval")
