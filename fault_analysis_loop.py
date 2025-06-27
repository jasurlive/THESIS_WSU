import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# loop all the excel files through all the steps of manipulating the data

output_dir = os.path.join(os.getcwd(), "dataset_outputs")
os.makedirs(output_dir, exist_ok=True)

#  folders = ["dataset/after_fault", "dataset/before_fault"]
folders = ["dataset/before_fault"]
excel_files = []
for folder in folders:
    folder_path = os.path.join(os.getcwd(), folder)
    excel_files.extend(glob.glob(os.path.join(folder_path, "*.xlsx")))

print(f"Found {len(excel_files)} Excel files to process.")

for current_file in excel_files:
    print(f"Processing: {current_file}")

    # Step 1: Load data
    df = pd.read_excel(current_file, skiprows=3)

    # Step 2: Omit all rows where columns C-L have only string values
    df = df[df.iloc[:, 2:12].map(pd.api.types.is_number).any(axis=1)].reset_index(
        drop=True
    )

    # Step 3: Skip rows where before and after values are the same
    def before_after_equal(row):
        before = row.iloc[2:7].values
        after = row.iloc[7:12].values
        return (before == after).all()

    df = df[~df.apply(before_after_equal, axis=1)].reset_index(drop=True)

    # Step 3.1: Further cleaning of NaN, NUM unit values
    if "Units" in df.columns:
        df = df[~(df["Units"].isna() | (df["Units"] == "NUM"))].reset_index(drop=True)

    # Step 4: Calculate mean values
    df_means = df.iloc[:, :2].copy()
    df_means["before"] = df.iloc[:, 2:7].mean(axis=1)
    df_means["after"] = df.iloc[:, 7:12].mean(axis=1)

    # Step 5: Calculate absolute % changes
    before_nonzero = df_means["before"].replace(0, pd.NA)
    df_means["%_change"] = (
        (df_means["after"] - df_means["before"]).abs() / before_nonzero.abs()
    ) * 100

    # Step 6: Apply threshold
    threshold = 15
    df_means_filtered = df_means[df_means["%_change"] > threshold].reset_index(
        drop=True
    )

    # Step 7: Bargraph (Before vs After)
    labels = df_means_filtered[df_means_filtered.columns[0]]
    before = df_means_filtered["before"]
    after = df_means_filtered["after"]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width / 2, before, width, label="Before", color="#5b9fd6")
    rects2 = ax.bar(x + width / 2, after, width, label="After", color="#9e6389")
    ax.set_yscale("log")
    ax.set_ylabel("Mean Value (log scale)")
    ax.set_title("Comparison of value changes (Before vs After)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    bargraph_path = os.path.join(
        output_dir,
        f"{os.path.splitext(os.path.basename(current_file))[0]}_bargraph.png",
    )
    plt.savefig(bargraph_path)
    plt.close(fig)

    # Step 8: Line plot
    plt.figure(figsize=(12, 6))
    plt.plot(labels, before, marker="o", linestyle="-", label="Before", color="#00457e")
    plt.plot(labels, after, marker="o", linestyle="-", label="After", color="#ff6658")
    plt.yscale("log")
    plt.ylabel("Mean Value (log scale)")
    plt.title("Before vs After (Line Plot, log scale)")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    lineplot_path = os.path.join(
        output_dir,
        f"{os.path.splitext(os.path.basename(current_file))[0]}_lineplot.png",
    )
    plt.savefig(lineplot_path)
    plt.close()

    # Step 9: Percentage change horizontal bar plot
    pct_change = df_means_filtered["%_change"]
    plt.figure(figsize=(10, 6))
    bars = plt.barh(labels, pct_change, color="#00457dcf")
    plt.xlabel("Absolute % Change")
    plt.title("Absolute Percentage Change by Field")
    plt.tight_layout()
    for bar in bars:
        plt.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.2f}%",
            va="center",
            ha="left",
        )
    pctbar_path = os.path.join(
        output_dir,
        f"{os.path.splitext(os.path.basename(current_file))[0]}_pctchange.png",
    )
    plt.savefig(pctbar_path)
    plt.close()
