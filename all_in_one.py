import glob
import os
import pandas as pd
import numpy as np  # <-- Add this import at the top
import matplotlib.pyplot as plt  # <-- Add this line


def is_excel_file_valid(file_path):
    # Check if file is a valid Excel file by extension and by reading a small part
    try:
        if file_path.endswith(".xls"):
            # Try reading with xlrd, but catch errors if file is not a real .xls
            import xlrd

            with xlrd.open_workbook(file_path):
                return True
        else:
            # Try reading with openpyxl for .xlsx
            import openpyxl

            openpyxl.load_workbook(file_path, read_only=True)
            return True
    except Exception:
        return False


def process_excel_file(file_path, output_dir):
    # Choose engine based on file extension
    if file_path.endswith(".xls"):
        engine = "xlrd"
    else:
        engine = "openpyxl"
    # Validate file before processing
    if not is_excel_file_valid(file_path):
        print(f"Skipping invalid or corrupt Excel file: {file_path}")
        return
    try:
        xls = pd.ExcelFile(file_path, engine=engine)
    except Exception as e:
        print(f"Error opening {file_path}: {e}")
        return
    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
        except Exception as e:
            print(f"Error reading sheet {sheet_name} in {file_path}: {e}")
            continue
        if df.empty or df.shape[1] < 12:
            continue  # skip empty or invalid sheets
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        df.columns = df.columns.str.strip()
        df_cleaned = df[df.iloc[:, 2:].nunique(axis=1) > 1].copy()
        df_cleaned.iloc[:, 2:] = df_cleaned.iloc[:, 2:].apply(
            pd.to_numeric, errors="coerce"
        )
        df_cleaned["Mean_Before_Fault"] = df_cleaned.iloc[:, 2:7].mean(axis=1)
        df_cleaned["Mean_After_Fault"] = df_cleaned.iloc[:, 7:12].mean(axis=1)
        mean_df = df_cleaned.iloc[:, [0, 1]].copy()
        mean_df["Mean_Before_Fault"] = df_cleaned["Mean_Before_Fault"]
        mean_df["Mean_After_Fault"] = df_cleaned["Mean_After_Fault"]
        mean_df = mean_df.fillna("N/A")
        mean_df[["Mean_Before_Fault", "Mean_After_Fault"]] = mean_df[
            ["Mean_Before_Fault", "Mean_After_Fault"]
        ].replace("N/A", pd.NA)
        mean_df[["Mean_Before_Fault", "Mean_After_Fault"]] = mean_df[
            ["Mean_Before_Fault", "Mean_After_Fault"]
        ].apply(pd.to_numeric, errors="coerce")
        mean_df["Change_Type"] = "Normal Change"
        mean_df.loc[
            (mean_df["Mean_Before_Fault"] == 0) & (mean_df["Mean_After_Fault"] != 0),
            "Change_Type",
        ] = "Zero to Non-Zero Change"
        mean_df["Absolute_Percentage_Change"] = (
            abs(
                (mean_df["Mean_After_Fault"] - mean_df["Mean_Before_Fault"])
                / mean_df["Mean_Before_Fault"]
            )
            * 100
        )
        mean_df.loc[
            mean_df["Change_Type"] == "Zero to Non-Zero Change",
            "Absolute_Percentage_Change",
        ] = float("inf")
        zero_to_non_zero_df = mean_df[
            mean_df["Change_Type"] == "Zero to Non-Zero Change"
        ]
        normal_change_df = mean_df[mean_df["Change_Type"] != "Zero to Non-Zero Change"]
        normal_change_df = normal_change_df.sort_values(
            by="Absolute_Percentage_Change", ascending=False
        )
        sorted_df = pd.concat(
            [zero_to_non_zero_df, normal_change_df], ignore_index=True
        )
        sorted_df = sorted_df.fillna("N/A")

        # === FILTER: 15% Threshold ===
        # Fix: Ensure Absolute_Percentage_Change is numeric for filtering
        sorted_df["Absolute_Percentage_Change_num"] = pd.to_numeric(
            sorted_df["Absolute_Percentage_Change"], errors="coerce"
        )
        threshold = 15
        filtered_df = sorted_df[
            (sorted_df["Absolute_Percentage_Change_num"] >= threshold)
            | (sorted_df["Absolute_Percentage_Change_num"] == float("inf"))
        ].copy()
        filtered_df = filtered_df.drop(columns=["Absolute_Percentage_Change_num"])

        plot_data = filtered_df.iloc[2:].copy()
        plot_data = plot_data.replace("N/A", pd.NA)
        plot_data = plot_data.dropna(
            subset=[
                "Mean_Before_Fault",
                "Mean_After_Fault",
                "Absolute_Percentage_Change",
            ]
        )
        # Drop out NUM and N-m units (add other values as needed)
        plot_data = plot_data[
            ~plot_data.iloc[:, 1].isin(["NUM", "N-m", "Deg", "deg", "Bar"])
        ]
        plot_data = plot_data.sort_values(
            by="Absolute_Percentage_Change", ascending=False
        )
        fields = plot_data.iloc[:, 0]
        units = plot_data.iloc[:, 1]
        x_labels = [f"{name} ({unit})" for name, unit in zip(fields, units)]
        mean_before = plot_data["Mean_Before_Fault"].astype(float).values
        mean_after = plot_data["Mean_After_Fault"].astype(float).values
        x = np.arange(len(x_labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(12, len(x_labels) * 0.7), 8))
        rects1 = ax.bar(
            x - width / 2, mean_before, width, label="Before Fault", color="skyblue"
        )
        rects2 = ax.bar(
            x + width / 2, mean_after, width, label="After Fault", color="steelblue"
        )
        ax.set_ylabel("Mean Value (log scale)")
        # Set plot title as Excel file name only (without extension)
        excel_file_name = os.path.splitext(os.path.basename(file_path))[0]
        ax.set_title(f"Mean Value Before and After Fault by Field\n{excel_file_name}")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=90)
        ax.legend()
        ax.set_yscale("log")
        for rects, values in zip([rects1, rects2], [mean_before, mean_after]):
            for rect, val in zip(rects, values):
                height = rect.get_height()
                display_val = f"{val:.2f}" if val != 0 else "0"
                if val == 0:
                    y_pos = ax.get_ylim()[0] * 1.01
                    va = "top"
                    offset = 8
                else:
                    y_pos = height
                    va = "bottom"
                    offset = 8
                ax.annotate(
                    display_val,
                    xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center",
                    va=va,
                    fontsize=8,
                    rotation=45,
                    rotation_mode="anchor",
                )
        plt.tight_layout()
        # Output file name: file_sheet.png
        safe_sheet = "".join([c if c.isalnum() else "_" for c in sheet_name])
        base = os.path.splitext(os.path.basename(file_path))[0]
        out_name = f"{base}_{safe_sheet}.png"
        out_path = os.path.join(output_dir, out_name)
        fig.savefig(out_path)
        plt.close(fig)


# Create output directory if not exists
output_dir = os.path.join(os.getcwd(), "dataset_bargraph_results")
os.makedirs(output_dir, exist_ok=True)

# Process all Excel files in both folders
folders = ["dataset/after_fault", "dataset/before_fault"]
for folder in folders:
    folder_path = os.path.join(os.getcwd(), folder)
    for ext in ("*.xlsx", "*.xls"):
        for file_path in glob.glob(os.path.join(folder_path, ext)):
            process_excel_file(file_path, output_dir)

print(f"All images saved to {output_dir}")
