import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === Lets Use Excel file directly ===
file_path = os.path.join(os.getcwd(), "before_fault_data.xlsx")

# skip header row
df = pd.read_excel(file_path, engine="openpyxl")
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

# Stripping whitespace
df.columns = df.columns.str.strip()

# Remove rows where all values from column index 2 onward have no change
df_cleaned = df[df.iloc[:, 2:].nunique(axis=1) > 1].copy()


# Convert columns C to L to numeric, coercing errors
df_cleaned.iloc[:, 2:] = df_cleaned.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")

# Calculate mean before fault (C to G -> index 2 to 6)
df_cleaned["Mean_Before_Fault"] = df_cleaned.iloc[:, 2:7].mean(axis=1)

# Calculate mean after fault (H to L -> index 7 to 11)
df_cleaned["Mean_After_Fault"] = df_cleaned.iloc[:, 7:12].mean(axis=1)

# New DataFrame with only field name, unit, and mean values
mean_df = df_cleaned.iloc[:, [0, 1]].copy()
mean_df["Mean_Before_Fault"] = df_cleaned["Mean_Before_Fault"]
mean_df["Mean_After_Fault"] = df_cleaned["Mean_After_Fault"]

# Replace NaN with "N/A" because I like it
mean_df = mean_df.fillna("N/A")

# Reverse N/A back to NaN to do calculations
mean_df[["Mean_Before_Fault", "Mean_After_Fault"]] = mean_df[
    ["Mean_Before_Fault", "Mean_After_Fault"]
].replace("N/A", pd.NA)
mean_df[["Mean_Before_Fault", "Mean_After_Fault"]] = mean_df[
    ["Mean_Before_Fault", "Mean_After_Fault"]
].apply(pd.to_numeric, errors="coerce")

# Set change types: normal and zero to non-zero
mean_df["Change_Type"] = "Normal Change"
mean_df.loc[
    (mean_df["Mean_Before_Fault"] == 0) & (mean_df["Mean_After_Fault"] != 0),
    "Change_Type",
] = "Zero to Non-Zero Change"

# Calculate absolute percentage change
mean_df["Absolute_Percentage_Change"] = (
    abs(
        (mean_df["Mean_After_Fault"] - mean_df["Mean_Before_Fault"])
        / mean_df["Mean_Before_Fault"]
    )
    * 100
)
mean_df.loc[
    mean_df["Change_Type"] == "Zero to Non-Zero Change", "Absolute_Percentage_Change"
] = float("inf")

# Separate and sort
zero_to_non_zero_df = mean_df[mean_df["Change_Type"] == "Zero to Non-Zero Change"]
normal_change_df = mean_df[mean_df["Change_Type"] != "Zero to Non-Zero Change"]
normal_change_df = normal_change_df.sort_values(
    by="Absolute_Percentage_Change", ascending=False
)

# Combine final sorted DataFrame
sorted_df = pd.concat([zero_to_non_zero_df, normal_change_df], ignore_index=True)
sorted_df = sorted_df.fillna("N/A")

# === Plotting ===

# Prepare data for a grouped barplot, sorted by percentage change (descending)
plot_data = sorted_df.iloc[2:].copy()
plot_data = plot_data.replace("N/A", pd.NA)
plot_data = plot_data.dropna(
    subset=["Mean_Before_Fault", "Mean_After_Fault", "Absolute_Percentage_Change"]
)
plot_data = plot_data.sort_values(by="Absolute_Percentage_Change", ascending=False)

fields = plot_data.iloc[:, 0]  # Field Name
units = plot_data.iloc[:, 1]  # Unit

x_labels = [f"{name} ({unit})" for name, unit in zip(fields, units)]
mean_before = plot_data["Mean_Before_Fault"].astype(float).values
mean_after = plot_data["Mean_After_Fault"].astype(float).values

import numpy as np

x = np.arange(len(x_labels))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(max(12, len(x_labels) * 0.7), 8))

rects1 = ax.bar(
    x - width / 2, mean_before, width, label="Before Fault", color="skyblue"
)
rects2 = ax.bar(
    x + width / 2, mean_after, width, label="After Fault", color="steelblue"
)

ax.set_ylabel("Mean Value (log scale)")
ax.set_title("Mean Value Before and After Fault by Field (Sorted by % Change)")
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=90)
ax.legend()

# Use log scale for y-axis to handle large value differences
ax.set_yscale("log")

# Annotate bars with a small rotation and always show 0 if value is 0
for rects, values in zip([rects1, rects2], [mean_before, mean_after]):
    for rect, val in zip(rects, values):
        height = rect.get_height()
        display_val = f"{val:.2f}" if val != 0 else "0"
        # For log scale, set y position for zero values just above the axis minimum
        if val == 0:
            y_pos = ax.get_ylim()[0] * 1.01  # slightly above axis min
            va = "top"
            offset = 8  # move closer to axis
        else:
            y_pos = height
            va = "bottom"
            offset = 8
        ax.annotate(
            display_val,
            xy=(rect.get_x() + rect.get_width() / 2, y_pos),
            xytext=(0, offset),  # vertical offset
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
            rotation=45,
            rotation_mode="anchor",
        )

# Save the plot
fig.savefig("bargraph_result.png")

plt.tight_layout()
plt.show()
