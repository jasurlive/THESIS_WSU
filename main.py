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

# Determine grid size
num_cols = 10
num_rows = (len(sorted_df.iloc[2:]) // num_cols) + 1
aspect_ratio = max(num_rows, num_cols)

fig, axes = plt.subplots(
    num_rows, num_cols, figsize=(aspect_ratio * 3, aspect_ratio * 3)
)
fig.tight_layout(pad=5.0)
axes = axes.flatten()

for idx, (index, row) in enumerate(sorted_df.iloc[2:].iterrows()):
    field_name = row.iloc[0]
    unit = row.iloc[1]
    mean_before = row.iloc[2]
    mean_after = row.iloc[3]

    data = {"Field Name": ["Before", "After"], "Mean Value": [mean_before, mean_after]}

    plot_df = pd.DataFrame(data)

    ax = axes[idx]

    sns.set_theme(style="whitegrid")
    sns.barplot(
        data=plot_df,
        x="Field Name",
        y="Mean Value",
        hue="Field Name",
        palette="Blues",
        ax=ax,
        legend=False,
    )

    ax.set_title(f"{field_name} ({unit})", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel(f"{unit}", fontsize=12)

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            fontsize=12,
            color="black",
            xytext=(0, 9),
            textcoords="offset points",
        )

    ax.set_xticks([0, 1])
    ax.grid(True, linestyle="--", alpha=0.6)

# Hide any unused subplots
for idx in range(len(sorted_df.iloc[2:]), len(axes)):
    axes[idx].axis("off")

plt.tight_layout()
plt.show()
