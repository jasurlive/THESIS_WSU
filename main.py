import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  # plot

df = pd.read_csv("mean_values_sorted.csv")
df.columns = df.columns.str.strip()

num_cols = 10
num_rows = (len(df.iloc[2:]) // num_cols) + 1


aspect_ratio = max(num_rows, num_cols)


fig, axes = plt.subplots(
    num_rows, num_cols, figsize=(aspect_ratio * 3, aspect_ratio * 3)
)
fig.tight_layout(pad=5.0)
axes = axes.flatten()

for idx, (index, row) in enumerate(df.iloc[2:].iterrows()):
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

for idx in range(len(df.iloc[2:]), len(axes)):
    axes[idx].axis("off")

plt.tight_layout()
plt.show()
