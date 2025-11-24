import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

sns.set_theme(style="whitegrid")

# Definir paleta de cinza personalizada (escuro, médio, claro)
CUSTOM_GRAY_PALETTE = ["#2b2b2b", "#7f7f7f", "#5e5e5e"]

# -----------------------------------------------------------
# 1. Carregar um único arquivo CSV
# -----------------------------------------------------------

def load_dataset(path):
    df = pd.read_csv(path)
    if 'cv' in df.columns:
        df = df.drop(columns=['cv'])
    return df

# -----------------------------------------------------------
# 2. Estatísticas descritivas gerais
# -----------------------------------------------------------

def descriptive_statistics(df):
    return df.describe()


# -----------------------------------------------------------
# 3. Estatísticas agrupadas
# -----------------------------------------------------------

def grouped_statistics(df):
    numeric_df = df.select_dtypes(include=["number"])
    by_comp = numeric_df.groupby(df["competence"]).agg(["mean", "median", "std"])
    by_dist = numeric_df.groupby(df["distance"]).agg(["mean", "median", "std"])
    by_both = numeric_df.groupby([df["competence"], df["distance"]]).agg(["mean", "median", "std"])
    return by_comp, by_dist, by_both


# -----------------------------------------------------------
# 4. Gráficos com matplotlib + seaborn
# -----------------------------------------------------------

def plot_boxplot_mean_by_competence(df, output_dir, dataset_name):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="competence", y="mean", palette=CUSTOM_GRAY_PALETTE)
    plt.title(f"Mean Distribution by Competence ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_boxplot_mean_by_competence.png")
    plt.savefig(path)
    plt.close()


def plot_boxplot_mean_by_distance(df, output_dir, dataset_name):
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x="distance", y="mean", palette=CUSTOM_GRAY_PALETTE)
    plt.xticks(rotation=45)
    plt.title(f"Mean Distribution by Distance ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_boxplot_mean_by_distance.png")
    plt.savefig(path)
    plt.close()


def plot_bar_mean_by_distance_competence(df, output_dir, dataset_name):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="distance", y="mean", hue="competence", palette=CUSTOM_GRAY_PALETTE)
    plt.xticks(rotation=45)
    plt.title(f"Mean by Distance and Competence ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_bar_mean_by_distance_competence.png")
    plt.savefig(path)
    plt.close()


def plot_boxplot_median_by_competence(df, output_dir, dataset_name):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="competence", y="median", palette=CUSTOM_GRAY_PALETTE)
    plt.title(f"Median Distribution by Competence ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_boxplot_median_by_competence.png")
    plt.savefig(path)
    plt.close()


def plot_boxplot_median_by_distance(df, output_dir, dataset_name):
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x="distance", y="median", palette=CUSTOM_GRAY_PALETTE)
    plt.xticks(rotation=45)
    plt.title(f"Median Distribution by Distance ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_boxplot_median_by_distance.png")
    plt.savefig(path)
    plt.close()


def plot_bar_median_by_distance_competence(df, output_dir, dataset_name):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="distance", y="median", hue="competence", palette=CUSTOM_GRAY_PALETTE)
    plt.xticks(rotation=45)
    plt.title(f"Median by Distance and Competence ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_bar_median_by_distance_competence.png")
    plt.savefig(path)
    plt.close()


# ----------- NEW PLOTS USING DURATION ----------------

def plot_boxplot_duration_by_competence(df, output_dir, dataset_name):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="competence", y="duration", palette=CUSTOM_GRAY_PALETTE)
    plt.title(f"Duration Distribution by Competence ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_boxplot_duration_by_competence.png")
    plt.savefig(path)
    plt.close()


def plot_boxplot_duration_by_distance(df, output_dir, dataset_name):
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x="distance", y="duration", palette=CUSTOM_GRAY_PALETTE)
    plt.xticks(rotation=45)
    plt.title(f"Duration Distribution by Distance ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_boxplot_duration_by_distance.png")
    plt.savefig(path)
    plt.close()


def plot_bar_duration_by_distance_competence(df, output_dir, dataset_name):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="distance", y="duration", hue="competence", palette=CUSTOM_GRAY_PALETTE)
    plt.xticks(rotation=45)
    plt.title(f"Duration by Distance and Competence ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_bar_duration_by_distance_competence.png")
    plt.savefig(path)
    plt.close()


# ----------- NEW PLOTS USING STD ----------------

def plot_boxplot_std_by_competence(df, output_dir, dataset_name):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="competence", y="std", palette=CUSTOM_GRAY_PALETTE)
    plt.title(f"Std Distribution by Competence ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_boxplot_std_by_competence.png")
    plt.savefig(path)
    plt.close()


def plot_boxplot_std_by_distance(df, output_dir, dataset_name):
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x="distance", y="std", palette=CUSTOM_GRAY_PALETTE)
    plt.xticks(rotation=45)
    plt.title(f"Std Distribution by Distance ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_boxplot_std_by_distance.png")
    plt.savefig(path)
    plt.close()


def plot_bar_std_by_distance_competence(df, output_dir, dataset_name):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="distance", y="std", hue="competence", palette=CUSTOM_GRAY_PALETTE)
    plt.xticks(rotation=45)
    plt.title(f"Std by Distance and Competence ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_bar_std_by_distance_competence.png")
    plt.savefig(path)
    plt.close()
# -----------------------------------------------------------
# 5. Main analysis function
# -----------------------------------------------------------

def analyze_dataset_by_group(path, group_column, output_dir="analysis_outputs"):

    os.makedirs(output_dir, exist_ok=True)

    df = load_dataset(path)

    # Split DataFrame by group
    grouped = df.groupby(group_column)

    for group_name, group_df in grouped:
        group_output_dir = os.path.join(output_dir, str(group_name))
        os.makedirs(group_output_dir, exist_ok=True)
        # 1. Statistics
        desc = descriptive_statistics(group_df)
        by_comp, by_dist, by_both = grouped_statistics(group_df)

        # 2. Save CSVs with results
        desc.to_csv(os.path.join(group_output_dir, "describe.csv"))
        by_comp.to_csv(os.path.join(group_output_dir, "group_by_competence.csv"))
        by_dist.to_csv(os.path.join(group_output_dir, "group_by_distance.csv"))
        by_both.to_csv(os.path.join(group_output_dir, "group_by_competence_distance.csv"))

        # 3. Correlation
        corr = group_df[["mean", "median", "std", "duration"]].corr()
        corr.to_csv(os.path.join(group_output_dir, "correlation.csv"))

        # 4. Plots
        plot_boxplot_mean_by_competence(group_df, group_output_dir, group_name)
        plot_boxplot_mean_by_distance(group_df, group_output_dir, group_name)
        plot_bar_mean_by_distance_competence(group_df, group_output_dir, group_name)

        # 5. New plots using duration
        plot_boxplot_duration_by_competence(group_df, group_output_dir, group_name)
        plot_boxplot_duration_by_distance(group_df, group_output_dir, group_name)
        plot_bar_duration_by_distance_competence(group_df, group_output_dir, group_name)

        # 6. New plots using median
        plot_boxplot_median_by_competence(group_df, group_output_dir, group_name)
        plot_boxplot_median_by_distance(group_df, group_output_dir, group_name)
        plot_bar_median_by_distance_competence(group_df, group_output_dir, group_name)

        # 7. New plots using std
        plot_boxplot_std_by_competence(group_df, group_output_dir, group_name)
        plot_boxplot_std_by_distance(group_df, group_output_dir, group_name)
        plot_bar_std_by_distance_competence(group_df, group_output_dir, group_name)

        # -----------------------------------------------------------
        # 8. Identify best and worst (competence, distance) pairs
        # -----------------------------------------------------------

        metrics = ["mean", "median", "std", "duration"]

        best_worst_rows = []

        for metric in metrics:
            # Best = lowest value
            best_idx = group_df[metric].idxmin()
            best_row = group_df.loc[best_idx]

            # Worst = highest value
            worst_idx = group_df[metric].idxmax()
            worst_row = group_df.loc[worst_idx]

            best_worst_rows.append({
                "metric": metric,
                "type": "best",
                "competence": best_row["competence"],
                "distance": best_row["distance"],
                "value": best_row[metric]
            })

            best_worst_rows.append({
                "metric": metric,
                "type": "worst",
                "competence": worst_row["competence"],
                "distance": worst_row["distance"],
                "value": worst_row[metric]
            })

        best_worst_df = pd.DataFrame(best_worst_rows)

        # Save final CSV
        best_worst_df.to_csv(os.path.join(group_output_dir, "best_worst_pairs.csv"), index=False)

        print(f"\n✅ Analysis completed for group '{group_name}'! Results saved in: {group_output_dir}")



# # Remove columns 'fit_time' and 'predict_time' from 'metrics_results_abalone.csv'
# abalone_df = pd.read_csv("metrics_results_abalone.csv")
# abalone_df = abalone_df.drop(columns=["fit_time", "predict_time"], errors="ignore")
# abalone_df.to_csv("metrics_results_abalone.csv", index=False)

# Join all 'metrics_results*.csv' files in the current folder
all_files = glob.glob("metrics_results*.csv")
dfs = [pd.read_csv(f) for f in all_files]
combined_df = pd.concat(dfs, ignore_index=True)

combined_df = combined_df.sort_values(by=["dataset", "distance"])
if 'cv' in combined_df.columns:
        combined_df = combined_df.drop(columns=['cv'])

# Remove duplicate rows
combined_df = combined_df.drop_duplicates()

# Save to CSV
combined_df = combined_df.round(4)
combined_df.to_csv("results.csv", index=False)

# Save to LaTeX
with open("results.tex", "w") as f:
    latex_str = combined_df.to_latex(index=False)
    latex_str = latex_str.replace('_', '\\_')
    f.write(latex_str)
    

analyze_dataset_by_group("results.csv", group_column="dataset")

# Load the CSV generated with the best pairs
# Load all best_worst_pairs CSVs inside analysis_outputs subfolders
best_worst_files = glob.glob("analysis_outputs/*/best_worst_pairs.csv")
dfs = [pd.read_csv(f) for f in best_worst_files]
best_worst_df = pd.concat(dfs, ignore_index=True)

# Filter only "best"
best_df = best_worst_df[best_worst_df["type"] == "best"]

metrics = ["mean", "median", "std", "duration"]

COMPETENCE_REGION_LIST = ['knn', 'cluster', 'output_profiles']
DISTANCE_HEURISTICS_LIST = [
    "Braycurtis", "Canberra", "Chebyshev", "Cityblock",
    "Cosine", "Euclidean", "Minkowski", "Sqeuclidean"
]

for metric in metrics:
    metric_best_df = best_df[best_df["metric"] == metric]

    # 1. Distribution of times each distance appears as best
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=metric_best_df,
        x="distance",
        order=DISTANCE_HEURISTICS_LIST,
        palette=CUSTOM_GRAY_PALETTE
    )
    plt.title(f"Frequency of Distance as Best ({metric})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"best_distance_distribution_{metric}.png")
    plt.close()

    # 2. Distribution of times each competence appears as best
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=metric_best_df,
        x="competence",
        order=COMPETENCE_REGION_LIST,
        palette=CUSTOM_GRAY_PALETTE
    )
    plt.title(f"Frequency of Competence as Best ({metric})")
    plt.tight_layout()
    plt.savefig(f"best_competence_distribution_{metric}.png")
    plt.close()

    # 3. Distribution of frequency of (competence, distance) pairs as best
    plt.figure(figsize=(12, 8))
    # Generate all possible pairs
    all_pairs = pd.MultiIndex.from_product(
        [COMPETENCE_REGION_LIST, DISTANCE_HEURISTICS_LIST],
        names=["competence", "distance"]
    ).to_frame(index=False)
    pair_counts = metric_best_df.groupby(["competence", "distance"]).size().reset_index(name="count")
    pair_counts = all_pairs.merge(pair_counts, on=["competence", "distance"], how="left").fillna(0)
    pair_counts["count"] = pair_counts["count"].astype(int)
    pair_counts = pair_counts.sort_values("count", ascending=False)
    pair_counts["pair_label"] = pair_counts.apply(lambda x: f"{x['competence']} | {x['distance']}", axis=1)
    sns.barplot(
        data=pair_counts,
        x="pair_label",
        y="count",
        palette=CUSTOM_GRAY_PALETTE
    )
    plt.title(f"Frequency of (Competence, Distance) Pairs as Best ({metric})")
    plt.xlabel("Pair (Competence | Distance)")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"best_pair_distribution_{metric}.png")
    plt.close()


metrics_to_plot = ["mean", "median", "std"]

# Best distance
distance_freq = (
    best_df[best_df["metric"].isin(metrics_to_plot)]
    .groupby(["distance", "metric"]).size().reset_index(name="count")
)
plt.figure(figsize=(10, 6))
sns.barplot(
    data=distance_freq,
    x="distance",
    y="count",
    hue="metric",
    order=DISTANCE_HEURISTICS_LIST,
    palette=CUSTOM_GRAY_PALETTE
)
plt.title("Frequency of Distance as Best (Mean, Median, Std)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("best_distance_distribution_combined.png")
plt.close()

# Best competence
competence_freq = (
    best_df[best_df["metric"].isin(metrics_to_plot)]
    .groupby(["competence", "metric"]).size().reset_index(name="count")
)
plt.figure(figsize=(8, 5))
sns.barplot(
    data=competence_freq,
    x="competence",
    y="count",
    hue="metric",
    order=COMPETENCE_REGION_LIST,
    palette=CUSTOM_GRAY_PALETTE
)
plt.title("Frequency of Competence as Best (Mean, Median, Std)")
plt.tight_layout()
plt.savefig("best_competence_distribution_combined.png")
plt.close()

# Best pair (competence, distance)
pair_freq = (
    best_df[best_df["metric"].isin(metrics_to_plot)]
    .groupby(["competence", "distance", "metric"]).size().reset_index(name="count")
)
pair_freq["pair_label"] = pair_freq.apply(lambda x: f"{x['competence']} | {x['distance']}", axis=1)
plt.figure(figsize=(12, 8))
sns.barplot(
    data=pair_freq,
    x="pair_label",
    y="count",
    hue="metric",
    palette=CUSTOM_GRAY_PALETTE
)
plt.title("Frequency of (Competence, Distance) Pairs as Best (Mean, Median, Std)")
plt.xlabel("Pair (Competence | Distance)")
plt.ylabel("Frequency")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("best_pair_distribution_combined.png")
plt.close()

# Worst distance
worst_distance_freq = (
    best_worst_df[
        (best_worst_df["type"] == "worst") &
        (best_worst_df["metric"].isin(metrics_to_plot))
    ]
    .groupby(["distance", "metric"]).size().reset_index(name="count")
)
plt.figure(figsize=(10, 6))
sns.barplot(
    data=worst_distance_freq,
    x="distance",
    y="count",
    hue="metric",
    order=DISTANCE_HEURISTICS_LIST,
    palette=CUSTOM_GRAY_PALETTE
)
plt.title("Frequency of Distance as Worst (Mean, Median, Std)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("worst_distance_distribution_combined.png")
plt.close()

# Worst competence
worst_competence_freq = (
    best_worst_df[
        (best_worst_df["type"] == "worst") &
        (best_worst_df["metric"].isin(metrics_to_plot))
    ]
    .groupby(["competence", "metric"]).size().reset_index(name="count")
)
plt.figure(figsize=(8, 5))
sns.barplot(
    data=worst_competence_freq,
    x="competence",
    y="count",
    hue="metric",
    order=COMPETENCE_REGION_LIST,
    palette=CUSTOM_GRAY_PALETTE
)
plt.title("Frequency of Competence as Worst (Mean, Median, Std)")
plt.tight_layout()
plt.savefig("worst_competence_distribution_combined.png")
plt.close()

# Worst pair (competence, distance)
worst_pair_freq = (
    best_worst_df[
        (best_worst_df["type"] == "worst") &
        (best_worst_df["metric"].isin(metrics_to_plot))
    ]
    .groupby(["competence", "distance", "metric"]).size().reset_index(name="count")
)
worst_pair_freq["pair_label"] = worst_pair_freq.apply(lambda x: f"{x['competence']} | {x['distance']}", axis=1)
plt.figure(figsize=(12, 8))
sns.barplot(
    data=worst_pair_freq,
    x="pair_label",
    y="count",
    hue="metric",
    palette=CUSTOM_GRAY_PALETTE
)
plt.title("Frequency of (Competence, Distance) Pairs as Worst (Mean, Median, Std)")
plt.xlabel("Pair (Competence | Distance)")
plt.ylabel("Frequency")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("worst_pair_distribution_combined.png")
plt.close()

# Worst distance (duration)
worst_duration_distance_freq = (
    best_worst_df[
        (best_worst_df["type"] == "worst") &
        (best_worst_df["metric"] == "duration")
    ]
    .groupby(["distance"]).size().reset_index(name="count")
)
plt.figure(figsize=(10, 6))
sns.barplot(
    data=worst_duration_distance_freq,
    x="distance",
    y="count",
    order=DISTANCE_HEURISTICS_LIST,
    palette=CUSTOM_GRAY_PALETTE
)
plt.title("Frequency of Distance as Worst (Duration)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("worst_distance_distribution_duration.png")
plt.close()

# Worst competence (duration)
worst_duration_competence_freq = (
    best_worst_df[
        (best_worst_df["type"] == "worst") &
        (best_worst_df["metric"] == "duration")
    ]
    .groupby(["competence"]).size().reset_index(name="count")
)
plt.figure(figsize=(8, 5))
sns.barplot(
    data=worst_duration_competence_freq,
    x="competence",
    y="count",
    order=COMPETENCE_REGION_LIST,
    palette=CUSTOM_GRAY_PALETTE
)
plt.title("Frequency of Competence as Worst (Duration)")
plt.tight_layout()
plt.savefig("worst_competence_distribution_duration.png")
plt.close()

# Worst pair (competence, distance) for duration
worst_duration_pair_freq = (
    best_worst_df[
        (best_worst_df["type"] == "worst") &
        (best_worst_df["metric"] == "duration")
    ]
    .groupby(["competence", "distance"]).size().reset_index(name="count")
)
worst_duration_pair_freq["pair_label"] = worst_duration_pair_freq.apply(lambda x: f"{x['competence']} | {x['distance']}", axis=1)
plt.figure(figsize=(12, 8))
sns.barplot(
    data=worst_duration_pair_freq,
    x="pair_label",
    y="count",
    palette=CUSTOM_GRAY_PALETTE
)
plt.title("Frequency of (Competence, Distance) Pairs as Worst (Duration)")
plt.xlabel("Pair (Competence | Distance)")
plt.ylabel("Frequency")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("worst_pair_distribution_duration.png")
plt.close()
