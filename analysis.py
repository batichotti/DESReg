import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

sns.set_theme(style="whitegrid")

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
    sns.boxplot(data=df, x="competence", y="mean")
    plt.title(f"Distribuição da Média por Competence ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_boxplot_mean_by_competence.png")
    plt.savefig(path)
    plt.close()


def plot_boxplot_mean_by_distance(df, output_dir, dataset_name):
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x="distance", y="mean")
    plt.xticks(rotation=45)
    plt.title(f"Distribuição da Média por Distance ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_boxplot_mean_by_distance.png")
    plt.savefig(path)
    plt.close()


def plot_bar_mean_by_distance_competence(df, output_dir, dataset_name):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="distance", y="mean", hue="competence")
    plt.xticks(rotation=45)
    plt.title(f"Média por Distance e Competence ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_bar_mean_by_distance_competence.png")
    plt.savefig(path)
    plt.close()


def plot_boxplot_median_by_competence(df, output_dir, dataset_name):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="competence", y="median")
    plt.title(f"Distribuição da Mediana por Competence ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_boxplot_median_by_competence.png")
    plt.savefig(path)
    plt.close()


def plot_boxplot_median_by_distance(df, output_dir, dataset_name):
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x="distance", y="median")
    plt.xticks(rotation=45)
    plt.title(f"Distribuição da Mediana por Distance ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_boxplot_median_by_distance.png")
    plt.savefig(path)
    plt.close()


def plot_bar_median_by_distance_competence(df, output_dir, dataset_name):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="distance", y="median", hue="competence")
    plt.xticks(rotation=45)
    plt.title(f"Mediana por Distance e Competence ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_bar_median_by_distance_competence.png")
    plt.savefig(path)
    plt.close()


# ----------- NOVOS GRÁFICOS USANDO DURATION ----------------

def plot_boxplot_duration_by_competence(df, output_dir, dataset_name):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="competence", y="duration")
    plt.title(f"Distribuição da Duration por Competence ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_boxplot_duration_by_competence.png")
    plt.savefig(path)
    plt.close()


def plot_boxplot_duration_by_distance(df, output_dir, dataset_name):
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x="distance", y="duration")
    plt.xticks(rotation=45)
    plt.title(f"Distribuição da Duration por Distance ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_boxplot_duration_by_distance.png")
    plt.savefig(path)
    plt.close()


def plot_bar_duration_by_distance_competence(df, output_dir, dataset_name):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="distance", y="duration", hue="competence")
    plt.xticks(rotation=45)
    plt.title(f"Duration por Distance e Competence ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_bar_duration_by_distance_competence.png")
    plt.savefig(path)
    plt.close()


# ----------- NOVOS GRÁFICOS USANDO STD ----------------

def plot_boxplot_std_by_competence(df, output_dir, dataset_name):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="competence", y="std")
    plt.title(f"Distribuição do Std por Competence ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_boxplot_std_by_competence.png")
    plt.savefig(path)
    plt.close()


def plot_boxplot_std_by_distance(df, output_dir, dataset_name):
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x="distance", y="std")
    plt.xticks(rotation=45)
    plt.title(f"Distribuição do Std por Distance ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_boxplot_std_by_distance.png")
    plt.savefig(path)
    plt.close()


def plot_bar_std_by_distance_competence(df, output_dir, dataset_name):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="distance", y="std", hue="competence")
    plt.xticks(rotation=45)
    plt.title(f"Std por Distance e Competence ({dataset_name})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_bar_std_by_distance_competence.png")
    plt.savefig(path)
    plt.close()
# -----------------------------------------------------------
# 5. Função principal de análise
# -----------------------------------------------------------

def analyze_dataset_by_group(path, group_column, output_dir="analysis_outputs"):

    os.makedirs(output_dir, exist_ok=True)

    df = load_dataset(path)

    # Dividir o DataFrame por grupo
    grouped = df.groupby(group_column)

    for group_name, group_df in grouped:
        group_output_dir = os.path.join(output_dir, str(group_name))
        os.makedirs(group_output_dir, exist_ok=True)
        # 1. Estatísticas
        desc = descriptive_statistics(group_df)
        by_comp, by_dist, by_both = grouped_statistics(group_df)

        # 2. Salvar CSVs com resultados
        desc.to_csv(os.path.join(group_output_dir, "describe.csv"))
        by_comp.to_csv(os.path.join(group_output_dir, "group_by_competence.csv"))
        by_dist.to_csv(os.path.join(group_output_dir, "group_by_distance.csv"))
        by_both.to_csv(os.path.join(group_output_dir, "group_by_competence_distance.csv"))

        # 3. Correlação
        corr = group_df[["mean", "median", "std", "duration"]].corr()
        corr.to_csv(os.path.join(group_output_dir, "correlation.csv"))

        # 4. Gráficos
        plot_boxplot_mean_by_competence(group_df, group_output_dir, group_name)
        plot_boxplot_mean_by_distance(group_df, group_output_dir, group_name)
        plot_bar_mean_by_distance_competence(group_df, group_output_dir, group_name)

        # 5. Novos gráficos usando duration
        plot_boxplot_duration_by_competence(group_df, group_output_dir, group_name)
        plot_boxplot_duration_by_distance(group_df, group_output_dir, group_name)
        plot_bar_duration_by_distance_competence(group_df, group_output_dir, group_name)

        # 6. Novos gráficos usando mediana
        plot_boxplot_median_by_competence(group_df, group_output_dir, group_name)
        plot_boxplot_median_by_distance(group_df, group_output_dir, group_name)
        plot_bar_median_by_distance_competence(group_df, group_output_dir, group_name)

        # 7. Novos gráficos usando std
        plot_boxplot_std_by_competence(group_df, group_output_dir, group_name)
        plot_boxplot_std_by_distance(group_df, group_output_dir, group_name)
        plot_bar_std_by_distance_competence(group_df, group_output_dir, group_name)

        # -----------------------------------------------------------
        # 8. Identificar melhores e piores pares de (competence, distance)
        # -----------------------------------------------------------

        metrics = ["mean", "median", "std", "duration"]

        best_worst_rows = []

        for metric in metrics:
            # Melhor = menor valor
            best_idx = group_df[metric].idxmin()
            best_row = group_df.loc[best_idx]

            # Pior = maior valor
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

        # Salvar CSV final
        best_worst_df.to_csv(os.path.join(group_output_dir, "best_worst_pairs.csv"), index=False)

        print(f"\n✅ Análise concluída para o grupo '{group_name}'! Resultados salvos em: {group_output_dir}")



# # Remover as colunas 'fit_time' e 'predict_time' do arquivo 'metrics_results_abalone.csv'
# abalone_df = pd.read_csv("metrics_results_abalone.csv")
# abalone_df = abalone_df.drop(columns=["fit_time", "predict_time"], errors="ignore")
# abalone_df.to_csv("metrics_results_abalone.csv", index=False)

# Juntar todos os arquivos 'metrics_results*.csv' da pasta atual
all_files = glob.glob("metrics_results*.csv")
dfs = [pd.read_csv(f) for f in all_files]
combined_df = pd.concat(dfs, ignore_index=True)

combined_df = combined_df.sort_values(by=["dataset", "distance"])
if 'cv' in combined_df.columns:
        combined_df = combined_df.drop(columns=['cv'])

# Remover linhas duplicadas
combined_df = combined_df.drop_duplicates()

# Salvar em CSV
combined_df = combined_df.round(4)
combined_df.to_csv("results.csv", index=False)

# Salvar em LaTeX
with open("results.tex", "w") as f:
    latex_str = combined_df.to_latex(index=False)
    latex_str = latex_str.replace('_', '\\_')
    f.write(latex_str)
    

analyze_dataset_by_group("results.csv", group_column="dataset")