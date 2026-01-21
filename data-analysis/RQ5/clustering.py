import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors

feature_cols = [
    "mean_cpu_usage_percent",
    "mean_ram_used_mb",
    "time_per_1000_steps_sec"
]

RANDOM_STATE = 42
def print_terminal_line():
    print("-" * os.get_terminal_size()[0])
    
def print_terminal_half_line():
    print("- " * (os.get_terminal_size()[0]//2))
    
def create_feature_data_frame(df : pd.DataFrame) -> pd.DataFrame:
    assert (df["max_steps"] > 0).all(), "Runs with 0 steps found"
    df["time_per_1000_steps_sec"] = (df["total_duration_sec"]/df["max_steps"]*1000)
    return df[feature_cols]
    
def get_sanitized_mask(X : pd.DataFrame):
    sanitized_mask =(
        (X["mean_cpu_usage_percent"] >= 0) &
        (X["mean_ram_used_mb"] >= 0) &
        (X["time_per_1000_steps_sec"] > 0)
    )
    return sanitized_mask

def get_subsample_mask(sanitized_df : pd.DataFrame, instances_per_machine_game_pair = 5):
    global RANDOM_STATE
    subsampled = sanitized_df.groupby(["cpu", "game_name", "training_type"], group_keys=False).apply(
        lambda x: x.sample(
            n=min(len(x), instances_per_machine_game_pair),
            random_state=RANDOM_STATE
        )
    )
    used_run_ids = set(subsampled["RunID"])
    return sanitized_df["RunID"].isin(used_run_ids)

def get_sanitized_and_subsampled_features_and_org_df(df : pd.DataFrame, subsample = True, subsample_cap = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = create_feature_data_frame(df)
    sanitized_mask = get_sanitized_mask(X)
    X_sanitized = X[sanitized_mask].copy()
    df_sanitized = df[sanitized_mask].copy()
    if subsample:
        subsample_mask = get_subsample_mask(df_sanitized, instances_per_machine_game_pair=subsample_cap)
        X_sanitized = X_sanitized[subsample_mask].copy()
        df_sanitized = df_sanitized[subsample_mask].copy()
    return X_sanitized, df_sanitized

def generate_cluster_summary_tables(X_sanitized,df_sanitized_with_clusters):
    cluster_summary = (
    X_sanitized
    .groupby("cluster")[feature_cols]
    .agg(["mean", "std", "min", "max"])
    )
    cpus = df_sanitized_with_clusters["cpu"].unique()
    print_terminal_line()
    print("Showing cluster summary")
    print(pd.Series(X_sanitized["cluster"]).value_counts())
    for feature in X_sanitized.columns:
        if(feature == "cluster"): continue
        print(f"\n{feature} summary:")
        print(cluster_summary[feature])
    print()
    cluster_training_type_view = pd.crosstab(df_sanitized_with_clusters["cluster"],df_sanitized_with_clusters["training_type"], normalize="index")
    print_terminal_half_line()
    print("Cluster counts")
    print(cluster_training_type_view)
    print()
    cluster_cpu_training_type_view = pd.crosstab(df_sanitized_with_clusters["cluster"], [df_sanitized_with_clusters[col] for col in ["cpu","training_type"]])
    for cpu in cpus:
        print_terminal_half_line()
        print(f"\nCluster summary for {cpu}")
        print(cluster_cpu_training_type_view[cpu])
    print("-" * os.get_terminal_size()[0])
    
def get_scaled_df_and_np(X_sanitized : pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    scaler = RobustScaler()
    X_log = X_sanitized.copy()
    X_log["time_per_1000_steps_sec"] = np.log1p(X_log["time_per_1000_steps_sec"])
    X_scaled_as_array = scaler.fit_transform(X_log)
    X_scaled = pd.DataFrame(
        X_scaled_as_array,
        columns=X_sanitized.columns,
        index = X_sanitized.index
    )
    return X_scaled, X_scaled_as_array

def show_clustered_3d_log(X_with_clusters, name) -> None:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    clusters = X_with_clusters["cluster"].unique()

    for cluster in clusters:
        mask = X_with_clusters["cluster"] == cluster
        ax.scatter(
            X_with_clusters.loc[mask, "mean_cpu_usage_percent"],
            X_with_clusters.loc[mask, "mean_ram_used_mb"],
            X_with_clusters.loc[mask, "time_per_1000_steps_sec"],
            label=f"Cluster {cluster}"
        )
    ax.set_xlabel("mean_cpu_usage_percent")
    ax.set_ylabel("mean_ram_used_mb")
    ax.set_zlabel("time_per_1000_steps_sec", labelpad=0)
    ax.legend(title="Clusters")
    plt.savefig(name)
    
def generate_by_cpu_3d_log(X, name) -> None:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    cpus = X["cpu"].unique()
    for cpu in cpus:
        mask = X["cpu"] == cpu
        ax.scatter(
            X.loc[mask, "mean_cpu_usage_percent"],
            X.loc[mask, "mean_ram_used_mb"],
            X.loc[mask, "time_per_1000_steps_sec"],
            label=f"CPU {cpu}"
        )
    ax.set_xlabel("mean_cpu_usage_percent")
    ax.set_ylabel("mean_ram_used_mb")
    ax.set_zlabel("time_per_1000_steps_sec", labelpad=0)
    ax.legend(title="CPUs")
    plt.savefig(name)
    # plt.show()

def kmeans_clustering(X_sanitized : pd.DataFrame, df_sanitized : pd.DataFrame, n_of_clusters = 4, ):
    global RANDOM_STATE
    df_sanitized_kmeans = df_sanitized.copy()
    X_sanitized_kmeans = X_sanitized.copy()
    X_scaled, X_scaled_as_array = get_scaled_df_and_np(X_sanitized_kmeans)
    kmeans = KMeans(n_clusters=n_of_clusters, init="k-means++", n_init="auto", random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(X_scaled_as_array)
    X_sanitized_kmeans["cluster"] = labels
    pd.Series(labels).value_counts()
    df_sanitized_kmeans["cluster"] = labels
    generate_cluster_summary_tables(X_sanitized_kmeans, df_sanitized_kmeans)
    show_clustered_3d_log(X_sanitized_kmeans, f"{n_of_clusters}_means_clustering_plot")
    
def dbscan_clustering(X_sanitized : pd.DataFrame, df_sanitized : pd.DataFrame, eps = 0.4):
    df_sanitized_dbscan = df_sanitized.copy()
    X_sanitized_dbscan = X_sanitized.copy()
    X_scaled_dbscan, X_scaled_as_array_dbscan = get_scaled_df_and_np(X_sanitized_dbscan)
    dbscan = DBSCAN(eps = eps)
    labels = dbscan.fit_predict(X_scaled_as_array_dbscan)
    X_sanitized_dbscan["cluster"] = labels
    df_sanitized_dbscan["cluster"] = labels
    pd.Series(labels).value_counts()
    generate_cluster_summary_tables(X_sanitized_dbscan, df_sanitized_dbscan)
    show_clustered_3d_log(X_sanitized_dbscan, f"dbscan_clustering_plot")

def kmeans_k_search(X_sanitized, df_sanitized, k_min = 2, k_max = 20):
    global RANDOM_STATE
    print_terminal_half_line()
    print(f"Silhouette scores for each k value in range {k_min} - {k_max}")
    k_range = range(k_min, k_max)

    inertias = []
    silhouette_scores = []
    
    X_scaled, X_scaled_as_array = get_scaled_df_and_np(X_sanitized)

    for k in k_range:
        # kmeans = KMeans(n_clusters=k, init="k-means++", n_init="auto", random_state=42)
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init="auto", random_state=RANDOM_STATE)
        labels = kmeans.fit_predict(X_scaled_as_array)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled_as_array, labels))
        
    results = pd.DataFrame({"k" : k_range, "inertia" : inertias, "silhouette_score" : silhouette_scores})

    print(results)

def plot_k_distance_graph(X, k = 5):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    distances = np.sort(distances[:, k-1])
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Points')
    plt.ylabel(f'{k}-th nearest neighbor distance')
    plt.title('K-distance Graph')
    plt.savefig(f"{k}_distance_plot")
    
def print_noise_values_on_epsilon_range(X_scaled_as_array_dbscan, eps_start, eps_end, eps_interval = 0.1):
    noise_dict = {}
    eps_curr = eps_start
    while(eps_curr <= eps_end):            
        dbscan = DBSCAN(eps = eps_curr, min_samples = 5)
        labels = dbscan.fit_predict(X_scaled_as_array_dbscan)
        noise_dict[eps_curr] = np.sum(labels == -1)/len(labels)
        eps_curr+=eps_interval
        
    print(f"Noise values on epsilon range {eps_start} - {eps_end} with interval {eps_interval}")
    for k, v in noise_dict.items():
        print(f"Epsilon = {k}, noise = {v}")
        
def read_config_as_dict(config_file_path : str):
    conf_dict = {}
    with open(config_file_path) as f:
        try: 
            full_yaml = yaml.safe_load(f)["clustering"]
            conf_dict["subsample_cap"] = full_yaml["subsample_cap"]
            conf_dict["random_state"] = full_yaml["random_state"]
            conf_dict["csv_file_path"] = full_yaml["csv_file_path"]
            conf_dict["k_min"] = full_yaml["kmeans"]["k_min"]
            conf_dict["k_max"] = full_yaml["kmeans"]["k_max"]
            conf_dict["k_final"] = full_yaml["kmeans"]["k_final"]
            conf_dict["k_dist"] = full_yaml["dbscan"]["k_dist"]
            conf_dict["eps_min"] = full_yaml["dbscan"]["eps_min"]
            conf_dict["eps_max"] = full_yaml["dbscan"]["eps_max"]
            conf_dict["eps_interval"] = full_yaml["dbscan"]["eps_interval"]
            conf_dict["eps_final"] = full_yaml["dbscan"]["eps_final"]
        except yaml.YAMLError as exc:
            print(exc)
            conf_dict = {
                "subsample_cap": 5,
                "random_state": 42,
                "csv_file_path": "summed_and_filtered.csv",
                "kmeans": {
                    "k_min": 2,
                    "k_max": 20,
                    "k_final": 4
                },
                "dbscan": {
                    "k_dist": 5,
                    "eps_min": 0.2,
                    "eps_max": 0.8,
                    "eps_interval": 0.1,
                    "eps_final": 0.4
                }
            }
        finally:
           return conf_dict 

def main(config_file_path : str):
    config_dict = read_config_as_dict(config_file_path)
    global RANDOM_STATE 
    RANDOM_STATE = config_dict["random_state"]
    df = pd.read_csv(config_dict["csv_file_path"])
    X_sanitized, df_sanitized = get_sanitized_and_subsampled_features_and_org_df(df)
    X_scaled, X_scaled_as_array = get_scaled_df_and_np(X_sanitized)
    
    print("Kmeans experiments")
    kmeans_k_search(X_sanitized, df_sanitized, k_min=config_dict["k_min"], k_max=config_dict["k_max"])
    kmeans_clustering(X_sanitized, df_sanitized, n_of_clusters=config_dict["k_final"])
    
    print_terminal_line()
    print("DBSCAN experiments")
    plot_k_distance_graph(X_scaled_as_array)
    print_noise_values_on_epsilon_range(X_scaled_as_array, eps_start=config_dict["eps_min"], eps_end=config_dict["eps_max"], eps_interval=config_dict["eps_interval"])
    dbscan_clustering(X_sanitized, df_sanitized, eps=config_dict["eps_final"])
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clustering.py <path_to_config_file>")
    else:
        main(sys.argv[1])