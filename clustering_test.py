import tpot
from tpot.tpot import TPOTClustering
from sklearn.datasets import load_iris
import pandas as pd
import neptune.new as neptune

iris = load_iris()
run_times = "5"

for i in range(0, 3630):
    api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwODNjNDRiNS02MDM4LTQ2NGEtYWQwMC00OGRhYjcwODc0ZDIifQ=="
    project_name = "MaleLab/Tpot4Clustering"
    project = neptune.init_project(project=project_name, api_token=api_token)
    
    run = neptune.init_run(
        project="MaleLab/Tpot4Clustering", with_id=run_id, api_token=api_token
    )
    

    columns = [
        "sys/id",
        "dataset",
        "clusters",
        "status",
        "scorers/sil",
        "gp/gen",
        "gp/pop",
        "mo",
        "pipeline",
        "scorers/bic",
        "scorers/chz",
        "scorers/dbi",
        "scorers/sil",
    ]

    runs_table_df = project.fetch_runs_table(columns=columns).to_pandas()

    run = runs_table_df[
        (runs_table_df["status"] < run_times) & (runs_table_df["scorers/bic"] == "-")
    ].sample()

    if run.empty:
        exit()

    dataset_name = run["dataset"].values[0]
    mo = run["mo"].values[0]
    gen = run["gp/gen"].values[0]
    pop = run["gp/pop"].values[0]
    run_id = run["sys/id"].values[0]
    run_number = int(run["status"].values[0])
    bic = run["scorers/bic"].values[0]
    chs = run["scorers/chz"].values[0]
    dbs = run["scorers/dbi"].values[0]
    sil = run["scorers/sil"].values[0]

    dataset = pd.read_csv(f"datasets/{dataset_name}.csv")

    _scorers = []
    if sil != "-":
        _scorers.append("sil")
    if dbs != "-":
        _scorers.append("dbs")
    if chs != "-":
        _scorers.append("chs")
    if bic != "-":
        _scorers.append("bic")

    # config scorers
    clusterer = TPOTClustering(
        generations=int(gen),
        population_size=int(pop),
        verbosity=2,
        config_dict=tpot.config.clustering_config_dict,
        n_jobs=1,
    )
    print(f"\n==================== TPOT CLUSTERING ==================== \n Run ID: {run_id} - Dataset: {dataset_name} - Scorers: {_scorers} - mo: {mo}")
    clusterer.fit(dataset, mo_function=mo, scorers=_scorers)

    pipeline, scores, clusters = clusterer.get_run_stats()
    
    run["clusters"] = clusters
    run["scorers/sil"].append(scores["sil"])
    run["scorers/chz"].append(scores["chs"])
    run["scorers/dbi"].append(scores["dbs"])
    run["scorers/bic"].append(scores["bic"])
    run["status"] = str(run_number + 1)

    run.sync()
    run.stop()
