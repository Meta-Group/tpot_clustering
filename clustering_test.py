import tpot
from tpot.tpot import TPOTClustering
from sklearn.datasets import load_iris
import pandas as pd
import neptune.new as neptune

iris = load_iris()

api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwODNjNDRiNS02MDM4LTQ2NGEtYWQwMC00OGRhYjcwODc0ZDIifQ=="
project_name = "MaleLab/Tpot4Clustering"
project = neptune.init_project(project=project_name, api_token=api_token)


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
    (runs_table_df.status == "wainting") & (runs_table_df["scorers/dbi"] == "-")
].sample()

dataset_name = run["dataset"].values[0]
mo = run["mo"].values[0]
gen = run["gp/gen"].values[0]
pop = run["gp/pop"].values[0]
run_id = run["sys/id"].values[0]

dataset = pd.read_csv(f"{dataset_name}.csv")

clusterer = TPOTClustering(
    generations=gen,
    population_size=pop,
    verbosity=2,
    config_dict=tpot.config.clustering_config_dict,
    n_jobs=1,
)

clusterer.fit(dataset, mo_function=mo)
# clusters, sil, chz, dbi, bic = clusterer.get_best_score()
# run = neptune.init_run(
#     project="MaleLab/Tpot4Clustering", with_id=run_id, api_token=api_token
# )
# run["clusters"] = clusters
# run["scorers/sil"] = sil
# run["scorers/chz"] = chz
# run["scorers/dbi"] = dbi
# run["scorers/bic"] = bic
# run["status"] = "done"
# run.sync()
# run.stop()
