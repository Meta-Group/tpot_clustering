import tpot
from tpot.tpot import TPOTClustering
from sklearn.datasets import load_iris
import pandas as pd
import neptune.new as neptune
import requests
import json


def get_run_config():
    find_one_url = "https://eu-central-1.aws.data.mongodb-api.com/app/data-vhbni/endpoint/data/v1/action/findOne"
    headers = {
        "Content-Type": "application/json",
        "Access-Control-Request-Headers": "*",
        "api-key": "tURg5ipsw8mFrPwY52B0d37bzsRQLyk1UFOjFz0fkicfra1FzlcrsDwOl4ctCymr",
    }
    payload = json.dumps(
        {
            "collection": "runs",
            "database": "tpot",
            "dataSource": "Malelab",
            "filter": {"status": "active"},
        }
    )

    response = requests.request("POST", find_one_url, headers=headers, data=payload)
    _response = response.json()
    return _response["document"]


def update_run(run, status):
    update_one_url = "https://eu-central-1.aws.data.mongodb-api.com/app/data-vhbni/endpoint/data/v1/action/updateOne"

    headers = {
        "Content-Type": "application/json",
        "Access-Control-Request-Headers": "*",
        "api-key": "tURg5ipsw8mFrPwY52B0d37bzsRQLyk1UFOjFz0fkicfra1FzlcrsDwOl4ctCymr",
    }

    payload = json.dumps(
        {
            "collection": "runs",
            "database": "tpot",
            "dataSource": "Malelab",
            "filter": {
                "_id": {"$oid": run["_id"]},
            },
            "update": {"$set": {"status": status}},
        }
    )

    response = requests.request("POST", update_one_url, headers=headers, data=payload)
    print(response.text)


while 1:
    api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwODNjNDRiNS02MDM4LTQ2NGEtYWQwMC00OGRhYjcwODc0ZDIifQ=="
    project_name = "MaleLab/Tpot4ClusteringLight"
    run_config = get_run_config()
    if not run_config:
        print("\n\n0 Active runs --- bye")
        quit()

    dataset_name = run_config["dataset"]
    mo = run_config["mo"]
    gen = run_config["gen"]
    pop = run_config["pop"]
    run_id = run_config["_id"]
    run_number = run_config["status"]
    bic = run_config["bic"]
    chs = run_config["chs"]
    dbs = run_config["dbs"]
    sil = run_config["sil"]

    dataset = pd.read_csv(f"datasets/{dataset_name}.csv")

    run = neptune.init_run(project=project_name, api_token=api_token)
    _scorers = []
    if sil != "-":
        _scorers.append("sil")
    if dbs != "-":
        _scorers.append("dbs")
    if chs != "-":
        _scorers.append("chs")
    if bic != "-":
        _scorers.append("bic")

    print(
        f"\n==================== TPOT CLUSTERING ==================== \n Run ID: {run_id} - Dataset: {dataset_name} - Scorers: {_scorers} - mo: {mo}"
    )
    try:
        update_run(run_config, "occupied")
        # config scorers
        clusterer = TPOTClustering(
            # generations=gen,
            # population_size=pop,
            generations=gen,
            population_size=pop,
            verbosity=2,
            config_dict=tpot.config.clustering_config_dict,
            n_jobs=1,
        )
        clusterer.fit(dataset, mo_function=mo, scorers=_scorers)

        pipeline, scores, clusters = clusterer.get_run_stats()
        print(f"Pipeline: {pipeline} Scores: {scores} Clusters: {clusters}")
        run["_id"] = run_id
        run["dataset"] = dataset_name
        run["mo"] = mo
        run["gen"] = gen
        run["pop"] = pop
        run["clusters"] = clusters
        run["pipeline"] = pipeline
        for scorer_name in _scorers:
            run[scorer_name] = round(scores[scorer_name], 4)
        update_run(run_config, "finished")

    except Exception as e:
        run["_id"] = run_id
        run["dataset"] = dataset_name
        run["mo"] = mo
        run["gen"] = gen
        run["pop"] = pop
        run["error_msg"] = e
        print(f"{e}")
        update_run(run_config, "error")

    run.sync()
    run.stop()
