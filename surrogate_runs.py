import tpot
from tpot.tpot import TPOTClustering
import pandas as pd
import neptune.new as neptune
import requests
import json
from pymfe.mfe import MFE
from sklearn.preprocessing import MinMaxScaler

model_name = "0_Biagio"
collection_name = "biagio_median"

def extract_metafeatures(dataset):
    X = dataset.values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    features = ['attr_conc', 'attr_ent', 'attr_to_inst', 'cohesiveness', 'cor', 'cov',
                'eigenvalues', 'inst_to_attr', 'iq_range', 'kurtosis', 'mad', 'max', 'mean', 'median', 'min',
                'nr_attr', 'nr_cor_attr', 'nr_inst',
                'one_itemset', 'range', 'sd', 'skewness',
                'sparsity', 't2', 't3', 't4', 't_mean', 'two_itemset', 'var', 'wg_dist',
                ]
    mfe = MFE(groups="all", features=features)
    mfe.fit(X)
    ft = mfe.extract()
    mfeatures = ['attr_conc.mean', 'attr_conc.sd', 'attr_ent.mean',
                       'attr_ent.sd', 'attr_to_inst', 'cohesiveness.mean', 'cohesiveness.sd',
                       # 'cor.mean',#'cor.sd',
                       'cov.mean',  # 'cov.sd',
                       'eigenvalues.mean', 'eigenvalues.sd',
                       'inst_to_attr', 'iq_range.mean', 'iq_range.sd',
                       # 'kurtosis.mean','kurtosis.sd',
                       'mad.mean', 'mad.sd',
                       # 'max.mean','max.sd','mean.mean','mean.sd',
                       'median.mean',
                       'median.sd',  # 'min.mean','min.sd',
                       'nr_attr', 'nr_cor_attr', 'nr_inst', 'one_itemset.mean',
                       'one_itemset.sd',
                       # 'range.mean','range.sd',
                       'sd.mean', 'sd.sd'
                       # ,'skewness.mean', 'skewness.sd'
                        , 'sparsity.mean', 'sparsity.sd', 't2', 't3', 't4', 't_mean.mean',
                       't_mean.sd', 'two_itemset.mean', 'two_itemset.sd', 'var.mean', 'var.sd',
                       'wg_dist.mean', 'wg_dist.sd',
                       'sil', 'dbs', 'clusters', 'cluster_diff'
                       ]

    _meta_features = [value for key, value in zip(ft[0], ft[1]) if key in mfeatures]
    return _meta_features


headers = {
    "Content-Type": "application/json",
    "Access-Control-Request-Headers": "*",
    "api-key": "tURg5ipsw8mFrPwY52B0d37bzsRQLyk1UFOjFz0fkicfra1FzlcrsDwOl4ctCymr",
}

def get_run_config():
    find_one_url = "https://eu-central-1.aws.data.mongodb-api.com/app/data-vhbni/endpoint/data/v1/action/findOne"
    payload = json.dumps(
        {
            "collection": collection_name,
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
    payload = json.dumps(
        {
            "collection": collection_name,
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
# for i in range(2):
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwODNjNDRiNS02MDM4LTQ2NGEtYWQwMC00OGRhYjcwODc0ZDIifQ=="
    project_name = "MaleLab/TPOTSurrogate"
    run_config = get_run_config()
    if not run_config:
        print("\n\n0 Active runs --- bye")
        quit()

    dataset_name = run_config["dataset"]
    gen = run_config["gen"]
    pop = run_config["pop"]
    run_id = run_config["_id"]
    run_number = run_config["status"]

    dataset = pd.read_csv(f"datasets/validation/{dataset_name}")
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(dataset)
    normalized_df = pd.DataFrame(normalized_data, columns=dataset.columns)

    labels = pd.read_csv(f"datasets/cluster_labels/{dataset_name}")
    labels = labels['Removed_Column'].tolist()

    _meta_features = extract_metafeatures(dataset)
    run = neptune.init_run(project=project_name, api_token=api_token)
    run["_id"] = run_id
    run["dataset"] = dataset_name
    run["gen"] = gen
    run["pop"] = pop

    print(f"\n==================== TPOT CLUSTERING SURROGATE ==================== \n Run ID: {run_id} - Dataset: {dataset_name}")

    try:
        update_run(run_config, "occupied")
        clusterer = TPOTClustering(
            generations=15,
            population_size=100,
            verbosity=2,
            config_dict=tpot.config.clustering_config_dict,
            n_jobs=1,
            # max_time_mins=1
            # early_stop=int(gen*0.1)
        )
        
        clusterer.fit(normalized_df, meta_features=_meta_features, labels_true=labels, model_name=model_name)
        pipeline, scores, clusters, surrogate_score = clusterer.get_run_stats()

        print(f">> Pipeline: {pipeline} Scores: {scores} Clusters: {clusters} Surrogate: {surrogate_score}<<")
        run["sil"] = scores['sil']
        run["dbs"] = scores['dbs']
        run["ari"] = scores['ari']
        run["clusters"] = clusters
        run["pipeline"] = pipeline
        run["surrogate_score"] = surrogate_score
        run["sys/tags"].add(model_name)
        update_run(run_config, "finished")

    except Exception as e:
        run["error_msg"] = e
        print(f"{e}")
        update_run(run_config, "error")

    run.sync()
    run.stop()
