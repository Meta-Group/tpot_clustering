import tpot
from tpot import TPOTClustering
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pymfe.mfe import MFE
import neptune

# dataset_name = "dim2-clusters3-instances2000-overlap1e-06-1e-05-aspectref5-aspectmaxmin1-radius1-imbalance1-rep11"
dataset_name = "dermatology"

dataset = pd.read_csv(f"datasets/validation/{dataset_name}.csv")
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(dataset)
normalized_df = pd.DataFrame(normalized_data, columns=dataset.columns)

labels = pd.read_csv(f"datasets/cluster_labels/{dataset_name}.csv")
labels = labels['Removed_Column'].tolist()

gen = 3
pop = 5

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
for i in range(1):

    try:
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwODNjNDRiNS02MDM4LTQ2NGEtYWQwMC00OGRhYjcwODc0ZDIifQ=="
        project_name = "MaleLab/PoAClustering"
        clusterer = TPOTClustering(
            generations=gen,
            population_size=pop,
            verbosity=2,
            config_dict=tpot.config.clustering_config_dict,
            n_jobs=-1,
            early_stop=int(gen*0.2)
        )
        _meta_features = extract_metafeatures(dataset)
        run = neptune.init_run(project=project_name, api_token=api_token)

        run["dataset"] = dataset_name
        run["gen"] = gen
        run["pop"] = pop
        print(f"\n==================== TPOT CLUSTERING ==================== \n Dataset: {dataset_name} - Surrogate Function")
        
        clusterer.fit(normalized_df, meta_features=_meta_features, labels_true=labels, model_name="2_ML2DAC")
        pipeline, scores, clusters, surrogate_score = clusterer.get_run_stats()

        print(f"\n---------------------------------------------------------\n")
        print("Results:")
        print(f"Pipeline: {pipeline}")
        print(f"Scores: {scores}")
        print(f"K: {clusters}")
        print(f"yhat: {surrogate_score}")
        print(f"gen: {gen}")

        # run["sil"] = scores['sil']
        # run["dbs"] = scores['dbs']
        # run["ari"] = scores['ari']
        # run["clusters"] = clusters
        # run["pipeline"] = pipeline
        # run["surrogate_score"] = surrogate_score
        
    except Exception as e:
        # run["error_msg"] = e
        print(f"{e}")

    # run.sync()
    # run.stop()