import tpot
from tpot import TPOTClustering
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
dataset_name = "a1"
# todo - dependecy injection on fit

run_id = "TEST"
# dataset = iris.data
dataset = pd.read_csv(f"datasets/training/{dataset_name}.csv")

mo = "mean_score"
gen = 10
pop = 100

# extrair meta-features
from pymfe.mfe import MFE
X = dataset.values

# Extract all available unsupervised measures
mfe = MFE(groups="all")
mfe.fit(X)
ft = mfe.extract()
n_instances = X.shape[0]
n_features = X.shape[1]

features = ['attr_conc.mean', 'attr_conc.sd', 'attr_ent.mean', 'attr_ent.sd',
       'attr_to_inst', 'cat_to_num', 'cohesiveness.mean', 'cohesiveness.sd',
       'cor.mean', 'cov.mean', 'eigenvalues.mean', 'eigenvalues.sd',
       'inst_to_attr', 'iq_range.mean', 'iq_range.sd', 'kurtosis.mean',
       'kurtosis.sd', 'mad.mean', 'mad.sd', 'max.mean', 'max.sd', 'mean.mean',
       'mean.sd', 'median.mean', 'median.sd', 'min.mean', 'min.sd', 'nr_attr',
       'nr_bin', 'nr_cat', 'nr_cor_attr', 'nr_inst', 'nr_norm', 'nr_num',
       'nr_outliers', 'one_itemset.mean', 'one_itemset.sd', 'range.mean',
       'range.sd', 'sd.mean', 'sd.sd', 'skewness.mean', 'skewness.sd',
       'sparsity.mean', 'sparsity.sd', 't2', 't3', 't4', 't_mean.mean',
       't_mean.sd', 'two_itemset.mean', 'two_itemset.sd', 'var.mean', 'var.sd',
       'wg_dist.mean', 'wg_dist.sd', 'instances', 'features', 'sil', 'dbs',
       'cluster_diff']

_meta_features = [value for key, value in zip(ft[0], ft[1]) if key in features]
_meta_features.extend([n_instances, n_features])

try:

    clusterer = TPOTClustering(
        generations=gen,
        population_size=pop,
        verbosity=0,
        config_dict=tpot.config.clustering_config_dict,
        n_jobs=-1,
        # early_stop=int(gen*0.2)
    )

    print(f"\n==================== TPOT CLUSTERING ==================== \n Run ID: {run_id} - Dataset: {dataset_name} - Surrogate Function")
    clusterer.fit(dataset, meta_features=_meta_features)
    pipeline, scores, clusters, labels = clusterer.get_run_stats()
    print(f"\n---------------------------------------------------------\n")
    print("Results:")
    print(f"Pipeline: {pipeline}")
    print(f"Scores: {scores}")
    print(f"K: {clusters}")
        

except Exception as e:
    print(f"Error: {e}")