import tpot
# from tpot import tpot
from tpot.tpot import TPOTClustering

from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# import dask_ml.model_selection
# import pandas as pd

iris = load_iris()
digits = load_digits()
# # a1 = pd.read_csv('a1.csv')

clusterer = TPOTClustering(
        generations=10,
        population_size=10,
        verbosity=2,
        config_dict=tpot.config.clustering_config_dict,
        n_jobs=1
        )

clusterer.fit(digits.data, mo_function='mean_score')
    