import xgboost as xgb
import joblib
from sklearn.ensemble import RandomForestRegressor

def xgb_reg(meta_features, sil, dbs, n_clusters):
    
    try:
        xbg_reg = xgb.Booster()
        xbg_reg.load_model("tpot/models/xgb.json")
        mf = meta_features.copy()
        mf.extend([sil,dbs,n_clusters])
        surrogate_score = xbg_reg.predict(xgb.DMatrix([mf])).tolist()[0] 
        return surrogate_score
        
    except Exception as e:
        print(f"XGB error: {e}")
        return float('inf')


def rf_sv5_reg(meta_features, sil, dbs, n_clusters):
    try:
        model = RandomForestRegressor()
        model = joblib.load("tpot/models/RFR_Sv5_b.joblib")
        mf = meta_features.copy()
        mf.extend([sil, dbs, n_clusters])
        surrogate_score = model.predict([mf])
        return surrogate_score

    except Exception as e:
        print(f"XGB error: {e}")
        return float('inf')
        
    
        
