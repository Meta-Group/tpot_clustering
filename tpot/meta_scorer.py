import xgboost as xgb

def xgb_reg(meta_features, sil, dbs):
    try:
        xbg_reg = xgb.Booster()
        xbg_reg.load_model("tpot/models/xgb.json")
        mf = meta_features.copy()
        mf.extend([sil,dbs]) 
        surrogate_score = xbg_reg.predict(xgb.DMatrix([mf])).tolist()[0] 
        
    except Exception as e:
        print(f"XGB error: {e}")
        
    return surrogate_score
    
        
