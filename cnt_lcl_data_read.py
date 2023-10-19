import pandas as pd

def read_data():
    # print("Arabic:")
    ar_tr = pd.read_csv("data/Arabic/ar_tr_6.4k.csv")
    ar_val = pd.read_csv("data/Arabic/ar_val_1.6k.csv")
    ar_ts = pd.read_csv("data/Arabic/ar_ts_2k.csv")
    # print("Russian:")
    ru_tr = pd.read_csv("data/Russian/ru_tr_6.4k.csv")
    ru_val = pd.read_csv("data/Russian/ru_val_1.6k.csv")
    ru_ts = pd.read_csv("data/Russian/ru_ts_2k.csv")
    # print("Korean:")
    ko_tr = pd.read_csv("data/Korean/ko_tr_640.csv")
    ko_val = pd.read_csv("data/Korean/ko_val_160.csv")
    ko_ts = pd.read_csv("data/Korean/ko_ts_200.csv")
    # print("Spanish:")
    sp_tr = pd.read_csv("data/Spanish/sp_tr_1280.csv")
    sp_val = pd.read_csv("data/Spanish/sp_val_320.csv")
    sp_ts = pd.read_csv("data/Spanish/sp_ts_400.csv")
    # print("English:")
    en_tr = pd.read_csv("data/English CLPsych/en_tr_640.csv")
    en_val = pd.read_csv("data/English CLPsych/en_val_160.csv")
    en_ts = pd.read_csv("data/English CLPsych/en_ts_200.csv")

    tr_df = pd.concat([ru_tr, ar_tr, ko_tr, sp_tr, en_tr])
    tr_df.pop(tr_df.columns[0])
    val_df = pd.concat([ru_val, ar_val, ko_val, sp_val, en_val])
    val_df.pop(val_df.columns[0])
    ts_df = pd.concat([ru_ts, ar_ts, ko_ts, sp_ts, en_ts])
    ts_df.pop(ts_df.columns[0])

    tr_df, ts_df, val_df = tr_df.dropna() , ts_df.dropna() , val_df.dropna()

    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_51k_evDis_data():
    # print("Arabic:")
    ar_tr = pd.read_csv("data/5_1k_evDis/arabic-1k/ar_tr_640.csv")
    ar_val = pd.read_csv("data/5_1k_evDis/arabic-1k/ar_val_160.csv")
    ar_ts = pd.read_csv("data/5_1k_evDis/arabic-1k/ar_ts_200.csv")
    # print("Russian:")
    ru_tr = pd.read_csv("data/5_1k_evDis/russian-1k/ru_tr_640.csv")
    ru_val = pd.read_csv("data/5_1k_evDis/russian-1k/ru_val_160.csv")
    ru_ts = pd.read_csv("data/5_1k_evDis/russian-1k/ru_ts_200.csv")
    # print("Korean:")
    ko_tr = pd.read_csv("data/5_1k_evDis/korean-1k/ko_tr_640.csv")
    ko_val = pd.read_csv("data/5_1k_evDis/korean-1k/ko_val_160.csv")
    ko_ts = pd.read_csv("data/5_1k_evDis/korean-1k/ko_ts_200.csv")
    # print("Spanish:")
    sp_tr = pd.read_csv("data/5_1k_evDis/spanish-1k/sp_tr_640.csv")
    sp_val = pd.read_csv("data/5_1k_evDis/spanish-1k/sp_val_160.csv")
    sp_ts = pd.read_csv("data/5_1k_evDis/spanish-1k/sp_ts_200.csv")
    # print("English:")
    en_tr = pd.read_csv("data/5_1k_evDis/english-1k/en_tr_640.csv")
    en_val = pd.read_csv("data/5_1k_evDis/english-1k/en_val_160.csv")
    en_ts = pd.read_csv("data/5_1k_evDis/english-1k/en_ts_200.csv")

    tr_df = pd.concat([ru_tr, ar_tr, ko_tr, sp_tr, en_tr])
    tr_df.pop(tr_df.columns[0])
    val_df = pd.concat([ru_val, ar_val, ko_val, sp_val, en_val])
    val_df.pop(val_df.columns[0])
    ts_df = pd.concat([ru_ts, ar_ts, ko_ts, sp_ts, en_ts])
    ts_df.pop(ts_df.columns[0])

    tr_df, ts_df, val_df = tr_df.dropna() , ts_df.dropna() , val_df.dropna()

    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_51k_unEvDis_data():
    # print("Arabic:")
    ar_tr = pd.read_csv("data/5_1k_unEvDis/arabic-1k/ar_tr_641.csv")
    ar_val = pd.read_csv("data/5_1k_unEvDis/arabic-1k/ar_val_159.csv")
    ar_ts = pd.read_csv("data/5_1k_unEvDis/arabic-1k/ar_ts_200.csv")
    # print("Russian:")
    ru_tr = pd.read_csv("data/5_1k_unEvDis/russian-1k/ru_tr_641.csv")
    ru_val = pd.read_csv("data/5_1k_unEvDis/russian-1k/ru_val_160.csv")
    ru_ts = pd.read_csv("data/5_1k_unEvDis/russian-1k/ru_ts_199.csv")
    # print("Korean:")
    ko_tr = pd.read_csv("data/5_1k_unEvDis/korean-1k/ko_tr_640.csv")
    ko_val = pd.read_csv("data/5_1k_unEvDis/korean-1k/ko_val_160.csv")
    ko_ts = pd.read_csv("data/5_1k_unEvDis/korean-1k/ko_ts_200.csv")
    # print("Spanish:")
    sp_tr = pd.read_csv("data/5_1k_unEvDis/spanish-1k/sp_tr_642.csv")
    sp_val = pd.read_csv("data/5_1k_unEvDis/spanish-1k/sp_val_159.csv")
    sp_ts = pd.read_csv("data/5_1k_unEvDis/spanish-1k/sp_ts_199.csv")
    # print("English:")
    en_tr = pd.read_csv("data/5_1k_unEvDis/english-1k/en_tr_640.csv")
    en_val = pd.read_csv("data/5_1k_unEvDis/english-1k/en_val_160.csv")
    en_ts = pd.read_csv("data/5_1k_unEvDis/english-1k/en_ts_200.csv")

    tr_df = pd.concat([ru_tr, ar_tr, ko_tr, sp_tr, en_tr])
    tr_df.pop(tr_df.columns[0])
    val_df = pd.concat([ru_val, ar_val, ko_val, sp_val, en_val])
    val_df.pop(val_df.columns[0])
    ts_df = pd.concat([ru_ts, ar_ts, ko_ts, sp_ts, en_ts])
    ts_df.pop(ts_df.columns[0])

    tr_df, ts_df, val_df = tr_df.dropna() , ts_df.dropna() , val_df.dropna()

    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_difQ_evDis_data():
    # print("Arabic:")
    ar_tr = pd.read_csv("data/difQ_evDis/arabic-6240/ar_tr_3994.csv")
    ar_val = pd.read_csv("data/difQ_evDis/arabic-6240/ar_val_1248.csv")
    ar_ts = pd.read_csv("data/difQ_evDis/arabic-6240/ar_ts_998.csv")
    # print("Russian:")
    ru_tr = pd.read_csv("data/difQ_evDis/russian-1283/ru_tr_823.csv")
    ru_val = pd.read_csv("data/difQ_evDis/russian-1283/ru_val_204.csv")
    ru_ts = pd.read_csv("data/difQ_evDis/russian-1283/ru_ts_256.csv")
    # print("Korean:")
    ko_tr = pd.read_csv("data/difQ_evDis/korean-392/ko_tr_252.csv")
    ko_val = pd.read_csv("data/difQ_evDis/korean-392/ko_val_62.csv")
    ko_ts = pd.read_csv("data/difQ_evDis/korean-392/ko_ts_77.csv")
    # print("Spanish:")
    sp_tr = pd.read_csv("data/difQ_evDis/spanish-1178/sp_tr_758.csv")
    sp_val = pd.read_csv("data/difQ_evDis/spanish-1178/sp_val_188.csv")
    sp_ts = pd.read_csv("data/difQ_evDis/spanish-1178/sp_ts_234.csv")
    # print("English:")
    en_tr = pd.read_csv("data/difQ_evDis/english-907/en_tr_583.csv")
    en_val = pd.read_csv("data/difQ_evDis/english-907/en_val_144.csv")
    en_ts = pd.read_csv("data/difQ_evDis/english-907/en_ts_180.csv")

    tr_df = pd.concat([ru_tr, ar_tr, ko_tr, sp_tr, en_tr])
    tr_df.pop(tr_df.columns[0])
    val_df = pd.concat([ru_val, ar_val, ko_val, sp_val, en_val])
    val_df.pop(val_df.columns[0])
    ts_df = pd.concat([ru_ts, ar_ts, ko_ts, sp_ts, en_ts])
    ts_df.pop(ts_df.columns[0])

    tr_df, ts_df, val_df = tr_df.dropna() , ts_df.dropna() , val_df.dropna()

    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_difQ_unEvDis_data():
    # print("Arabic:")
    ar_tr = pd.read_csv("data/difQ_unEvDis/arabic-6240/ar_tr_3994.csv")
    ar_val = pd.read_csv("data/difQ_unEvDis/arabic-6240/ar_val_1248.csv")
    ar_ts = pd.read_csv("data/difQ_unEvDis/arabic-6240/ar_ts_998.csv")
    # print("Russian:")
    ru_tr = pd.read_csv("data/difQ_unEvDis/russian-1283/ru_tr_823.csv")
    ru_val = pd.read_csv("data/difQ_unEvDis/russian-1283/ru_val_204.csv")
    ru_ts = pd.read_csv("data/difQ_unEvDis/russian-1283/ru_ts_256.csv")
    # print("Korean:")
    ko_tr = pd.read_csv("data/difQ_unEvDis/korean-392/ko_tr_252.csv")
    ko_val = pd.read_csv("data/difQ_unEvDis/korean-392/ko_val_62.csv")
    ko_ts = pd.read_csv("data/difQ_unEvDis/korean-392/ko_ts_77.csv")
    # print("Spanish:")
    sp_tr = pd.read_csv("data/difQ_unEvDis/spanish-1178/sp_tr_754.csv")
    sp_val = pd.read_csv("data/difQ_unEvDis/spanish-1178/sp_val_188.csv")
    sp_ts = pd.read_csv("data/difQ_unEvDis/spanish-1178/sp_ts_234.csv")
    # print("English:")
    en_tr = pd.read_csv("data/difQ_unEvDis/english-907/en_tr_583.csv")
    en_val = pd.read_csv("data/difQ_unEvDis/english-907/en_val_144.csv")
    en_ts = pd.read_csv("data/difQ_unEvDis/english-907/en_ts_180.csv")

    tr_df = pd.concat([ru_tr, ar_tr, ko_tr, sp_tr, en_tr])
    tr_df.pop(tr_df.columns[0])
    val_df = pd.concat([ru_val, ar_val, ko_val, sp_val, en_val])
    val_df.pop(val_df.columns[0])
    ts_df = pd.concat([ru_ts, ar_ts, ko_ts, sp_ts, en_ts])
    ts_df.pop(ts_df.columns[0])

    tr_df, ts_df, val_df = tr_df.dropna() , ts_df.dropna() , val_df.dropna()

    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_ar_10k_data():
    # print("Arabic:")
    ar_tr = pd.read_csv("data/Arabic/ar_tr_6.4k.csv")
    # print(ar_tr['label'].value_counts())
    ar_val = pd.read_csv("data/Arabic/ar_val_1.6k.csv")
    # print(ar_val['label'].value_counts())
    ar_ts = pd.read_csv("data/Arabic/ar_ts_2k.csv")
    # print(ar_ts['label'].value_counts())
    ar_tr.pop(ar_tr.columns[0])
    ar_val.pop(ar_val.columns[0])
    ar_ts.pop(ar_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = ar_tr.dropna() , ar_ts.dropna() , ar_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_ar_1k_data():
    # print("Arabic:")
    ar_tr = pd.read_csv("data/3_1k-2_2k/arabic-1k/ar_tr_640.csv")
    # print(ar_tr['label'].value_counts())
    ar_val = pd.read_csv("data/3_1k-2_2k/arabic-1k/ar_val_160.csv")
    # print(ar_val['label'].value_counts())
    ar_ts = pd.read_csv("data/3_1k-2_2k/arabic-1k/ar_ts_200.csv")
    # print(ar_ts['label'].value_counts())
    ar_tr.pop(ar_tr.columns[0])
    ar_val.pop(ar_val.columns[0])
    ar_ts.pop(ar_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = ar_tr.dropna() , ar_ts.dropna() , ar_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_ar_1k_unEvDis_data():
    # print("Arabic:")
    ar_tr = pd.read_csv("data/5_1k_unEvDis/arabic-1k/ar_tr_641.csv")
    # print(ar_tr['label'].value_counts())
    ar_val = pd.read_csv("data/5_1k_unEvDis/arabic-1k/ar_val_159.csv")
    # print(ar_val['label'].value_counts())
    ar_ts = pd.read_csv("data/5_1k_unEvDis/arabic-1k/ar_ts_200.csv")
    # print(ar_ts['label'].value_counts())
    ar_tr.pop(ar_tr.columns[0])
    ar_val.pop(ar_val.columns[0])
    ar_ts.pop(ar_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = ar_tr.dropna() , ar_ts.dropna() , ar_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_ar_difQ_evDis_data():
    # print("Arabic:")
    ar_tr = pd.read_csv("data/difQ_evDis/arabic-6240/ar_tr_3994.csv")
    ar_val = pd.read_csv("data/difQ_evDis/arabic-6240/ar_val_1248.csv")
    ar_ts = pd.read_csv("data/difQ_evDis/arabic-6240/ar_ts_998.csv")
    ar_tr.pop(ar_tr.columns[0])
    ar_val.pop(ar_val.columns[0])
    ar_ts.pop(ar_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = ar_tr.dropna() , ar_ts.dropna() , ar_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_ar_difQ_unEvDis_data():
    # print("Arabic:")
    ar_tr = pd.read_csv("data/difQ_unEvDis/arabic-6240/ar_tr_3994.csv")
    ar_val = pd.read_csv("data/difQ_unEvDis/arabic-6240/ar_val_1248.csv")
    ar_ts = pd.read_csv("data/difQ_unEvDis/arabic-6240/ar_ts_998.csv")
    ar_tr.pop(ar_tr.columns[0])
    ar_val.pop(ar_val.columns[0])
    ar_ts.pop(ar_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = ar_tr.dropna() , ar_ts.dropna() , ar_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_ru_10k_data():
    # print("Russian:")
    ru_tr = pd.read_csv("data/Russian/ru_tr_6.4k.csv")
    # print(ar_tr['label'].value_counts())
    ru_val = pd.read_csv("data/Russian/ru_val_1.6k.csv")
    # print(ar_val['label'].value_counts())
    ru_ts = pd.read_csv("data/Russian/ru_ts_2k.csv")
    # print(ar_ts['label'].value_counts())
    ru_tr.pop(ru_tr.columns[0])
    ru_val.pop(ru_val.columns[0])
    ru_ts.pop(ru_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = ru_tr.dropna() , ru_ts.dropna() , ru_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_ru_1k_data():
    # print("Russian:")
    ru_tr = pd.read_csv("data/5_1k_evDis/russian-1k/ru_tr_640.csv")
    # print(ar_tr['label'].value_counts())
    ru_val = pd.read_csv("data/5_1k_evDis/russian-1k/ru_val_160.csv")
    # print(ar_val['label'].value_counts())
    ru_ts = pd.read_csv("data/5_1k_evDis/russian-1k/ru_ts_200.csv")
    # print(ar_ts['label'].value_counts())
    ru_tr.pop(ru_tr.columns[0])
    ru_val.pop(ru_val.columns[0])
    ru_ts.pop(ru_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = ru_tr.dropna() , ru_ts.dropna() , ru_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df
  
def read_ru_1k_unEvDis_data():
    # print("Russian:")
    ru_tr = pd.read_csv("data/5_1k_unEvDis/russian-1k/ru_tr_641.csv")
    # print(ar_tr['label'].value_counts())
    ru_val = pd.read_csv("data/5_1k_unEvDis/russian-1k/ru_val_160.csv")
    # print(ar_val['label'].value_counts())
    ru_ts = pd.read_csv("data/5_1k_unEvDis/russian-1k/ru_ts_199.csv")
    # print(ar_ts['label'].value_counts())
    ru_tr.pop(ru_tr.columns[0])
    ru_val.pop(ru_val.columns[0])
    ru_ts.pop(ru_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = ru_tr.dropna() , ru_ts.dropna() , ru_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_ru_difQ_evDis_data():
    # print("Russian:")
    ru_tr = pd.read_csv("data/difQ_evDis/russian-1283/ru_tr_823.csv")
    ru_val = pd.read_csv("data/difQ_evDis/russian-1283/ru_val_204.csv")
    ru_ts = pd.read_csv("data/difQ_evDis/russian-1283/ru_ts_256.csv")
    ru_tr.pop(ru_tr.columns[0])
    ru_val.pop(ru_val.columns[0])
    ru_ts.pop(ru_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = ru_tr.dropna() , ru_ts.dropna() , ru_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_ru_difQ_unEvDis_data():
    # print("Russian:")
    ru_tr = pd.read_csv("data/difQ_unEvDis/russian-1283/ru_tr_823.csv")
    ru_val = pd.read_csv("data/difQ_unEvDis/russian-1283/ru_val_204.csv")
    ru_ts = pd.read_csv("data/difQ_unEvDis/russian-1283/ru_ts_256.csv")
    ru_tr.pop(ru_tr.columns[0])
    ru_val.pop(ru_val.columns[0])
    ru_ts.pop(ru_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = ru_tr.dropna() , ru_ts.dropna() , ru_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df
    
def read_en_1k_data():
    # print("Russian:")
    en_tr = pd.read_csv("data/English CLPsych/en_tr_640.csv")
    # print(ar_tr['label'].value_counts())
    en_val = pd.read_csv("data/English CLPsych/en_val_160.csv")
    # print(ar_val['label'].value_counts())
    en_ts = pd.read_csv("data/English CLPsych/en_ts_200.csv")
    # print(ar_ts['label'].value_counts())
    en_tr.pop(en_tr.columns[0])
    en_val.pop(en_val.columns[0])
    en_ts.pop(en_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = en_tr.dropna() , en_ts.dropna() , en_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_en_difQ_evDis_data():
    # print("Russian:")
    en_tr = pd.read_csv("data/difQ_evDis/english-907/en_tr_583.csv")
    en_val = pd.read_csv("data/difQ_evDis/english-907/en_val_144.csv")
    en_ts = pd.read_csv("data/difQ_evDis/english-907/en_ts_180.csv")
    en_tr.pop(en_tr.columns[0])
    en_val.pop(en_val.columns[0])
    en_ts.pop(en_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = en_tr.dropna() , en_ts.dropna() , en_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_en_difQ_unEvDis_data():
    # print("Russian:")
    en_tr = pd.read_csv("data/difQ_unEvDis/english-907/en_tr_583.csv")
    en_val = pd.read_csv("data/difQ_unEvDis/english-907/en_val_144.csv")
    en_ts = pd.read_csv("data/difQ_unEvDis/english-907/en_ts_180.csv")
    en_tr.pop(en_tr.columns[0])
    en_val.pop(en_val.columns[0])
    en_ts.pop(en_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = en_tr.dropna() , en_ts.dropna() , en_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_ko_1k_data():
    # print("Korean:")
    ko_tr = pd.read_csv("data/Korean/ko_tr_640.csv")
    # print(ko_tr['label'].value_counts())
    ko_val = pd.read_csv("data/Korean/ko_val_160.csv")
    # print(ko_val['label'].value_counts())
    ko_ts = pd.read_csv("data/Korean/ko_ts_200.csv")
    # print(ko_ts['label'].value_counts())
    ko_tr.pop(ko_tr.columns[0])
    ko_val.pop(ko_val.columns[0])
    ko_ts.pop(ko_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = ko_tr.dropna() , ko_ts.dropna() , ko_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_ko_difQ_evDis_data():
    # print("Korean:")
    ko_tr = pd.read_csv("data/difQ_evDis/korean-392/ko_tr_252.csv")
    ko_val = pd.read_csv("data/difQ_evDis/korean-392/ko_val_62.csv")
    ko_ts = pd.read_csv("data/difQ_evDis/korean-392/ko_ts_77.csv")
    ko_tr.pop(ko_tr.columns[0])
    ko_val.pop(ko_val.columns[0])
    ko_ts.pop(ko_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = ko_tr.dropna() , ko_ts.dropna() , ko_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_ko_difQ_unEvDis_data():
    # print("Korean:")
    ko_tr = pd.read_csv("data/difQ_unEvDis/korean-392/ko_tr_252.csv")
    ko_val = pd.read_csv("data/difQ_unEvDis/korean-392/ko_val_62.csv")
    ko_ts = pd.read_csv("data/difQ_unEvDis/korean-392/ko_ts_77.csv")
    ko_tr.pop(ko_tr.columns[0])
    ko_val.pop(ko_val.columns[0])
    ko_ts.pop(ko_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = ko_tr.dropna() , ko_ts.dropna() , ko_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_sp_2k_data():
    # print("Spanish:")
    sp_tr = pd.read_csv("data/Spanish/sp_tr_1280.csv")
    # print(sp_tr['label'].value_counts())
    sp_val = pd.read_csv("data/Spanish/sp_val_320.csv")
    # print(sp_val['label'].value_counts())
    sp_ts = pd.read_csv("data/Spanish/sp_ts_400.csv")
    # print(sp_ts['label'].value_counts())
    sp_tr.pop(sp_tr.columns[0])
    sp_val.pop(sp_val.columns[0])
    sp_ts.pop(sp_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = sp_tr.dropna() , sp_ts.dropna() , sp_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_sp_1k_data():
    # print("Spanish:")
    sp_tr = pd.read_csv("data/5_1k_evDis/spanish-1k/sp_tr_640.csv")
    # print(sp_tr['label'].value_counts())
    sp_val = pd.read_csv("data/5_1k_evDis/spanish-1k/sp_val_160.csv")
    # print(sp_val['label'].value_counts())
    sp_ts = pd.read_csv("data/5_1k_evDis/spanish-1k/sp_ts_200.csv")
    # print(sp_ts['label'].value_counts())
    sp_tr.pop(sp_tr.columns[0])
    sp_val.pop(sp_val.columns[0])
    sp_ts.pop(sp_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = sp_tr.dropna() , sp_ts.dropna() , sp_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_sp_1k_unEvDis_data():
    # print("Spanish:")
    sp_tr = pd.read_csv("data/5_1k_unEvDis/spanish-1k/sp_tr_642.csv")
    # print(sp_tr['label'].value_counts())
    sp_val = pd.read_csv("data/5_1k_unEvDis/spanish-1k/sp_val_159.csv")
    # print(sp_val['label'].value_counts())
    sp_ts = pd.read_csv("data/5_1k_unEvDis/spanish-1k/sp_ts_199.csv")
    # print(sp_ts['label'].value_counts())
    sp_tr.pop(sp_tr.columns[0])
    sp_val.pop(sp_val.columns[0])
    sp_ts.pop(sp_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = sp_tr.dropna() , sp_ts.dropna() , sp_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_sp_difQ_evDis_data():
    # print("Spanish:")
    sp_tr = pd.read_csv("data/difQ_evDis/spanish-1178/sp_tr_758.csv")
    sp_val = pd.read_csv("data/difQ_evDis/spanish-1178/sp_val_188.csv")
    sp_ts = pd.read_csv("data/difQ_evDis/spanish-1178/sp_ts_234.csv")
    sp_tr.pop(sp_tr.columns[0])
    sp_val.pop(sp_val.columns[0])
    sp_ts.pop(sp_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = sp_tr.dropna() , sp_ts.dropna() , sp_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

def read_sp_difQ_unEvDis_data():
    # print("Spanish:")
    sp_tr = pd.read_csv("data/difQ_unEvDis/spanish-1178/sp_tr_754.csv")
    sp_val = pd.read_csv("data/difQ_unEvDis/spanish-1178/sp_val_188.csv")
    sp_ts = pd.read_csv("data/difQ_unEvDis/spanish-1178/sp_ts_234.csv")
    sp_tr.pop(sp_tr.columns[0])
    sp_val.pop(sp_val.columns[0])
    sp_ts.pop(sp_ts.columns[0])
    #total 12 empty records (8 in tr, 1 in val, 3 in test)
    tr_df, ts_df, val_df = sp_tr.dropna() , sp_ts.dropna() , sp_val.dropna()
    # print(ts_df.shape)
    # print(tr_df['label'].value_counts())
    # print(val_df['label'].value_counts())
    # print(ts_df['label'].value_counts())
    
    return tr_df, val_df, ts_df

data_funcs_dict = {
    'read_data': read_data,
    'read_51k_evDis_data': read_51k_evDis_data,
    'read_51k_unEvDis_data': read_51k_unEvDis_data,
    'read_difQ_evDis_data': read_difQ_evDis_data,
    'read_difQ_unEvDis_data': read_difQ_unEvDis_data,
    'read_ar_10k_data': read_ar_10k_data,
    'read_ar_1k_data': read_ar_1k_data,
    'read_ar_1k_unEvDis_data': read_ar_1k_unEvDis_data,
    'read_ar_difQ_evDis_data': read_ar_difQ_evDis_data,
    'read_ar_difQ_unEvDis_data': read_ar_difQ_unEvDis_data,
    'read_ru_10k_data': read_ru_10k_data,
    'read_ru_1k_data': read_ru_1k_data,
    'read_ru_1k_unEvDis_data': read_ru_1k_unEvDis_data,
    'read_ru_difQ_evDis_data': read_ru_difQ_evDis_data,
    'read_ru_difQ_unEvDis_data': read_ru_difQ_unEvDis_data,
    'read_en_1k_data': read_en_1k_data,
    'read_en_difQ_evDis_data': read_en_difQ_evDis_data,
    'read_en_difQ_unEvDis_data': read_en_difQ_unEvDis_data,
    'read_ko_1k_data': read_ko_1k_data,
    'read_ko_difQ_evDis_data': read_ko_difQ_evDis_data,
    'read_ko_difQ_unEvDis_data': read_ko_difQ_unEvDis_data,
    'read_sp_2k_data': read_sp_2k_data,
    'read_sp_1k_data': read_sp_1k_data,
    'read_sp_1k_unEvDis_data': read_sp_1k_unEvDis_data,
    'read_sp_difQ_evDis_data': read_sp_difQ_evDis_data,
    'read_sp_difQ_unEvDis_data': read_sp_difQ_unEvDis_data,
    'test_allEvDis': [read_ar_10k_data, read_ru_10k_data, read_en_1k_data, read_ko_1k_data, read_sp_2k_data],
    'test_51kEvDis': [read_ar_1k_data, read_ru_1k_data, read_en_1k_data, read_ko_1k_data, read_sp_1k_data],
    'test_51kUnEvDis': [read_ar_1k_unEvDis_data, read_ru_1k_unEvDis_data, read_en_1k_data, read_ko_1k_data, read_sp_1k_unEvDis_data],
    'test_difQ_evDis': [read_ar_difQ_evDis_data, read_ru_difQ_evDis_data, read_en_difQ_evDis_data, read_ko_difQ_evDis_data, read_sp_difQ_evDis_data],
    'test_difQ_unEvDis': [read_ar_difQ_unEvDis_data, read_ru_difQ_unEvDis_data, read_en_difQ_unEvDis_data, read_ko_difQ_unEvDis_data, read_sp_difQ_unEvDis_data],
}


