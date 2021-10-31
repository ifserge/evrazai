import pandas as pd
import numpy as np
import os, random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def metric(answers, user_csv):
    delta_c = np.abs(np.array(answers[:,1]) - np.array(user_csv[:,1]))
    hit_rate_c = np.int64(delta_c < 0.02)

    delta_t = np.abs(np.array(answers[:,0]) - np.array(user_csv[:,0]))
    hit_rate_t = np.int64(delta_t < 20)

    N = np.size(answers[:,0])

    return np.sum(hit_rate_c + hit_rate_t) / 2 / N

fix_all_seeds(42) 

DATA_DIR = '../data/'

config = {
    'lom_hidden_dim': 4,
    'sip_hidden_dim': 4,
    'main_hidden_dim': 32,
    'head_hidden_dim': 8,
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

df_train_static = pd.read_csv(DATA_DIR + 'plavki_train.csv').drop_duplicates('NPLV').reset_index(drop=True)
df_test_static = pd.read_csv(DATA_DIR + 'plavki_test.csv')

target = pd.read_csv(DATA_DIR + 'target_train.csv')
df_train_static = df_train_static.merge(target, on='NPLV', how='left')

for plavki in [df_train_static, df_test_static]:
    plavki['plavka_VR_NACH'] = pd.to_datetime(plavki['plavka_VR_NACH'])
    plavki['plavka_VR_KON'] = pd.to_datetime(plavki['plavka_VR_KON'])
    plavki['plav_seconds'] = (((plavki.plavka_VR_KON.values - plavki.plavka_VR_NACH.values))*1e-9).astype(int)
    
chugun_train = pd.read_csv(DATA_DIR + 'chugun_train.csv')
chugun_train.columns = ['NPLV'] + [x+'_ch' for x in chugun_train.columns.tolist()[1:-1]] + ['DATA_ZAMERA']
chugun_test = pd.read_csv(DATA_DIR + 'chugun_test.csv')
chugun_test.columns = ['NPLV'] + [x+'_ch' for x in chugun_test.columns.tolist()[1:-1]] + ['DATA_ZAMERA']

df_train_static = df_train_static.merge(chugun_train, on='NPLV', how='left').drop(columns = ['DATA_ZAMERA'])
df_test_static = df_test_static.merge(chugun_test, on='NPLV', how='left').drop(columns = ['DATA_ZAMERA'])

lom_train = pd.read_csv(DATA_DIR + 'lom_train.csv')
lom_test = pd.read_csv(DATA_DIR + 'lom_test.csv')

for f in ['VDL','NML']:
    le = LabelEncoder()
    le.fit(lom_train[f])
    lom_train[f] = le.transform(lom_train[f])
    lom_test[f] = le.transform(lom_test[f])
    
lom_cat_feats_dict = {
    'VDL': (lom_train['VDL'].max() + 1, 3),
    'NML': (lom_train['NML'].max() + 1, 3)
}
lom_cat_feats = ['VDL', 'NML']
lom_num_feats = ['VES']

ss = StandardScaler()
lom_train[lom_num_feats] = np.log1p(lom_train[lom_num_feats])
lom_test[lom_num_feats] = np.log1p(lom_test[lom_num_feats])

ss.fit(lom_train[lom_num_feats])
lom_train[lom_num_feats] = ss.transform(lom_train[lom_num_feats])
lom_test[lom_num_feats] = ss.transform(lom_test[lom_num_feats])
    
sip_train = pd.read_csv(DATA_DIR + 'sip_train.csv')
sip_test = pd.read_csv(DATA_DIR + 'sip_test.csv')

for df in [sip_train, sip_test]:
    df['DAT_OTD'] = pd.to_datetime(df['DAT_OTD'])

time_cols = ['NPLV', 'plavka_VR_NACH', 'plav_seconds']
sip_train = sip_train.merge(df_train_static[time_cols], on='NPLV', how='left')
sip_train['time_from_start'] = (sip_train['DAT_OTD'] - sip_train['plavka_VR_NACH']).dt.seconds
sip_train = sip_train.loc[sip_train.time_from_start <= sip_train.plav_seconds].drop(columns=time_cols[1:]+['DAT_OTD']).reset_index(drop=True)
sip_test = sip_test.merge(df_test_static[time_cols], on='NPLV', how='left')
sip_test['time_from_start'] = (sip_test['DAT_OTD'] - sip_test['plavka_VR_NACH']).dt.seconds
sip_test = sip_test.loc[sip_test.time_from_start <= sip_test.plav_seconds].drop(columns=time_cols[1:]+['DAT_OTD']).reset_index(drop=True)

    
for f in ['VDSYP', 'NMSYP']:
    le = LabelEncoder()
    sip_train[f] = sip_train[f].astype(str)
    sip_test[f] = sip_test[f].astype(str)
    common = set(sip_train[f].unique().tolist()) & set(sip_test[f].unique().tolist())
    sip_train.loc[~sip_train[f].isin(common), f] = 'None'
    sip_test.loc[~sip_test[f].isin(common), f] = 'None'
    le.fit(['None'] + sip_train[f].unique().tolist())
    sip_train[f] = le.transform(sip_train[f])
    sip_test[f] = le.transform(sip_test[f])
    
sip_cat_feats_dict = {
    'VDSYP': (max(sip_train['VDSYP'].max() + 1,sip_test['VDSYP'].max() + 1), 3),
    'NMSYP': (max(sip_train['NMSYP'].max() + 1,sip_test['NMSYP'].max() + 1), 3)
}

sip_cat_feats = ['VDSYP', 'NMSYP']
sip_num_feats = ['VSSYP']
ss = StandardScaler()
ss.fit(sip_train[sip_num_feats])
sip_train[sip_num_feats] = ss.transform(sip_train[sip_num_feats])
sip_test[sip_num_feats] = ss.transform(sip_test[sip_num_feats])

def make_index_df(df):
    index_dfs = []
    for g,q in df.groupby('NPLV'):
        ts = q.plav_seconds.values[0] + 1
        index_df = pd.DataFrame({
            'NPLV': [g]*ts,
            'time_from_start': list(range(ts))
        })
        index_dfs.append(index_df)
    return pd.concat(index_dfs, ignore_index=True).reset_index(drop=True)
train_index_df = make_index_df(df_train_static)
test_index_df = make_index_df(df_test_static)

gas_train = pd.read_csv(DATA_DIR + 'gas_train.csv')
gas_test = pd.read_csv(DATA_DIR + 'gas_test.csv')

for df in [gas_train, gas_test]:
    df['Time'] = pd.to_datetime(df['Time'])

time_cols = ['NPLV', 'plavka_VR_NACH', 'plav_seconds']
gas_train = gas_train.merge(df_train_static[time_cols], on='NPLV', how='left')
gas_train['time_from_start'] = (gas_train['Time'] - gas_train['plavka_VR_NACH']).dt.seconds
gas_train = gas_train.loc[gas_train.time_from_start <= gas_train.plav_seconds].drop(columns=time_cols[1:]+['Time']).reset_index(drop=True)
gas_test = gas_test.merge(df_test_static[time_cols], on='NPLV', how='left')
gas_test['time_from_start'] = (gas_test['Time'] - gas_test['plavka_VR_NACH']).dt.seconds
gas_test = gas_test.loc[gas_test.time_from_start <= gas_test.plav_seconds].drop(columns=time_cols[1:]+['Time']).reset_index(drop=True)

gas_train = train_index_df.merge(gas_train, on=['NPLV','time_from_start'], how='left')
gas_test = test_index_df.merge(gas_test, on=['NPLV','time_from_start'], how='left')

produv_train = pd.read_csv(DATA_DIR + 'produv_train.csv')
produv_test = pd.read_csv(DATA_DIR + 'produv_test.csv')

for df in [produv_train, produv_test]:
    df['SEC'] = pd.to_datetime(df['SEC'])

time_cols = ['NPLV', 'plavka_VR_NACH', 'plav_seconds']
produv_train = produv_train.merge(df_train_static[time_cols], on='NPLV', how='left')
produv_train['time_from_start'] = (produv_train['SEC'] - produv_train['plavka_VR_NACH']).dt.seconds
produv_train = produv_train.loc[produv_train.time_from_start <= produv_train.plav_seconds].drop(columns=time_cols[1:]+['SEC']).reset_index(drop=True)
produv_test = produv_test.merge(df_test_static[time_cols], on='NPLV', how='left')
produv_test['time_from_start'] = (produv_test['SEC'] - produv_test['plavka_VR_NACH']).dt.seconds
produv_test = produv_test.loc[produv_test.time_from_start <= produv_test.plav_seconds].drop(columns=time_cols[1:]+['SEC']).reset_index(drop=True)

train_df = gas_train.merge(produv_train, on=['NPLV', 'time_from_start'], how='left')
test_df = gas_test.merge(produv_test, on=['NPLV', 'time_from_start'], how='left')

ts_num_fts = ['V', 'T', 'O2', 'N2', 'H2', 'CO2', 'CO', 'AR', 
              'O2_pressure', 'RAS', 'POL']
train_df[ts_num_fts] = train_df[ts_num_fts].ffill().bfill().interpolate().fillna(0)
test_df[ts_num_fts] = test_df[ts_num_fts].ffill().bfill().interpolate().fillna(0)

ss = StandardScaler()
ss.fit(train_df[ts_num_fts])
train_df[ts_num_fts] = ss.transform(train_df[ts_num_fts])
test_df[ts_num_fts] = ss.transform(test_df[ts_num_fts])

static_cat_feats = ['plavka_NAPR_ZAD', 'plavka_TIPE_FUR', 
                    'plavka_TIPE_GOL', 'plavka_NMZ']

static_num_feats = ['plavka_STFUT','plavka_ST_FURM','plav_seconds','plavka_ST_GOL',
                    'VES_ch', 'T_ch', 'SI_ch', 'MN_ch', 'S_ch', 'P_ch', 'CR_ch',
                    'NI_ch', 'CU_ch', 'V_ch', 'TI_ch']

staic_cat_feats_dict = {}
for f in static_cat_feats:
    le = LabelEncoder()
    common = set(df_train_static[f].unique().tolist()) & set(df_test_static[f].unique().tolist())
    drop_c = []
    for c in common:
        tt = df_train_static.loc[df_train_static[f] == c, f].shape[0] + df_test_static.loc[df_test_static[f] == c, f].shape[0]
        if tt < 5:
            drop_c.append(c)
    print(drop_c)
    common = common - set(drop_c)
    df_train_static.loc[~df_train_static[f].isin(common), f] = 'None'
    df_test_static.loc[~df_test_static[f].isin(common), f] = 'None'
    le.fit(df_train_static[f])
    df_train_static[f] = le.transform(df_train_static[f])
    df_test_static[f] = le.transform(df_test_static[f])
    staic_cat_feats_dict[f] = (df_train_static[f].max() + 1, 3)
    
ss = StandardScaler()
ss.fit(df_train_static[static_num_feats])
df_train_static[static_num_feats] = ss.transform(df_train_static[static_num_feats])
df_test_static[static_num_feats] = ss.transform(df_test_static[static_num_feats])

target_fts = ['TST','C']

target_scaler = StandardScaler()
df_train_static[target_fts] = target_scaler.fit_transform(df_train_static[target_fts])

class EvrazDatasetV(Dataset):
    def __init__(self, static_train_df, lom_df, sip_df, ts_df):
        self.static_train_df = static_train_df.copy()
        self.lom_df = lom_df.copy()
        self.sip_df = sip_df.copy()
        self.ts_df = ts_df.copy()
        
        self.nlmv =  self.static_train_df.NPLV.unique()

    def __len__(self):
        return len(self.nlmv)

    def __getitem__(self, idx):    
        nlmv = self.nlmv[idx]
        
        return {"static_cat": self.static_train_df.loc[self.static_train_df.NPLV == nlmv, static_cat_feats].values[0],
                "static_numeric": self.static_train_df.loc[self.static_train_df.NPLV == nlmv, static_num_feats].values[0],
                "lom_cat": self.lom_df.loc[self.lom_df.NPLV == nlmv, lom_cat_feats].values,
                "lom_numeric": self.lom_df.loc[self.lom_df.NPLV == nlmv, lom_num_feats].values,
                "sip_cat": self.sip_df.loc[self.sip_df.NPLV == nlmv, sip_cat_feats].values,
                "sip_numeric": self.sip_df.loc[self.sip_df.NPLV == nlmv, sip_num_feats].values,
                "ts_numeric": self.ts_df.loc[self.ts_df.NPLV == nlmv, ts_num_fts].values,
                "y": self.static_train_df.loc[self.static_train_df.NPLV == nlmv, target_fts].values[0]
               }
    
class AttentionWeightedAverage(nn.Module):
    def __init__(self, hidden_dim: int, return_attention: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.return_attention = return_attention
        
        self.attention_vector = nn.Parameter(
            torch.empty(self.hidden_dim, dtype=torch.float32),
            requires_grad=True,
        )
        nn.init.xavier_normal_(self.attention_vector.unsqueeze(-1))

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ):
        logits = x.matmul(self.attention_vector)
        ai = (logits - logits.max()).exp()

        ai = ai * mask
        att_weights = ai / (ai.sum(dim=1, keepdim=True) + 1e-12)
        weighted_input = x * att_weights.unsqueeze(-1)
        output = weighted_input.sum(dim=1)

        if self.return_attention:
            return output, att_weights
        else:
            return output, None
        

CONV_FEAT_SIZE = 16
CONV_DELS = [1,2,4,8,16,32]

class EvrazModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.static_cat_embs = nn.ModuleList([
            nn.Embedding(b[0], b[1])
        for a,b in staic_cat_feats_dict.items()] )
        
        self.lom_cat_embs = nn.ModuleList([
            nn.Embedding(b[0], b[1])
        for a,b in lom_cat_feats_dict.items()] )
        
        self.sip_cat_embs = nn.ModuleList([
            nn.Embedding(b[0], b[1])
        for a,b in sip_cat_feats_dict.items()] )
        
        self.lom_input_size = sum([b[1] for _,b in lom_cat_feats_dict.items()]) + len(lom_num_feats)
        self.lom_nn = nn.GRU(self.lom_input_size, config['lom_hidden_dim'], batch_first=True)
        self.sip_input_size = sum([b[1] for _,b in sip_cat_feats_dict.items()]) + len(sip_num_feats)
        self.sip_nn = nn.GRU(self.sip_input_size, config['sip_hidden_dim'], batch_first=True)
        self.static_input_size = sum([b[1] for _,b in staic_cat_feats_dict.items()]) + len(static_num_feats)
        
        self.main_nn_input_size = len(ts_num_fts) + self.static_input_size + config['lom_hidden_dim'] + config['sip_hidden_dim']
        self.main_nn = nn.GRU(self.main_nn_input_size, config['main_hidden_dim'], batch_first=True)
        self.conv_attn = AttentionWeightedAverage(config['main_hidden_dim'])
        self.head_input_size = config['main_hidden_dim']
          
        
        self.head = nn.Sequential(
            nn.LayerNorm(self.head_input_size),
            nn.Linear(self.head_input_size, config['head_hidden_dim']),
            nn.LayerNorm(config['head_hidden_dim']),
            nn.SiLU(),
            nn.Linear(config['head_hidden_dim'], 2)
        )
                
    def get_conv_size(self, i):
        if i == 0:
            return len(ts_num_fts)
        return CONV_FEAT_SIZE
    
    def forward(self, static_cat, static_numeric, 
                lom_cat, lom_numeric,
                sip_cat, sip_numeric,
                ts_numeric):
        ss = []
        for i,m in enumerate(self.static_cat_embs):
            ss.append(m(static_cat[:,i]))
        static = torch.cat(ss + [static_numeric], dim=-1)
        
        lom = []
        for i,m in enumerate(self.lom_cat_embs):
            lom.append(m(lom_cat[:,:,i]))
        lom = torch.cat(lom + [lom_numeric], dim=-1)
        _, lom = self.lom_nn(lom)
        
        sip = []
        for i,m in enumerate(self.sip_cat_embs):
            sip.append(m(sip_cat[:,:,i]))
        sip = torch.cat(sip + [sip_numeric], dim=-1)
        _, sip = self.sip_nn(sip)
        
        static_x = torch.cat([static, lom[:,-1], sip[:,-1]], dim=-1)
        static_x = static_x.unsqueeze(dim=-1).transpose(1,2).repeat(1,ts_numeric.shape[1],1)
        y = torch.cat([static_x, ts_numeric], dim=-1)
        y, _ = self.main_nn(y)
        y = self.conv_attn(y, torch.ones((y.shape[0], y.shape[1])).to(y.device))[0]
                
        return self.head(y)


kf = KFold(n_splits=5, shuffle=True, random_state=42)

nplvs = df_train_static.NPLV.unique()

ifold = 0
for vipl_train, vipl_valid in kf.split(nplvs):
    nplv_train = [nplvs[x] for x in vipl_train if nplvs[x] not in [511156,512299]]
    nplv_valid = [nplvs[x] for x in vipl_valid if nplvs[x] not in [511156,512299]]

    train_ds = EvrazDatasetV(df_train_static.loc[df_train_static.NPLV.isin(nplv_train)].reset_index(drop=True),
                             lom_train.loc[lom_train.NPLV.isin(nplv_train)].reset_index(drop=True),
                             sip_train.loc[sip_train.NPLV.isin(nplv_train)].reset_index(drop=True),
                             train_df.loc[train_df.NPLV.isin(nplv_train)].reset_index(drop=True)
                            )
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=1, num_workers=0)
    valid_ds = EvrazDatasetV(df_train_static.loc[df_train_static.NPLV.isin(nplv_valid)].reset_index(drop=True),
                             lom_train.loc[lom_train.NPLV.isin(nplv_valid)].reset_index(drop=True),
                             sip_train.loc[sip_train.NPLV.isin(nplv_valid)].reset_index(drop=True),
                             train_df.loc[train_df.NPLV.isin(nplv_valid)].reset_index(drop=True)
                            )
    valid_dl = DataLoader(valid_ds, shuffle=False, batch_size=1, num_workers=0)

    model = EvrazModel(config).to(device)
    criterion1 = nn.L1Loss()
    criterion2 = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    model.train();
    optimizer.zero_grad()

    best_metric = 0.0
    for epoch in range(30):
        for x in train_dl:
            out = model(x["static_cat"].long().to(device),
                        x['static_numeric'].float().to(device),
                        x['lom_cat'].long().to(device),
                        x['lom_numeric'].float().to(device),
                        x['sip_cat'].long().to(device),
                        x['sip_numeric'].float().to(device),
                        x['ts_numeric'].float().to(device)
                       )
            loss1 = criterion1(out[:,0], x['y'].float().to(device)[:,0])
            loss2 = criterion2(out[:,1], x['y'].float().to(device)[:,1])

            loss = loss1*0.5+loss2

            loss.backward()
            optimizer.step() 
            optimizer.zero_grad()

        model.eval()
        outs, ys = [], []
        with torch.no_grad():
            for x in valid_dl:
                out = model(x["static_cat"].long().to(device),
                            x['static_numeric'].float().to(device),
                            x['lom_cat'].long().to(device),
                            x['lom_numeric'].float().to(device),
                            x['sip_cat'].long().to(device),
                            x['sip_numeric'].float().to(device),
                            x['ts_numeric'].float().to(device)
                           )
                ys.append(x['y'].numpy())
                outs.append(out.detach().cpu().numpy())

        outs = np.concatenate(outs, axis=0)
        ys = np.concatenate(ys, axis=0)

        outs = target_scaler.inverse_transform(outs)
        ys = target_scaler.inverse_transform(ys)

        m = metric(ys, outs)
        print(epoch, m)
        if m > best_metric:
            torch.save(model.state_dict(), 'v_model' + str(ifold) + '.pth')
            best_metric = m
        model.train()
        optimizer.zero_grad()
    ifold += 1