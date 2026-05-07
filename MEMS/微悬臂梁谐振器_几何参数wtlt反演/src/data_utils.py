# 目录结构（复制到项目根目录）
# ├── main.py
# └── src/
#     ├── data_utils.py
#     ├── models.py
#     ├── losses.py
#     ├── trainer.py
#     ├── infer.py
#     └── viz.py

# ================================
# File: src/data_utils.py
# ================================
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --------- 加载数据（严格按你提供的实现整合） ---------
def load_data(file_path='./wt_lt/data/wt_lt_big.h5', scaler_path='./wt_lt/data/norm_scalers_big.pkl'):
    # ====================
    # 1. 加载数据 + 常数参数
    # ====================
    with pd.HDFStore(file_path, 'r') as store:
        df = store['data_valid']  # 已去除NaN的样本数据
        constants = store.get_storer('data').attrs.constants
        phi_full = np.array(constants['phi'])
    # ====================
    # 2. 设置全局常数参数（转导因子）
    # ====================
    trans_factor_global = (
        constants['eps_0'] * constants['V'] * constants['electrode_length'] * constants['t'] 
        / constants['d']**2
    )
    FAC_FIXED = constants['Vac_ground'] * trans_factor_global  # Fac 是常量
    constants['mcoef_b'] = df['m_coef_b'].iloc[0]
    constants['kcoef_b'] = df['k_coef_b'].iloc[0]
    constants['kcoef_b3'] = df['k_coef_b3'].iloc[0]
    # ====================
    # 3. 从 freq, m_c 推出 omega, y
    # ====================
    def preprocess_sample(row, trans_factor):
        freq = np.array(row['freq'])  # Hz
        m_c = np.array(row['m_c'])    # 原始响应
        omega = 2 * np.pi * freq      # rad/s
        y = m_c * 1e-9 / (omega * trans_factor)
        return freq, m_c, omega, y

    omega_list, y_list, wt_list, lt_list = [], [], [], []
    freq_list, mc_list = [], []
    M_list, dkt_list, dk3t_list, Fac_list ,C_list = [], [], [], [], []

    for _, row in df.iterrows():
        freq, m_c, omega, y = preprocess_sample(row, trans_factor_global)
        omega_list.append(omega)
        y_list.append(y)
        freq_list.append(freq)
        mc_list.append(m_c)
        wt_list.append(row['w_t'])
        lt_list.append(row['l_t'])

        # 额外参数提取
        M = row['Mass']
        dkt = row['k_tt'] - row['k_e']
        dk3t = row['k_t3'] - row['k_e3']
        Q = constants['Q']
        c = np.sqrt(M * (row['k_tt'])) / Q

        M_list.append(M)
        dkt_list.append(dkt)
        dk3t_list.append(dk3t)
        Fac_list.append(FAC_FIXED)  # 所有样本都用固定值
        C_list.append(c)
    print(f"Processed {len(omega_list)} samples.")

    scalers = {}

    # output: w_t, l_t
    wt_arr = np.array(wt_list).reshape(-1, 1)
    lt_arr = np.array(lt_list).reshape(-1, 1)

    scalers['wt'] = MinMaxScaler().fit(wt_arr)
    scalers['lt'] = MinMaxScaler().fit(lt_arr)

    wt_norm = scalers['wt'].transform(wt_arr).flatten()
    lt_norm = scalers['lt'].transform(lt_arr).flatten()

    # para: M, Δkt, Δkt3, Fac, c
    M_arr = np.array(M_list).reshape(-1, 1)
    dkt_arr = np.array(dkt_list).reshape(-1, 1)
    dk3t_arr = np.array(dk3t_list).reshape(-1, 1)
    Fac_arr = np.array(Fac_list).reshape(-1, 1)  # constant
    c_arr = np.array(C_list).reshape(-1, 1)

    scalers['M'] = MinMaxScaler().fit(M_arr)
    scalers['dkt'] = MinMaxScaler().fit(dkt_arr)
    scalers['dk3t'] = MinMaxScaler().fit(dk3t_arr)
    scalers['c'] = MinMaxScaler().fit(c_arr)

    # 不归一化 Fac
    Fac_norm = np.ones_like(Fac_arr.flatten())
    M_norm = scalers['M'].transform(M_arr).flatten()
    dkt_norm = scalers['dkt'].transform(dkt_arr).flatten()
    dk3t_norm = scalers['dk3t'].transform(dk3t_arr).flatten()
    c_norm = scalers['c'].transform(c_arr).flatten()

    # ====================
    # 8. 特征统计提取函数（基于样本级别归一化 + y放大）
    # ====================
    Y_SCALE_FACTOR = 1e8  # 将 y 放大，便于统计处理

    def extract_statistical_features_per_sample(omega_list, y_list):
        feature_list = []
        sample_scalers = []
        omega_norm_list = []
        y_norm_list = []

        for omega, y in zip(omega_list, y_list):
            y_scaled = y * Y_SCALE_FACTOR
            omega_min, omega_max = np.min(omega), np.max(omega)
            y_min, y_max = np.min(y_scaled), np.max(y_scaled)
            omega_norm = (omega - omega_min) / (omega_max - omega_min + 1e-8)
            y_norm = (y_scaled - y_min) / (y_max - y_min + 1e-8)

            feat = [
                omega_min, omega_max, np.std(omega), np.mean(omega),
                y_min, y_max, np.std(y_scaled), np.mean(y_scaled),
            ]
            feature_list.append(feat)
            omega_norm_list.append(omega_norm)
            y_norm_list.append(y_norm)
            sample_scalers.append({
                'omega_min': float(omega_min), 'omega_max': float(omega_max),
                'y_min': float(y_min), 'y_max': float(y_max)
            })

        return np.array(feature_list), sample_scalers, omega_norm_list, y_norm_list

    # 特征编码作为神经网络输入
    X_feat_raw, per_sample_scalers, omega_norm_all, y_norm_all = extract_statistical_features_per_sample(omega_list, y_list)

    # 将 per_sample_scalers 合并存入 scalers 中统一保存
    scalers['X_feat'] = StandardScaler().fit(X_feat_raw)
    scalers['per_sample_scalers'] = per_sample_scalers

    X_feat = scalers['X_feat'].transform(X_feat_raw)
    Y_target = np.stack([wt_norm, lt_norm], axis=1)
    Y_phys = np.stack([M_norm, dkt_norm, dk3t_norm, c_norm], axis=1)

    phi = list(df['phi'].values)

    # 统一保存所有归一化器和样本级参数
    os.makedirs(os.path.dirname(scaler_path) or ".", exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scalers, f)
    print("Saved all scalers (including per-sample) to ", scaler_path)
    
    return (omega_list, y_list, wt_list, lt_list, freq_list, mc_list,
            M_list, dkt_list, dk3t_list, Fac_list, C_list,
            phi_full, constants, phi,
            X_feat, Y_target, Y_phys, omega_norm_all, y_norm_all,
            scalers, per_sample_scalers)

# --------- 模型/数据集/Loader（按你提供的实现拆分） ---------
def _as_list_of_arrays(obj):
    """
    将输入安全地转为由 numpy.ndarray 组成的 list。
    支持：list[np.ndarray] / tuple[np.ndarray] / dict[key->np.ndarray] / np.ndarray(对象数组)。
    """
    if isinstance(obj, dict):
        return [np.asarray(v) for v in obj.values()]
    if isinstance(obj, (list, tuple)):
        return [np.asarray(v) for v in obj]
    obj = np.asarray(obj, dtype=object)
    return [np.asarray(x) for x in obj.tolist()]

class PINNDataset(Dataset):
    def __init__(self, X_feat, Y_target, Y_phys, phi, omega_norm_all, y_norm_all):
        self.X_feat = torch.tensor(X_feat, dtype=torch.float32)
        self.Y_target = torch.tensor(Y_target, dtype=torch.float32)
        self.Y_phys = torch.tensor(Y_phys, dtype=torch.float32)
        self.phi = phi
        self.omega_norm_all = omega_norm_all
        self.y_norm_all = y_norm_all

    def __len__(self):
        return len(self.X_feat)

    def __getitem__(self, idx):
        return {
            'X': self.X_feat[idx],
            'Y_target': self.Y_target[idx],
            'Y_phys': self.Y_phys[idx],
            'phi': self.phi[idx],
            'omega': self.omega_norm_all[idx],
            'y': self.y_norm_all[idx]
        }

def variable_length_collate(batch):
    batch_dict = {key: [item[key] for item in batch] for key in batch[0]}
    for key in ['X', 'Y_target', 'Y_phys']:
        batch_dict[key] = torch.stack(batch_dict[key])  # only stack fixed-length tensors
    return batch_dict

def build_dataloaders(X_feat, Y_target, Y_phys, phi, omega_norm_all, y_norm_all, batch_size=32):
    idx_train, idx_test = train_test_split(np.arange(len(X_feat)), test_size=0.2, random_state=42)

    train_set = PINNDataset(
        X_feat[idx_train], Y_target[idx_train], Y_phys[idx_train],
        [phi[i] for i in idx_train],
        [omega_norm_all[i] for i in idx_train],
        [y_norm_all[i] for i in idx_train]
    )

    test_set = PINNDataset(
        X_feat[idx_test], Y_target[idx_test], Y_phys[idx_test],
        [phi[i] for i in idx_test],
        [omega_norm_all[i] for i in idx_test],
        [y_norm_all[i] for i in idx_test]
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=variable_length_collate)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, collate_fn=variable_length_collate)

    return train_loader, test_loader

def build_eval_loader_from_prepared(
    X_feat,
    Y_target,
    Y_phys,
    phi_full,
    omega_norm_all,
    y_norm_all,
    batch_size=32,
    shuffle=False
):
    """
    输入均为你已准备好的数组/列表/字典，返回一个与原 test_loader 完全一致的 DataLoader。
    """
    X_feat   = np.array(X_feat)
    Y_target = np.array(Y_target)
    Y_phys   = np.array(Y_phys)

    phi_list   = _as_list_of_arrays(phi_full)
    omega_list = _as_list_of_arrays(omega_norm_all)
    y_list     = _as_list_of_arrays(y_norm_all)

    n = len(X_feat)
    assert len(Y_target) == n and len(Y_phys) == n, "Y_target/Y_phys 与 X_feat 数量不一致"
    assert len(phi_list) == n and len(omega_list) == n and len(y_list) == n, "phi/omega/y 与 X_feat 数量不一致"

    dataset = PINNDataset(
        X_feat=X_feat,
        Y_target=Y_target,
        Y_phys=Y_phys,
        phi=phi_list,
        omega_norm_all=omega_list,
        y_norm_all=y_list
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=variable_length_collate)
    return loader








