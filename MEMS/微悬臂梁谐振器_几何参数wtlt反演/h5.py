import numpy as np
import pandas as pd

h5_path = r"C:\Users\se42\Desktop\案例\PINN4Science\MEMS\微悬臂梁谐振器_几何参数wtlt反演\data\wt_lt_big.h5"
out_path = r"C:\Users\se42\Desktop\案例\PINN4Science\MEMS\微悬臂梁谐振器_几何参数wtlt反演\data\wt_lt_big_compact.npz"

with pd.HDFStore(h5_path, "r") as store:
    df = store["data_valid"]
    constants = store.get_storer("data").attrs.constants

trans_factor = (
    constants["eps_0"] * constants["V"] * constants["electrode_length"] * constants["t"]
    / constants["d"] ** 2
)

features = []
targets = []

for _, row in df.iterrows():
    freq = np.asarray(row["freq"], dtype=np.float32)
    mc = np.asarray(row["m_c"], dtype=np.float32)
    omega = (2.0 * np.pi * freq).astype(np.float32)
    y = (mc * 1e-9 / (omega * trans_factor + 1e-30)).astype(np.float32)
    y_scaled = y * 1e8

    features.append([
        omega.min(), omega.max(), omega.std(), omega.mean(),
        y_scaled.min(), y_scaled.max(), y_scaled.std(), y_scaled.mean(),
    ])
    targets.append([row["w_t"], row["l_t"]])

np.savez_compressed(
    out_path,
    X_raw=np.asarray(features, dtype=np.float32),
    Y_raw=np.asarray(targets, dtype=np.float32),
)

print("saved:", out_path)
