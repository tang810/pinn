import numpy as np
import os
import pandas as pd
from natsort import natsorted
from scipy.interpolate import griddata
from tqdm import tqdm
import matplotlib.pyplot as plt

in_dir = os.path.abspath(
    "/home/chunyang/projects/ACDM4SCIENCE/data/foilslice_600"
)
out_dir = os.path.abspath(
    "/home/chunyang/projects/ACDM4SCIENCE/data/foil_slice_processed"
)

def process_file(filepath: str, grid_num=(1024, 1024)) -> np.array:
    df = pd.read_csv(filepath)
    # readin all variables of interests
    u = df.iloc[:, 2]
    v = df.iloc[:, 4]
    p = df.iloc[:, 6]
    # coordinate input
    x = df.iloc[:, 8]
    y = df.iloc[:, 10]
    # mesh generation
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    x_coord, y_coord = np.meshgrid(
        np.linspace(x_min, x_max, grid_num[0]),
        np.linspace(y_min, y_max, grid_num[1])
    )
    x_coord, y_coord = x_coord.flatten(), y_coord.flatten()
    # interpolate variable of interest
    vars = []
    for voi in (u, v, p):
        interpolate = griddata(
            (x, y), voi, (x_coord, y_coord)
        ).reshape(grid_num)
        vars.append(interpolate)
    res = np.stack(vars, axis=0)
    return res
    


if __name__ == "__main__":
    print("Processing file")
    file_path = [
        os.path.join(root, filename)
        for root, folder, filenames in os.walk(in_dir)
        for filename in natsorted(filenames)
    ]
    for idx, fp in enumerate(tqdm(file_path), 0):
        interpolated = process_file(fp)
        np.save(
            os.path.join(out_dir, f"foilslice_{idx:05d}"),  # output path
            interpolated,  # interpolated numpy files
        )
        
    print("Processing done!")
    