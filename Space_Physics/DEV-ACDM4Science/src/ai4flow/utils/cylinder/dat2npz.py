import numpy as np
import os
from natsort import natsorted
from tqdm import tqdm

# data path
save_path = './data/'
path = r'F:\full\Vort2D_Re1000'
# read data
total_data = []
for root, _, file_list in os.walk(path):
    for file in tqdm(natsorted(file_list)[54452:]):
        # print('Processing: ', file)
        data = np.genfromtxt(os.path.join(root, file), skip_header=2, skip_footer=23544)
        # u, v, w, p, Vorticity_x, Vorticity_y, Vorticity_z, Divergence, Q = \
        #     data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6], data[:, 7], data[:, 8], data[:, 9], data[:, 10]
        u, v, p = data[:, 2], data[:, 3], data[:, 5]
        # save npz
        np.savez(os.path.join(save_path, file[:-4] + '.npz'), u=u, v=v, p=p)
