import numpy as np
import os
from natsort import natsorted
from scipy.interpolate import griddata
from multiprocessing import Pool
from tqdm import tqdm


# Function to process each file
def process_file(args):
    idx, file_path, x_origin, y_origin, grid_num, x_coord, y_coord, save_path = args
    data = np.load(file_path)
    physical_quantity_interp = []
    for quantity in [data['u'], data['v'], data['p']]:
        quantity_interp = griddata((x_origin, y_origin), quantity, (x_coord, y_coord), method='cubic').reshape(grid_num)
        physical_quantity_interp.append(quantity_interp)
    # save
    np.save(os.path.join(save_path, os.path.basename(file_path).split('.')[0] + '.npy'),
            np.array(physical_quantity_interp))


def main():
    # data path
    origin_data_path = './origin_data/'
    npz_data_path = './data/'
    save_path = './interp_data/'

    # data range
    temp = np.genfromtxt(os.path.join(origin_data_path, 'vort2d_0_0.dat'), skip_header=2, skip_footer=23544)
    x_min, x_max = np.min(temp[:, 0]), np.max(temp[:, 0])
    y_min, y_max = np.min(temp[:, 1]), np.max(temp[:, 1])
    # new interp grid
    grid_num = (256, 256)
    x_coord, y_coord = np.meshgrid(np.linspace(x_min, x_max, grid_num[0]), np.linspace(y_min, y_max, grid_num[1]))
    x_coord, y_coord = x_coord.flatten(), y_coord.flatten()
    x_origin, y_origin = temp[:, 0], temp[:, 1]

    file_paths = [os.path.join(root, file) for root, _, file_list in os.walk(npz_data_path) for file in
                  natsorted(file_list)]
    # 准备所有需要传递给process_file的参数，包括文件的索引
    args_list = [(idx, file_path, x_origin, y_origin, grid_num, x_coord, y_coord, save_path) for idx, file_path in
                 enumerate(file_paths)]

    with Pool(processes=20) as pool:
        for _ in tqdm(pool.imap_unordered(process_file, args_list), total=len(file_paths)):
            pass


if __name__ == "__main__":
    main()
    print('Interpolation finished!')
