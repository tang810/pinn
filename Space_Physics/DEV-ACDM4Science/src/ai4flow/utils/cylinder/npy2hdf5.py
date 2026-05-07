import os
import numpy as np
from tqdm import tqdm
import h5py


def convert_npy_to_hdf5(npy_folder, hdf5_file):
    # 创建或打开HDF5文件
    with h5py.File(hdf5_file, 'w') as h5f:
        # 遍历目录中的每个文件
        for i in tqdm(range(100000)):
            file_name = f"vort2d_0_{i}.npy"
            file_path = os.path.join(npy_folder, file_name)

            # 读取Numpy数组
            data = np.load(file_path)
            # 将数组存储到HDF5文件中
            h5f.create_dataset(file_name.split('.')[0].split('_')[-1], data=data)


if __name__ == "__main__":
    npy_folder = './interp_data'  # 更新为你的.npy文件所在的文件夹路径
    hdf5_file = './final_data/cylinder.h5'  # 更新为你想要保存HDF5文件的路径
    convert_npy_to_hdf5(npy_folder, hdf5_file)
