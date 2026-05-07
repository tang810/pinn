import scipy.io
import os
import numpy as np
import pandas as pd
# file_path = '/save/magnetic/Data/Dataset/Ship_SNR=0dB/Small_ship_model,L=57.2m,W=8.6m,rz0=90m,ry0=90m,theta=6.2832rad,V=9ms,SNR=0dB.mat'
# 归一化函数
def min_max_normalize_pandas(df):
    return (df - df.min()) / (df.max() - df.min())
    # file_path = '/save/magnetic/Data/Dataset/Ship_SNR=0dB/'
def ReadData_normal(file_path):
    measurepoint_list = []
    data_list = []

    for file in os.listdir(file_path):
        # print(file)
        data = scipy.io.loadmat(os.path.join(file_path,file))
        # print(data.keys())
        # ## 数据
        Data = data['Data']
        # 转换为 Pandas DataFrame 并加入列表
        measurepoint = Data["measurepoint"][0,0]
        # ## 船长 船宽 航向
        L = Data["L"][0,0].repeat(len(measurepoint),0)
        W = Data["W"][0,0].repeat(len(measurepoint),0)
        Theta = Data["theta"][0,0].repeat(len(measurepoint),0)
        MagS0 = Data["MagS0"][0,0].T
        M = Data["M"][0,0]
        Mxyz = M.reshape(16,3)
        mxyz_list = []
        for i in range(16):
            # print(Mxyz[i:i+1,:].shape)
            mxyz = np.tile(Mxyz[i,:],(len(measurepoint),1))
            # print(mxyz.shape)
            mxyz_df = pd.DataFrame(mxyz,columns=[f"m{i}x",f"m{i}y",f"m{i}z"])
            mxyz_list.append(mxyz_df)
        ## 转换为 DataFrame，并命名列
        mxyz_data = pd.concat(mxyz_list,axis=1)
        # print(mxyz_data)
        measurepoint_df = pd.DataFrame(measurepoint,columns=["z","x","y"])
        # print(measurepoint_df.head())
        L_df = pd.DataFrame(L,columns=["L"])
        W_df = pd.DataFrame(W,columns=["W"])
        Theta_df = pd.DataFrame(Theta,columns=["Theta"])
        MagS0_df = pd.DataFrame(MagS0,columns=["Hx","Hy","Hz"])
        # combined_df = pd.concat([measurepoint_df,L_df,W_df,Theta_df,MagS0_df],axis=1)
        combined_df = pd.concat([MagS0_df,measurepoint_df,L_df,W_df,Theta_df,mxyz_data],axis=1)
        # measurepoint_list.append(measurepoint_df)
        data_list.append(combined_df)
    # measurepoint_data = pd.concat(measurepoint_list, axis=0, ignore_index=True)
    data = pd.concat(data_list, axis=0, ignore_index=True)
    # data = min_max_normalize_pandas(data)
    # 选定需要归一化的列
    columns_to_normalize = data.iloc[:, 9:].columns
    print(f'columns_to_normalize:{columns_to_normalize}')
    # 仅对选定列归一化
    data[columns_to_normalize] = (data[columns_to_normalize] - data[columns_to_normalize].min()) / \
                            (data[columns_to_normalize].max() - data[columns_to_normalize].min())
    # print(df)
    # print("measurepoint_data",data.shape)
    # print("前几行数据：\n",data.head())
    # output_path = "/save/magnetic/Data/Dataset/ship_SNR0dB.csv"
    # data.to_csv(output_path)
    return data

def ReadData(file_path):
    measurepoint_list = []
    data_list = []

    for file in os.listdir(file_path):
        # print(file)
        data = scipy.io.loadmat(os.path.join(file_path,file))
        # print(data.keys())
        # ## 数据
        Data = data['Data']
        # 转换为 Pandas DataFrame 并加入列表
        measurepoint = Data["measurepoint"][0,0]
        # ## 船长 船宽 航向
        L = Data["L"][0,0].repeat(len(measurepoint),0)
        W = Data["W"][0,0].repeat(len(measurepoint),0)
        Theta = Data["theta"][0,0].repeat(len(measurepoint),0)
        MagS0 = Data["MagS0"][0,0].T
        M = Data["M"][0,0]
        Mxyz = M.reshape(16,3)
        mxyz_list = []
        for i in range(16):
            # print(Mxyz[i:i+1,:].shape)
            mxyz = np.tile(Mxyz[i,:],(len(measurepoint),1))
            # print(mxyz.shape)
            mxyz_df = pd.DataFrame(mxyz,columns=[f"m{i}x",f"m{i}y",f"m{i}z"])
            mxyz_list.append(mxyz_df)
        ## 转换为 DataFrame，并命名列
        mxyz_data = pd.concat(mxyz_list,axis=1)
        # print(mxyz_data)
        measurepoint_df = pd.DataFrame(measurepoint,columns=["x","y","z"])
        # print(measurepoint_df.head())
        L_df = pd.DataFrame(L,columns=["L"])
        W_df = pd.DataFrame(W,columns=["W"])
        Theta_df = pd.DataFrame(Theta,columns=["Theta"])
        MagS0_df = pd.DataFrame(MagS0,columns=["Hx","Hy","Hz"])
        # combined_df = pd.concat([measurepoint_df,L_df,W_df,Theta_df,MagS0_df],axis=1)
        combined_df = pd.concat([MagS0_df,measurepoint_df,L_df,W_df,Theta_df,mxyz_data],axis=1)
        # measurepoint_list.append(measurepoint_df)
        data_list.append(combined_df)
    # measurepoint_data = pd.concat(measurepoint_list, axis=0, ignore_index=True)
    data = pd.concat(data_list, axis=0, ignore_index=True)
    # 获取当前列名
    columns = list(data.columns)
    # 交换 x 和 z 列的位置
    columns[3], columns[4],columns[5] = columns[5], columns[3], columns[4]

    # 重新排列列顺序
    data = data[columns]
    # data = min_max_normalize_pandas(data)
    # print("measurepoint_data",data.shape)
    # print("前几行数据：\n",data.head())
    # output_path = "/save/magnetic/Data/Dataset/ship_SNR0dB.csv"
    # data.to_csv(output_path)
    return data
