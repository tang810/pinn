import requests
from dotenv import load_dotenv
import csv
import os
from pathlib import Path
import pandas as pd
import chardet
import json
# 加载 .env 文件
load_dotenv()
# 读取环境变量
server_base = os.getenv('server_base')


def read_data_file(file_path:str):
    # 创建 Path 对象
    path = Path(file_path)
    # 提取文件扩展名
    file_suffix = path.suffix
    
    if file_suffix in [".json",".txt",".csv"]:
        if file_suffix==".json":
            code,data=read_json(file_path)
        elif file_suffix==".csv":
            code,data=read_csv(file_path)
        elif file_suffix==".txt":
            code,data=read_txt(file_path)

    else:
        code,data=-1,"不允许的数据格式！"
    return code,data

def read_json(file_path):
    # 读取文件的前几个字节来检测编码
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        encoding = result['encoding']
        print(f"检测到的编码格式: {encoding}")

    # 读取JSON文件
    with open(file_path, 'r', encoding=encoding) as file:
        data = json.load(file)

    # 处理数据并转换为所需的格式
    processed_data = []
    for entry in data:
        if ("time" not in entry.keys()) and ("date" not in entry.keys())and ("日期" not in entry.keys()) and ("时间" not in entry.keys()):
            return -1 ,"数据中缺少时间字段！"
        json_data_={}
        for key in entry.keys():
            
            if key=="time" or key=="date" or key=="日期" or key=="时间":
                time_str = entry[key].replace('UTC=', '')
                json_data_["date"]=time_str
            else:
                if isinstance(entry[key], list):
                    for idx,data_ in enumerate(entry[key]):
                        json_data_["{}_{}".format(key,idx)]=data_
                elif isinstance(entry[key], str):
                    json_data_[key]=entry[key]
        processed_data.append(json_data_)
    # 将处理后的数据转换为DataFrame
    df = pd.DataFrame(processed_data)
    return 0,df

def read_csv(file_path):
    # 读取文件的前几个字节来检测编码
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        encoding = result['encoding']
        print(f"检测到的编码格式: {encoding}")

    # 使用检测到的编码格式读取CSV文件
    df = pd.read_csv(file_path, encoding=encoding)
    return 0,df

def read_txt(file_path):
    # 读取文件的前几个字节来检测编码
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        encoding = result['encoding']
        print(f"检测到的编码格式: {encoding}")

    # 使用检测到的编码格式读取CSV文件
    df = pd.read_csv(file_path, encoding=encoding)
    return 0,df















def upload_file(file_path,img_name, taskid, team_name):
    # FastAPI 服务器地址
    url = "{}/api/uploadBase".format(server_base)

    # 表单数据
    form_data = {
        "taskid": taskid,
        "team_name": team_name
    }

    # 读取文件内容
    with open(file_path, "rb") as file:
        # 发送 POST 请求
        files = [('files', (img_name, file))]
        print(files)
        response = requests.post(url, data=form_data, files=files)

    # 检查响应状态码
    if response.status_code == 200:
        print("文件上传成功:", response.json())
    else:
        print("文件上传失败:", response.text)