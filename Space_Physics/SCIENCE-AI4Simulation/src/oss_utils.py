# -*- coding: utf-8 -*-
import oss2
import os
from oss2.credentials import EnvironmentVariableCredentialsProvider
from dotenv import load_dotenv
load_dotenv()
# 从环境变量中获取访问凭证。运行本代码示例之前，请确保已设置环境变量OSS_ACCESS_KEY_ID和OSS_ACCESS_KEY_SECRET。
auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())

# 填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
endpoint = "https://oss-cn-beijing.aliyuncs.com"
# 填写自定义域名，例如example.com。
cname = 'https://scien42.tech'

# 填写Endpoint对应的Region信息，例如cn-hangzhou。注意，v4签名下，必须填写该参数
region = "cn-beijing"

def oss_upload(bucket_name:str ,save_path:str,file_obj):

    bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)

    # 填写Object完整路径和本地文件的完整路径。Object完整路径中不能包含Bucket名称。
    # 如果未指定本地路径，则默认从示例程序所属项目对应本地路径中上传文件。
    return bucket.put_object(save_path, file_obj)

def oss_upload_by_path(bucket_name:str ,save_path:str,file_path:str):

    bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)

    # 填写Object完整路径和本地文件的完整路径。Object完整路径中不能包含Bucket名称。
    # 如果未指定本地路径，则默认从示例程序所属项目对应本地路径中上传文件。
    return bucket.put_object_from_file(save_path, file_path)


def oss_list(bucket_name:str ,path:str):
    # yourBucketName填写存储空间名称。
    bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)

    # 列举fun文件夹下的文件与子文件夹名称，不列举子文件夹下的文件。
    files=[]
    for obj in oss2.ObjectIterator(bucket, prefix=path):
        # 通过is_prefix方法判断obj是否为文件夹。（请注意判断是否为文件夹需要配置delimiter和prefix来完成模拟文件夹功能）
        if not obj.is_prefix():  # 判断obj为文件夹。
            files.append(os.path.basename(str(obj.key)))
    return files

def oss_del_list(bucket_name:str ,file_path:str,file_list:list):
    # yourBucketName填写存储空间名称。
    bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)
    exists=[]
    not_exists=[]
    for file_name in file_list:
        exist = bucket.object_exists(os.path.join(file_path,file_name))
        if exist:
            del_results=bucket.delete_object(os.path.join(file_path,file_name)) 
            if del_results.status==204:
                exists.append(file_name)
        else:
            not_exists.append(file_name)
    return exists,not_exists


def download_to_file(bucket_name:str ,work_path:str,save_path:str,file_list:list):

    bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)
    for file_name in file_list:
        bucket.get_object_to_file(os.path.join(work_path,file_name), os.path.join(save_path,file_name))


def get_image_url(bucket_name:str,image_path:str):
    # yourBucketName填写存储空间名称。
    bucket = oss2.Bucket(auth, cname, bucket_name, region=region,is_cname=True)

    # 生成上传文件的签名URL，有效时间为100年。
    # 生成签名URL时，OSS默认会对Object完整路径中的正斜线（/）进行转义，从而导致生成的签名URL无法直接使用。
    # 设置slash_safe为True，OSS不会对Object完整路径中的正斜线（/）进行转义，此时生成的签名URL可以直接使用。
    url = bucket.sign_url('GET', image_path, 3153600000, slash_safe=True)
    return url  

