# -*- coding: utf-8 -*-
"""
MinIO Object Storage Client - 优化版
支持异步操作、批量处理、完善的错误处理和文件元数据管理
"""

import os
import io
import logging
import asyncio
import hashlib
from typing import Union, List, Optional, Dict, Any, BinaryIO, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class FileMetadata:
    """文件元数据类"""
    original_filename: str
    file_hash: str
    storage_key: str
    bucket: str
    size: int
    content_type: str
    upload_time: str
    user_name: Optional[str] = None
    taskid: Optional[str] = None
    session_id: Optional[str] = None
    aliases: Optional[List[str]] = None
    preview_url: Optional[str] = None  # 新增：预览URL字段
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # 确保 aliases 不为 None
        if result['aliases'] is None:
            result['aliases'] = [result['original_filename']]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileMetadata':
        # 向后兼容
        if 'aliases' not in data or data['aliases'] is None:
            data['aliases'] = [data['original_filename']]
        if 'preview_url' not in data:
            data['preview_url'] = None
        return cls(**data)
    
    def add_alias(self, filename: str) -> bool:
        """添加文件名别名，返回是否为新别名"""
        if self.aliases is None:
            self.aliases = [self.original_filename]
        
        if filename not in self.aliases:
            self.aliases.append(filename)
            return True
        return False


@dataclass
class StorageConfig:
    """存储配置类"""
    endpoint: str
    access_key: str
    secret_key: str
    region: str = "us-east-1"
    secure: bool = True
    timeout: int = 60
    max_retries: int = 3
    max_concurrent_uploads: int = 5
    
    @classmethod
    def from_env(cls) -> 'StorageConfig':
        """从环境变量加载配置"""
        endpoint = os.getenv("MINIO_ENDPOINT")
        access_key = os.getenv("MINIO_ACCESS_KEY_ID")
        secret_key = os.getenv("MINIO_ACCESS_KEY_SECRET")
        
        if not all([endpoint, access_key, secret_key]):
            raise ValueError("Missing required MinIO environment variables")
        
        return cls(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            region=os.getenv("MINIO_REGION", "us-east-1"),
            secure=os.getenv("MINIO_SECURE", "true").lower() == "true",
            timeout=int(os.getenv("MINIO_TIMEOUT", "60")),
            max_retries=int(os.getenv("MINIO_MAX_RETRIES", "3")),
            max_concurrent_uploads=int(os.getenv("MINIO_MAX_CONCURRENT", "5"))
        )


class StorageError(Exception):
    """存储操作异常基类"""
    pass


class FileNotFoundError(StorageError):
    """文件不存在异常"""
    pass


class BucketNotFoundError(StorageError):
    """存储桶不存在异常"""
    pass


class NetworkError(StorageError):
    """网络连接异常"""
    pass


class AuthenticationError(StorageError):
    """认证失败异常"""
    pass


class QuotaExceededError(StorageError):
    """配额超限异常"""
    pass

class StorageClient:
    """优化的MinIO存储客户端"""
    
    def __init__(self, config: Optional[StorageConfig] = None):
        """初始化存储客户端"""
        self.config = config or StorageConfig.from_env()
        self._client = self._create_client()
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_uploads)
        
    def _create_client(self) -> boto3.client:
        """创建S3客户端"""
        try:
            boto_config = Config(
                retries={'max_attempts': self.config.max_retries},
                read_timeout=self.config.timeout,
                connect_timeout=self.config.timeout,
                region_name=self.config.region,
                max_pool_connections=self.config.max_concurrent_uploads * 2
            )
            
            return boto3.client(
                's3',
                endpoint_url=self.config.endpoint,
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                region_name=self.config.region,
                config=boto_config
            )
        except NoCredentialsError as e:
            raise AuthenticationError(f"Invalid credentials: {e}")
        except Exception as e:
            raise StorageError(f"Failed to create storage client: {e}")
    
    def calculate_file_hash(self, file_content: bytes) -> str:
        """计算文件的SHA256哈希值"""
        return hashlib.sha256(file_content).hexdigest()
    
    def generate_storage_key(self, file_hash: str, filename: str, category: str = "uploads") -> str:
        """生成存储键名"""
        file_ext = os.path.splitext(filename)[1].lower()
        date_path = datetime.now().strftime("%Y/%m/%d")
        return f"{category}/{date_path}/{file_hash[:2]}/{file_hash}{file_ext}"
    
    def get_content_type(self, filename: str) -> str:
        """根据文件扩展名确定MIME类型"""
        ext = os.path.splitext(filename)[1].lower()
        content_type_map = {
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.ppt': 'application/vnd.ms-powerpoint',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.xls': 'application/vnd.ms-excel',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.csv': 'text/csv',
            '.txt': 'text/plain',
            '.json': 'application/json',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp',
            '.svg': 'image/svg+xml'
        }
        return content_type_map.get(ext, 'application/octet-stream')
    
    # =============== 同步方法 ===============
    
    def put_object(
        self,
        bucket: str,
        key: str,
        data: Union[bytes, BinaryIO, str],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        content_disposition: Optional[str] = None,
        cache_control: Optional[str] = None,
    ) -> Dict[str, Any]:
        """上传对象到存储桶，支持 Content-Type / Content-Disposition / Cache-Control"""
        try:
            # 统一成类文件对象
            if isinstance(data, str):
                data = data.encode('utf-8')
            if isinstance(data, (bytes, bytearray)):
                data = io.BytesIO(data)

            extra_args: Dict[str, Any] = {}
            if content_type:
                extra_args['ContentType'] = content_type
            if content_disposition:
                extra_args['ContentDisposition'] = content_disposition
            if cache_control:
                extra_args['CacheControl'] = cache_control
            if metadata:
                extra_args['Metadata'] = metadata

            self._client.upload_fileobj(
                Fileobj=data,
                Bucket=bucket,
                Key=key,
                ExtraArgs=extra_args if extra_args else None
            )

            logger.info(f"[put_object] uploaded: {bucket}/{key} ct={content_type} cd={content_disposition}")
            return {
                'bucket': bucket,
                'key': key,
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
            }

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchBucket':
                raise BucketNotFoundError(f"Bucket '{bucket}' not found")
            raise StorageError(f"Failed to upload object {bucket}/{key}: {e}") from e
        except Exception as e:
            raise StorageError(f"Failed to upload object {bucket}/{key}: {e}") from e

    
    def get_object(self, bucket: str, key: str) -> bytes:
        """获取对象内容"""
        try:
            response = self._client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise FileNotFoundError(f"Object '{key}' not found in bucket '{bucket}'")
            elif error_code == 'NoSuchBucket':
                raise BucketNotFoundError(f"Bucket '{bucket}' not found")
            raise StorageError(f"Failed to get object {bucket}/{key}: {e}")
    
    def object_exists(self, bucket: str, key: str) -> bool:
        """检查对象是否存在"""
        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code in ['NoSuchKey', '404']:
                return False
            elif error_code == 'NoSuchBucket':
                raise BucketNotFoundError(f"Bucket '{bucket}' not found")
            raise StorageError(f"Failed to check object existence {bucket}/{key}: {e}")
    
    def delete_object(self, bucket: str, key: str) -> Dict[str, Any]:
        """删除单个对象"""
        try:
            self._client.delete_object(Bucket=bucket, Key=key)
            logger.info(f"Successfully deleted object: {bucket}/{key}")
            return {
                'bucket': bucket,
                'key': key,
                'status': 'deleted',
                'timestamp': datetime.now().isoformat()
            }
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                raise BucketNotFoundError(f"Bucket '{bucket}' not found")
            raise StorageError(f"Failed to delete object {bucket}/{key}: {e}")
    
    def list_objects(self, bucket: str, prefix: str = "", max_keys: int = 1000) -> Dict[str, Any]:
        """列出存储桶中的对象"""
        try:
            params = {'Bucket': bucket, 'MaxKeys': max_keys}
            if prefix:
                params['Prefix'] = prefix
            
            response = self._client.list_objects_v2(**params)
            
            objects = []
            for obj in response.get('Contents', []):
                objects.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'etag': obj['ETag'].strip('"')
                })
            
            objects.sort(key=lambda x: x['last_modified'], reverse=True)
            
            return {
                'objects': objects,
                'count': len(objects),
                'is_truncated': response.get('IsTruncated', False),
                'timestamp': datetime.now().isoformat()
            }
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                raise BucketNotFoundError(f"Bucket '{bucket}' not found")
            raise StorageError(f"Failed to list objects in {bucket}: {e}")
    
    def generate_presigned_url(
        self, bucket: str, key: str, expires_in: int = 3600,
        response_content_type: Optional[str] = None,
        response_content_disposition: Optional[str] = None,
    ) -> str:
        params = {'Bucket': bucket, 'Key': key}
        if response_content_type:
            params['ResponseContentType'] = response_content_type
        if response_content_disposition:
            params['ResponseContentDisposition'] = response_content_disposition
        return self._client.generate_presigned_url('get_object', Params=params, ExpiresIn=expires_in)

    
    def get_public_url(self, bucket: str, key: str) -> str:
        """获取公共访问URL"""
        endpoint = self.config.endpoint.replace('http://', '').replace('https://', '')
        protocol = 'http' if self.config.secure else 'http'
        return f"{protocol}://{endpoint}/{bucket}/{key}"
    
    # =============== 异步方法 ===============
    
    async def aput_object(
        self, bucket: str, key: str, data: Union[bytes, BinaryIO, str],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        content_disposition: Optional[str] = None,
        cache_control: Optional[str] = None
    ) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.put_object, bucket, key, data, content_type, metadata, content_disposition, cache_control
        )

    
    async def aget_object(self, bucket: str, key: str) -> bytes:
        """异步获取对象"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.get_object, bucket, key)
    
    async def aobject_exists(self, bucket: str, key: str) -> bool:
        """异步检查对象是否存在"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.object_exists, bucket, key)
    
    async def adelete_object(self, bucket: str, key: str) -> Dict[str, Any]:
        """异步删除对象"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.delete_object, bucket, key)
    
    async def alist_objects(self, bucket: str, prefix: str = "", max_keys: int = 1000) -> Dict[str, Any]:
        """异步列出对象"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.list_objects, bucket, prefix, max_keys)
    
    # =============== 批量操作方法 ===============
    
    async def batch_upload_files(self, files_data: List[Dict[str, Any]], 
                                progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """批量上传文件"""
        results = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_uploads)
        
        async def upload_single_file(file_data: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    result = await self.aput_object(
                        bucket=file_data['bucket'],
                        key=file_data['key'],
                        data=file_data['content'],
                        content_type=file_data.get('content_type'),
                        metadata=file_data.get('metadata')
                    )
                    result.update({
                        'original_filename': file_data.get('original_filename'),
                        'file_hash': file_data.get('file_hash')
                    })
                    
                    if progress_callback:
                        await progress_callback(result)
                    
                    return result
                except Exception as e:
                    error_result = {
                        'original_filename': file_data.get('original_filename'),
                        'error': str(e),
                        'status': 'failed'
                    }
                    
                    if progress_callback:
                        await progress_callback(error_result)
                    
                    return error_result
        
        tasks = [upload_single_file(file_data) for file_data in files_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    'error': str(result),
                    'status': 'failed'
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def batch_delete_objects(self, bucket: str, keys: List[str]) -> Dict[str, Any]:
        """批量删除对象"""
        if not keys:
            return {'deleted': [], 'errors': [], 'total': 0}
        
        try:
            delete_objects = [{'Key': key} for key in keys]
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self._executor,
                lambda: self._client.delete_objects(
                    Bucket=bucket,
                    Delete={'Objects': delete_objects}
                )
            )
            
            deleted = response.get('Deleted', [])
            errors = response.get('Errors', [])
            
            logger.info(f"Batch delete completed: {len(deleted)}/{len(keys)} successful")
            return {
                'deleted': [obj['Key'] for obj in deleted],
                'errors': [{'key': err['Key'], 'code': err['Code'], 'message': err['Message']} 
                          for err in errors],
                'total': len(keys),
                'timestamp': datetime.now().isoformat()
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                raise BucketNotFoundError(f"Bucket '{bucket}' not found")
            raise StorageError(f"Failed to delete objects from {bucket}: {e}")
    
    # =============== 文件元数据管理 ===============
    
    def create_file_metadata(self, filename: str, file_content: bytes, bucket: str,
                           user_name: str = None, taskid: str = None, session_id: str = None) -> FileMetadata:
        """创建文件元数据"""
        file_hash = self.calculate_file_hash(file_content)
        storage_key = self.generate_storage_key(file_hash, filename)
        content_type = self.get_content_type(filename)
        
        return FileMetadata(
            original_filename=filename,
            file_hash=file_hash,
            storage_key=storage_key,
            bucket=bucket,
            size=len(file_content),
            content_type=content_type,
            upload_time=datetime.now().isoformat(),
            user_name=user_name,
            taskid=taskid,
            session_id=session_id,
            aliases=[filename],
        )
    
    async def save_file_metadata(self, metadata: FileMetadata, metadata_bucket: str = None) -> bool:
        """保存文件元数据到存储"""
        try:
            bucket = metadata_bucket or metadata.bucket
            metadata_key = f"metadata/{metadata.file_hash[:2]}/{metadata.file_hash}.meta.json"
            
            metadata_json = json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2)
            
            await self.aput_object(
                bucket=bucket,
                key=metadata_key,
                data=metadata_json.encode('utf-8'),
                content_type='application/json'
            )
            
            logger.info(f"Saved metadata for file: {metadata.original_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save metadata for {metadata.original_filename}: {e}")
            return False
    
    async def get_file_metadata(self, file_hash: str, metadata_bucket: str) -> Optional[FileMetadata]:
        """获取文件元数据"""
        try:
            metadata_key = f"metadata/{file_hash[:2]}/{file_hash}.meta.json"
            metadata_json = await self.aget_object(metadata_bucket, metadata_key)
            metadata_dict = json.loads(metadata_json.decode('utf-8'))
            return FileMetadata.from_dict(metadata_dict)
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Failed to get metadata for hash {file_hash}: {e}")
            return None
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


# =============== 全局存储客户端管理 ===============

_storage_client: Optional[StorageClient] = None


def get_storage_client() -> StorageClient:
    """获取全局存储客户端实例"""
    global _storage_client
    if _storage_client is None:
        _storage_client = StorageClient()
    return _storage_client


def init_storage_client(config: StorageConfig) -> StorageClient:
    """初始化全局存储客户端"""
    global _storage_client
    _storage_client = StorageClient(config)
    return _storage_client


# =============== 兼容性接口（临时使用） ===============

async def oss_upload(bucket_name: str, save_path: str, file_obj) -> Dict[str, Any]:
    """兼容旧接口的上传方法（增加 Content-Type 与 inline 显示）"""
    client = get_storage_client()
    # 用保存路径推断类型：.png/.jpg/.webp...
    content_type = client.get_content_type(save_path)
    # 上传
    await client.aput_object(
        bucket=bucket_name,
        key=save_path,
        data=file_obj,
        content_type=content_type,
        metadata={"ContentDisposition": "inline"}  # 也可以放到 ExtraArgs 内部，这里复用 metadata 语义
    )
    return {"status": 200, "msg": "Upload succeeded", "path": save_path}

def oss_list(bucket_name: str, path: str) -> List[str]:
    """兼容旧接口的列表方法"""
    client = get_storage_client()
    result = client.list_objects(bucket_name, path)
    return [os.path.basename(obj['key']) for obj in result['objects']]


def oss_del_list(bucket_name: str, file_path: str, file_list: List[str]) -> tuple:
    """兼容旧接口的删除方法"""
    client = get_storage_client()
    exists = []
    not_exists = []
    
    for file_name in file_list:
        key = os.path.join(file_path, file_name)
        try:
            if client.object_exists(bucket_name, key):
                client.delete_object(bucket_name, key)
                exists.append(file_name)
            else:
                not_exists.append(file_name)
        except Exception:
            not_exists.append(file_name)
    
    return exists, not_exists


def get_image_url(bucket_name: str, image_path: str) -> str:
    client = get_storage_client()
    ctype = client.get_content_type(image_path)
    # return client.generate_presigned_url(
    #     bucket_name, image_path, expires_in=604800,
    #     response_content_type=ctype,
    #     response_content_disposition="inline"
    return client.get_public_url(bucket_name, image_path)
# =============== 兼容性接口（继续补充） ===============

def download_to_file(bucket_name: str, work_path: str, save_path: str, file_list: list) -> None:
    """
    旧接口等价实现（同步）：
    从 MinIO 下载多个对象到本地目录：
    - 对象键：os.path.join(work_path, file_name)
    - 本地路径：os.path.join(save_path, file_name)
    """
    client = get_storage_client()
    Path(save_path).mkdir(parents=True, exist_ok=True)

    for file_name in file_list:
        key = os.path.join(work_path, file_name)
        try:
            content = client.get_object(bucket_name, key)
            local_path = os.path.join(save_path, file_name)
            Path(os.path.dirname(local_path) or ".").mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(content)
            logger.info(f"[download_to_file] saved: {bucket_name}/{key} -> {local_path}")
        except Exception as e:
            logger.error(f"[download_to_file] failed: {bucket_name}/{key} -> {e}")


async def adownload_to_file(bucket_name: str, work_path: str, save_path: str, file_list: list) -> None:
    """
    旧接口的异步版本（如将来有需要）：
    """
    client = get_storage_client()
    Path(save_path).mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_event_loop()

    async def _download_one(file_name: str):
        key = os.path.join(work_path, file_name)
        try:
            content = await client.aget_object(bucket_name, key)
            local_path = os.path.join(save_path, file_name)
            Path(os.path.dirname(local_path) or ".").mkdir(parents=True, exist_ok=True)
            # 文件写入保持同步即可
            def _write_bytes():
                with open(local_path, "wb") as f:
                    f.write(content)
            await loop.run_in_executor(None, _write_bytes)
            logger.info(f"[adownload_to_file] saved: {bucket_name}/{key} -> {local_path}")
        except Exception as e:
            logger.error(f"[adownload_to_file] failed: {bucket_name}/{key} -> {e}")

    await asyncio.gather(*[_download_one(name) for name in file_list])


# =============== 桶存在性检查（同事代码会用到） ===============

class BucketNotReachableError(StorageError):
    """HeadBucket 失败（不可达/权限）"""
    pass


def bucket_exists(bucket_name: str) -> bool:
    """
    同步：检查桶是否存在
    """
    client = get_storage_client()
    try:
        client._client.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchBucket", "NotFound"):
            return False
        # 其他错误：网络/权限等，视为不可达，但区别于“不存在”
        raise BucketNotReachableError(f"head_bucket failed: {e}") from e


async def abucket_exists(bucket_name: str) -> bool:
    """
    异步：检查桶是否存在（同事的 /uploadFile 在用）
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, bucket_exists, bucket_name)
