# -*- coding: utf-8 -*-
import os
import sys
import asyncio
import tempfile
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from pydantic import PrivateAttr
from alpha.actions import Action
from alpha.logs import logger
from src.llm_utils import SeLLM, load_config
from src.storage_utils import adownload_to_file, get_storage_client, oss_upload, get_image_url
from .utils import CodeRetriever, extract_all_python_codes

class Coding(Action):
    name: str = "AI4Simulations"
    desc: str = "基于AI算法的方程正向求解以及逆向计算"
    _code_retriever: CodeRetriever = PrivateAttr(default=None)

    def _get_code_retriever(self) -> CodeRetriever:
        if self._code_retriever is None:
            self._code_retriever = CodeRetriever()
        return self._code_retriever

    async def _stream_llm_response(self, llm, messages, websocket=None) -> str:
        collected_chunks = []
        import openai
        try:
            stream_res = await llm.acompletion_text(messages, temperature=0.7, timeout=10)
            async for chunk in stream_res:
                if chunk.choices and chunk.choices[0].delta:
                    chunk_msg = chunk.choices[0].delta.content or ""
                    if chunk_msg:
                        collected_chunks.append(chunk_msg)
                        if websocket: await websocket.send_text(chunk_msg)
        except Exception as e:
            logger.exception(f"[LLM_Stream-LOG] LLM Stream 异常: {str(e)}")
            raise
        return "".join(collected_chunks)

    async def run(self, instruction: str, *args):
        websocket = args[0]
        user_name, taskid, file_metadata = args[1], args[2], args[3]
        config = load_config("config/config.yaml")
        llm = SeLLM(base_url=config["base_url_1"], api_key=config["api_key"])
        await websocket.send_text("%Line Break%")
        await websocket.send_text(json.dumps({"type": "info", "message": "💻 开始执行主控脚本... "}))
        return ""

class DiT4ScienceAction(Action):
    name: str = "DiT4ScienceInference"
    desc: str = "运行 DiT4Science 推理，支持 MinIO 数据动态加载"

    async def run(self, instruction: str, *args):
        websocket = args[0]
        user_name, taskid = args[1], args[2]
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            client = get_storage_client()
            bucket = os.getenv("MINIO_BUCKET", "alpha")
            data_prefix = os.getenv("DIT_DATA_PREFIX", "AI4Simulations/DiT4Science/data/")
            objects = client.list_objects(bucket, prefix=data_prefix)
            file_names = [os.path.relpath(obj['key'], data_prefix) for obj in objects['objects'] if not obj['key'].endswith('/')]
            local_data_path = os.path.join(tmp_dir, "test_data")
            os.makedirs(local_data_path, exist_ok=True)
            await adownload_to_file(bucket, data_prefix, local_data_path, file_names)
            
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            dit_project_path = os.path.join(project_root, "DEV-DiT4Science")
            weights_dir = os.path.join(dit_project_path, "weights")
            os.makedirs(weights_dir, exist_ok=True)
            weights_to_check = ["6300000.pt", "cond_all_epoch=152-step=167840.ckpt", "res_all_epoch=109-step=120669.ckpt"]
            weights_prefix = os.getenv("DIT_WEIGHTS_PREFIX", "AI4Simulations/DiT4Science/weights/")
            for w_file in weights_to_check:
                if not os.path.exists(os.path.join(weights_dir, w_file)):
                    await adownload_to_file(bucket, weights_prefix, weights_dir, [w_file])
            
            python_executable = os.getenv("DIT_PYTHON_PATH", sys.executable)
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{dit_project_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
            env["DIT_DATA_PATH"] = local_data_path
            proc = await asyncio.create_subprocess_exec(python_executable, "-u", os.path.join(dit_project_path, "test_inference.py"),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=dit_project_path, env=env)
            
            async def _pipe_logs(reader):
                while True:
                    line = await reader.readline()
                    if not line: break
                    await websocket.send_text(line.decode())
            await asyncio.gather(_pipe_logs(proc.stdout), _pipe_logs(proc.stderr))
            await proc.wait()
        return "DiT4Science 推理已完成。"

class ACDM4ScienceAction(Action):
    name: str = "ACDM4ScienceInference"
    desc: str = "运行 ACDM4Science 推理，支持模型权重和数据的动态加载"

    async def run(self, instruction: str, *args):
        websocket = args[0]
        user_name, taskid = args[1], args[2]
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            client = get_storage_client()
            bucket = os.getenv("MINIO_BUCKET", "alpha")
            data_prefix = os.getenv("ACDM_DATA_PREFIX", "AI4Simulations/ACDM4Science/data/")
            try:
                objects = client.list_objects(bucket, prefix=data_prefix)
                file_names = [os.path.relpath(obj['key'], data_prefix) for obj in objects['objects'] if not obj['key'].endswith('/')]
            except: file_names = []
            
            local_data_path = os.path.join(tmp_dir, "test_data")
            os.makedirs(local_data_path, exist_ok=True)
            if file_names: await adownload_to_file(bucket, data_prefix, local_data_path, file_names)
            
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            acdm_project_path = os.path.join(project_root, "DEV-ACDM4Science")
            weights_dir = os.path.join(acdm_project_path, "weights")
            os.makedirs(weights_dir, exist_ok=True)
            weight_file = "Model_E1600.pth"
            weights_prefix = os.getenv("ACDM_WEIGHTS_PREFIX", "AI4Simulations/ACDM4Science/weights/")
            local_w_path = os.path.join(weights_dir, weight_file)
            if not os.path.exists(local_w_path):
                await adownload_to_file(bucket, weights_prefix, weights_dir, [weight_file])
            
            python_executable = os.getenv("ACDM_PYTHON_PATH", sys.executable)
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{acdm_project_path}{os.pathsep}{os.path.join(acdm_project_path, 'ai4flow')}{os.pathsep}{os.path.join(acdm_project_path, 'ai4flow', 'acdm')}{os.pathsep}{env.get('PYTHONPATH', '')}"
            
            inference_script = f"""
import os, sys, torch, numpy as np
from torch.utils.data import DataLoader, SequentialSampler
sys.path.append(r"{os.path.join(acdm_project_path, 'ai4flow')}")
sys.path.append(r"{os.path.join(acdm_project_path, 'ai4flow', 'acdm')}")
try:
    from turbpred.v2d_dataset import V2dDataset
    from turbpred.data_transformations import Transforms
    from turbpred.model import PredictionModel
    from turbpred.params import DataParams, TrainingParams, LossParams, ModelParamsDecoder
except ImportError:
    from dataset.airfoil_acdm import airfoilDataset as V2dDataset
    from dataset.airfoil_acdm import Transforms
    from turbpred.model import PredictionModel
    from turbpred.params import DataParams, TrainingParams, LossParams, ModelParamsDecoder

device = "cuda:0" if torch.cuda.is_available() else "cpu"
test_dataset_path = r"{local_data_path}"
model_weight_path = r"{local_w_path}"
try:
    testSet = V2dDataset(name="V2D dataset", dataDir=test_dataset_path, sequenceLength=[3, 2], rey=1)
except Exception:
    from dataset.airfoil_acdm import airfoilDataset
    testSet = airfoilDataset(name="V2D dataset", dataDir=test_dataset_path, sequenceLength=[3, 2], rey=1, self_cond=True)
testSet.transform = Transforms()
testLoader = DataLoader(testSet, sampler=SequentialSampler(testSet), batch_size=len(testSet) if len(testSet)>0 else 1)
model = PredictionModel.load(model_weight_path, useGPU=torch.cuda.is_available())
model.to(device).eval()
with torch.no_grad():
    for s, sample in enumerate(testLoader, 0):
        data = sample["data"].to(device)
        simParameters = sample["simParameters"].to(device) if type(sample["simParameters"]) is not dict else None
        prediction, _, _ = model(data, simParameters)
        print(f"Sample {{s}} predicted, shape: {{prediction.shape}}")
"""
            temp_script_path = os.path.join(tmp_dir, "acdm_inference_run.py")
            with open(temp_script_path, "w", encoding="utf-8") as f: f.write(inference_script)
            proc = await asyncio.create_subprocess_exec(python_executable, "-u", temp_script_path,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=acdm_project_path, env=env)
            async def _pipe_logs(reader):
                while True:
                    line = await reader.readline()
                    if not line: break
                    await websocket.send_text(line.decode())
            await asyncio.gather(_pipe_logs(proc.stdout), _pipe_logs(proc.stderr))
            await proc.wait()
        return "ACDM4Science 推理已完成。"

class GNN4SInferenceAction(Action):
    name: str = "GNN4SInference"
    desc: str = "运行 GNN4Science 推理，支持多种科学计算场景的图神经网络模拟"

    async def run(self, instruction: str, *args):
        websocket = args[0]
        user_name, taskid = args[1], args[2]
        demo_type = "airfoil"
        if "cylinder" in instruction.lower(): demo_type = "cylinder_flow"
        elif "cardio" in instruction.lower(): demo_type = "cardiovascular"
        elif "flag" in instruction.lower(): demo_type = "flag_simple"
        elif "molecule" in instruction.lower() or "mol" in instruction.lower(): demo_type = "mol_gen_qm9"

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        gnn4s_path = os.path.join(project_root, "DEV-GNN4Science")
        
        # 1. 自动准备推理资产 (从 MinIO 同步权重和数据)
        client = get_storage_client()
        bucket = os.getenv("MINIO_BUCKET", "alpha")
        base_prefix = "AI4Simulations/GNN4Science/"
        
        await websocket.send_text(f"正在准备 {demo_type} 推理资产...\n")
        
        # 定义任务所需的关键文件列表
        task_assets = {
            "airfoil": {
                "ckpt": ["airfoil/infer/airfoil.pth"],
                "data": ["airfoil/raw/airfoil_test_name_list.txt", "airfoil/raw/airfoil_surface.h5", 
                         "airfoil/raw/airfoil_sec_z_0.10.h5", "airfoil/raw/deg_0_mach_1.3.h5"]
            },
            "cylinder_flow": {
                "ckpt": ["cylinder_flow/infer/cylinder_flow.pth"],
                "data": ["cylinder_flow/raw/cylinder_flow_test.h5", "cylinder_flow/raw/meta.json"]
            },
            "flag_simple": {
                "ckpt": ["flag_simple/infer/flag_simple.pth"],
                "data": ["flag_simple/raw/flag_simple_test.h5", "flag_simple/raw/meta.json"]
            },
            "cardiovascular": {
                "ckpt": ["cardiovascular/infer/cardiovascular.pth"],
                "data": ["cardiovascular/processed/dataset_info.json"] # 常规同步 processed 目录，此处仅示例核心
            }
        }

        current_task = task_assets.get(demo_type, {})
        
        # 同步权重
        ckpt_dir = os.path.join(gnn4s_path, "checkpoint")
        for ckpt_rel in current_task.get("ckpt", []):
            local_path = os.path.join(ckpt_dir, ckpt_rel)
            if not os.path.exists(local_path):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                await websocket.send_text(f"下载权重: {ckpt_rel}\n")
                await adownload_to_file(bucket, f"{base_prefix}checkpoint/", ckpt_dir, [ckpt_rel])
        
        # 同步数据
        data_dir = os.path.join(gnn4s_path, "dataset")
        for data_rel in current_task.get("data", []):
            local_path = os.path.join(data_dir, data_rel)
            # 强化逻辑：对于 .txt 或 .json 等轻量级元数据文件，强制同步确保列表是最新的
            is_metadata = data_rel.endswith(('.txt', '.json', '.yaml'))
            if is_metadata or not os.path.exists(local_path):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                msg = f"更新元数据: {data_rel}" if is_metadata else f"下载必要数据: {data_rel}"
                await websocket.send_text(f"{msg}\n")
                await adownload_to_file(bucket, f"{base_prefix}dataset/", data_dir, [data_rel])
        
        # 对于 Airfoil，需要拉取文件夹下的所有 parts 并确保使用分片列表
        if demo_type == "airfoil":
             await websocket.send_text("优化 Airfoil 数据加载: 切换至分片模式以节省显存...\n")
             # 1. 下载所有分片
             prefix = f"{base_prefix}dataset/airfoil/raw/deg_0_mach_1.3/"
             objects = client.list_objects(bucket, prefix=prefix)
             parts = [os.path.relpath(obj['key'], f"{base_prefix}dataset/") for obj in objects['objects'] if obj['key'].endswith('.h5')]
             if parts:
                 await adownload_to_file(bucket, f"{base_prefix}dataset/", data_dir, parts)
             
             # 2. 强制改写本地 name_list.txt，确保不加载那个 1.1GB 的大文件
             local_nl = os.path.join(gnn4s_path, "dataset", "airfoil", "raw", "airfoil_test_name_list.txt")
             correct_lines = [f"deg_0_mach_1.3/deg_0_mach_1.3_part_{i}" for i in range(26)]
             with open(local_nl, "w") as f:
                 f.write("\n".join(correct_lines))
             await websocket.send_text("Airfoil 分片列表已就绪。\n")
        
        # 对于心血管任务，通常是整个 processed 目录，这里特殊处理一下拉取全量 grph
        if demo_type == "cardiovascular":
            local_processed = os.path.join(data_dir, "cardiovascular", "processed")
            if not os.path.exists(local_processed) or len(os.listdir(local_processed)) < 5:
                await websocket.send_text("下载心血管图结构数据集...\n")
                prefix = f"{base_prefix}dataset/cardiovascular/processed/"
                objects = client.list_objects(bucket, prefix=prefix)
                grph_files = [os.path.relpath(obj['key'], f"{base_prefix}dataset/") for obj in objects['objects'] if obj['key'].endswith('.grph')]
                await adownload_to_file(bucket, f"{base_prefix}dataset/", data_dir, grph_files)

        # 2. 执行推理
        python_executable = os.getenv("GNN4S_PYTHON_PATH", sys.executable)
        env = os.environ.copy()
        # 强制使用 Linux 的路径分隔符 : (因为远程执行环境是 Linux)
        path_sep = ":" if python_executable.startswith("/") else os.pathsep
        env["PYTHONPATH"] = f"{gnn4s_path}{path_sep}{env.get('PYTHONPATH', '')}"
        script_path = os.path.join(gnn4s_path, "demo", f"{demo_type}_infer.py")
        
        # 增强鲁棒性：正式运行前确保可能的输出目录已存在
        output_parent = os.path.join(gnn4s_path, "dataset", demo_type.replace("_flow", ""))
        output_dir_standard = os.path.join(output_parent, "output")
        os.makedirs(output_dir_standard, exist_ok=True)
        
        await websocket.send_text(f"开始运行 [GNN4S] 推理脚本: {demo_type}\n")
        proc = await asyncio.create_subprocess_exec(python_executable, "-u", script_path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=gnn4s_path, env=env)
        
        # 实时同步日志，并捕获任何报错信息
        async def _pipe_logs(reader, stream_name):
            while True:
                line = await reader.readline()
                if not line: break
                text = line.decode(errors='replace')
                # 发送到前端
                await websocket.send_text(text)
                # 如果是 stderr 且包含 Error 关键字，进行记录
                if stream_name == "stderr" and "Error" in text:
                    logger.error(f"[GNN4S-{demo_type}] 子进程错误: {text.strip()}")

        await asyncio.gather(
            _pipe_logs(proc.stdout, "stdout"),
            _pipe_logs(proc.stderr, "stderr")
        )
        await proc.wait()

        # 3. 自动同步推理结果 (可视化图片/GIF 上传至 MinIO)
        await websocket.send_text("正在搜寻可视化结果并同步至云端...\n")
        
        # 搜索路径：优先尝试标准路径，再尝试通配搜索
        possible_output_dirs = [
            output_dir_standard,
            os.path.join(gnn4s_path, "dataset", demo_type, "output"),
            os.path.join(gnn4s_path, "output")
        ]
        
        found_any = False
        for d in possible_output_dirs:
            if not os.path.exists(d): continue
            
            result_files = []
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(('.png', '.jpg', '.gif')):
                        result_files.append(os.path.join(root, f))
            
            if result_files:
                await websocket.send_text(f"在目录 {d} 发现 {len(result_files)} 个可视化文件，准备同步...\n")
                for local_file in result_files:
                    filename = os.path.basename(local_file)
                    oss_path = f"AI4Simulations/outputs/GNN4Science/{taskid}/{filename}"
                    try:
                        with open(local_file, "rb") as f_obj:
                            await oss_upload(bucket, oss_path, f_obj)
                        
                        img_url = get_image_url(bucket, oss_path)
                        if img_url:
                            msg = {"type": "image", "url": img_url, "name": f"{demo_type} 结果: {filename}"}
                            await websocket.send_text(f"IMAGE_LINK:{json.dumps(msg)}\n")
                            found_any = True
                    except Exception as e:
                        await websocket.send_text(f"上传 {filename} 失败: {str(e)}\n")
                # 只要在一个目录下找到了文件，就不再搜寻其他备选目录（避免重复）
                if found_any: break
        
        if not found_any:
            await websocket.send_text(f"🛑 未能在预期目录中搜寻到可视化结果文件 (.jpg/.gif)。\n")
            await websocket.send_text(f"检查路径: {possible_output_dirs}\n")

        return f"GNN4S {demo_type} 推理流程处理完毕。"

class DLSurrogateInferenceAction(Action):
    name: str = "DLSurrogateInference"
    desc: str = "运行 DLSurrogate 推理，支持形状优化和代理模型求解"

    async def run(self, instruction: str, *args):
        websocket = args[0]
        user_name, taskid = args[1], args[2]
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        dls_path = os.path.join(project_root, "DEV-DLSurrogate", "dl-surrogates")
        
        # 1. 自动准备推理所需的“核心轻量资产”（权重+归一化参数），不下载全量数据集
        client = get_storage_client()
        bucket = os.getenv("MINIO_BUCKET", "alpha")
        dls_weights_prefix = os.getenv("DLS_WEIGHTS_PREFIX", "AI4Simulations/DLSurrogate/weights/")
        
        # 定义核心资产及其存放位置
        asset_map = {
            os.path.join(dls_path, "shape-opt", "levelset", "weights"): [
                "cond_all_epoch=152-step=167840.ckpt",
                "res_all_epoch=109-step=120669.ckpt"
            ],
            os.path.join(dls_path, "models", "dataset-ranged-400"): [
                "max_inputs.pickle",
                "max_targets.pickle"
            ],
            # DiT4Science 的主模型权重（通常很大，单独检查）
            os.path.join(project_root, "DEV-DLSurrogate", "results", "004-DiT-L-2", "checkpoints"): [
                "6300000.pt"
            ]
        }

        await websocket.send_text("正在检查推理核心资产（权重与归一化参数）...\n")
        for local_dir, files in asset_map.items():
            os.makedirs(local_dir, exist_ok=True)
            for f in files:
                if not os.path.exists(os.path.join(local_dir, f)):
                    await websocket.send_text(f"资产缺失，从 MinIO 获取: {f}\n")
                    try:
                        await adownload_to_file(bucket, dls_weights_prefix, local_dir, [f])
                    except Exception as e:
                        await websocket.send_text(f"资产 {f} 下载失败: {str(e)}\n")

        # 2. 准备执行环境
        main_script = os.path.join(dls_path, "shape-opt", "levelset", "main_Deep_DiT.py")
        python_executable = os.getenv("DLS_PYTHON_PATH", sys.executable)
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{dls_path}{os.pathsep}{os.path.join(dls_path, 'shape-opt')}{os.pathsep}{os.path.join(dls_path, 'train')}{os.pathsep}{env.get('PYTHONPATH', '')}"
        
        loader_content = f"""
import os, sys
dls_path = r"{dls_path}"
# 修复硬编码路径
sys.path.insert(0, dls_path)
sys.path.insert(0, os.path.join(dls_path, 'shape-opt'))
sys.path.insert(0, os.path.join(dls_path, 'shape-opt', 'levelset'))
os.environ["DL_SURROGATES_ROOT"] = dls_path
main_script_path = r"{main_script}"
if not os.path.exists(main_script_path):
    print(f"Error: Script not found: {{main_script_path}}")
    sys.exit(1)
with open(main_script_path, "r", encoding="utf-8") as f:
    code = f.read()
    # 动态修补 Mayuan 的绝对路径
    code = code.replace("/data/sda/mayuan/dl-surrogates-main", dls_path)
    code = code.replace("/data/sda/mayuan/results/004-DiT-L-2-channel_idx-1/checkpoints/6300000.pt", os.path.join(r"{project_root}", "DEV-DLSurrogate", "results", "004-DiT-L-2", "checkpoints", "6300000.pt"))
    exec(code, {{"__name__": "__main__", "__file__": main_script_path}})
"""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w', encoding='utf-8') as f:
            f.write(loader_content)
            temp_script = f.name
        try:
            await websocket.send_text("启动 DLSurrogate 推理...\n")
            proc = await asyncio.create_subprocess_exec(python_executable, "-u", temp_script,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=os.path.dirname(main_script), env=env)
            async def _pipe_logs(reader):
                while True:
                    line = await reader.readline()
                    if not line: break
                    await websocket.send_text(line.decode(errors='replace'))
            await asyncio.gather(_pipe_logs(proc.stdout), _pipe_logs(proc.stderr))
            await proc.wait()
        finally: os.remove(temp_script)
        return "DLSurrogate 推理执行完毕。"
