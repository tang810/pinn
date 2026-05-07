# -*- coding: utf-8 -*-
import os
import re
import sys
import asyncio
import tempfile
import time
import uuid
import pathlib
import json
from typing import List, Dict, Tuple, Optional, Callable, Any
import numpy as np
from sentence_transformers import CrossEncoder
from alpha.logs import logger

try:
    import resource  # POSIX only
except ImportError:
    resource = None

# 修改正则，提取所有 python 代码块
CODE_BLOCK_PATTERN = re.compile(
    r"```python(.*?)```",
    re.DOTALL | re.IGNORECASE
)

def extract_all_python_codes(text: str) -> list:
    """从文本中提取所有 ```python ... ``` 代码块"""
    return [code.strip() for code in CODE_BLOCK_PATTERN.findall(text)]

def _tc_snapshot(dirpath: str) -> Dict[str, float]:
    p = pathlib.Path(dirpath)
    return {str(f): f.stat().st_mtime for f in p.rglob("*") if f.is_file()}

def _tc_delta_files(before: Dict[str, float], after: Dict[str, float]) -> List[str]:
    return [p for p, m in after.items() if p not in before or after[p] != before[p]]

async def run_code(
    code: str,
    user_name: str,
    *,
    websocket=None,
    on_stdout: Optional[Callable[[str], None]] = None,
    on_stderr: Optional[Callable[[str], None]] = None,
    timeout: Optional[float] = None,
    mem_limit_mb: Optional[int] = None,
    cpu_time_sec: Optional[int] = None,
    workdir: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> Tuple[str, str, List[str]]:
    base_tmp = workdir or os.path.join(tempfile.gettempdir(), "ai4pde_exec", user_name)
    os.makedirs(base_tmp, exist_ok=True)
    script_path = os.path.join(base_tmp, f"{int(time.time())}_{uuid.uuid4().hex}.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)

    before = _tc_snapshot(base_tmp)

    preexec_fn = None
    if resource is not None and os.name == "posix":
        def _set_limits():
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        preexec_fn = _set_limits

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-u", script_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=base_tmp, env=env,
        preexec_fn=preexec_fn if os.name == "posix" else None,
    )

    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []

    async def _pump(reader, is_err=False):
        while True:
            line = await reader.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace")
            (stderr_chunks if is_err else stdout_chunks).append(text)
            if is_err:
                if on_stderr: on_stderr(text)
                if websocket: await websocket.send_text(text)
            else:
                if on_stdout: on_stdout(text)
                if websocket: await websocket.send_text(text)

    pump_out = asyncio.create_task(_pump(proc.stdout, is_err=False))
    pump_err = asyncio.create_task(_pump(proc.stderr, is_err=True))

    await asyncio.gather(pump_out, pump_err)
    await proc.wait()

    after = _tc_snapshot(base_tmp)
    new_files = _tc_delta_files(before, after)

    return "".join(stdout_chunks), "".join(stderr_chunks), new_files

class CodeRetriever:
    def __init__(self, 
                 json_file_path: str = "./src/Knowledgebase/registry/equations.index.json",
                 reranker_model_path: str = "/mnt/sdb/bge/bge-reranker-large",
                 score_threshold: float = 0.75):
        self.json_file_path = json_file_path
        self.reranker_model_path = reranker_model_path
        self.score_threshold = score_threshold
        self.projects: List[Dict[str, Any]] = []
        self.path_desc_cache: Dict[str, str] = {}
        self._project_desc_indexed: Dict[int, bool] = {}
        self._load_projects()
        try:
            self.reranker = CrossEncoder(self.reranker_model_path)
            logger.info(f"[CodeRetriever] 成功加载 reranker: {self.reranker_model_path}")
        except Exception as e:
            logger.exception(f"[CodeRetriever] reranker 加载失败: {str(e)}")
            self.reranker = None

    def _load_projects(self) -> None:
        if not os.path.exists(self.json_file_path):
            logger.error(f"[CodeRetriever] JSON 文件不存在: {self.json_file_path}")
            return
        try:
            with open(self.json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._json_root = data
            projects = []
            for domain in data.get("modules", []):
                domain_name = domain.get("name", "")
                for proj in domain.get("submodules", []):
                    projects.append({
                        "domain": domain_name,
                        "name": proj.get("name", ""),
                        "path": proj.get("path", ""),
                        "description": proj.get("description", ""),
                    })
            self.projects = projects
            logger.info(f"[CodeRetriever] 已加载项目数量: {len(self.projects)}")
        except Exception as e:
            logger.exception(f"[CodeRetriever] 加载 JSON 出错: {e}")

    def find_matching_project(self, query: str):
        if not self.projects or self.reranker is None:
            return None, 0.0, None
        texts = [f"{p.get('domain','')} | {p.get('name','')} | {p.get('description','')}" for p in self.projects]
        pairs = [[str(query), t[:600]] for t in texts]
        try:
            scores = self.reranker.predict(pairs)
            best_idx = int(np.argmax(scores))
            return self.projects[best_idx], float(scores[best_idx]), best_idx
        except Exception as e:
            logger.exception(f"[find_matching_project] reranker 评分异常: {e}")
            return None, 0.0, None

    def _find_project_node_by_path(self, project_path: str):
        root = getattr(self, "_json_root", None)
        if not root: return None
        for domain in root.get("modules", []):
            for proj in domain.get("submodules", []) or []:
                if proj.get("path") == project_path: return proj
        return None

    def get_project_item(self, project_idx: int, item_type: str) -> str | None:
        if project_idx is None or project_idx < 0 or project_idx >= len(self.projects):
            return None
        proj = self.projects[project_idx]
        proj_path = proj["path"]
        found_node = self._find_project_node_by_path(proj_path)
        if not found_node: return None
        if item_type in {"src", "results", "data", "model"}:
            submods = { (sm.get("name") or "").lower(): sm for sm in (found_node.get("submodules") or []) }
            return f"{proj_path}/{item_type}" if item_type in submods else None
        if item_type == "readme":
            for f in (found_node.get("files") or []):
                if (f.get("name") or "").lower().endswith(".md"): return f.get("path")
            return None
        if item_type == "main":
            for f in (found_node.get("files") or []):
                name = (f.get("name") or "").lower()
                if name.endswith("_main.py") or name.endswith("main.py"): return f.get("path")
            return None
        return None

    def _index_project_descriptions(self, project_idx: int) -> None:
        if project_idx is None or project_idx < 0 or project_idx >= len(self.projects) or self._project_desc_indexed.get(project_idx):
            return
        proj = self.projects[project_idx]
        target_path = proj["path"]
        found_node = self._find_project_node_by_path(target_path)
        if not found_node:
            self._project_desc_indexed[project_idx] = True
            return
        def ingest(node):
            p, d = node.get("path"), node.get("description", "")
            if p: self.path_desc_cache[p] = d
            for f in (node.get("files") or []):
                if f.get("path"): self.path_desc_cache[f.get("path")] = f.get("description", "")
            for sm in (node.get("submodules") or []): ingest(sm)
        ingest(found_node)
        self._project_desc_indexed[project_idx] = True

    def get_item_description(self, project_idx: int, file_path: str) -> str:
        if project_idx is None or project_idx < 0 or project_idx >= len(self.projects): return ""
        self._index_project_descriptions(project_idx)
        return self.path_desc_cache.get(file_path, "")
