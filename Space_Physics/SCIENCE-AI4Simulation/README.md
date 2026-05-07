<!--
text: "SCIENCE-AI4Simulation：AI4Simulation 平台后端工程"
area: "面向多案例、多角色协作与仿真任务编排的 AI4Simulation 平台服务端项目。"
tags: ["平台工程", "多案例编排", "FastAPI", "AI Agent"]
search: ["AI4Simulation", "platform backend", "FastAPI"]
-->

# SCIENCE-AI4Simulation：平台后端工程（Platform Project）

## 项目定位
本项目是 **AI4Simulation 平台后端工程**，用于统一承载多类科学仿真场景的任务编排、角色协作、文件上传与服务接口。

它不是单一案例仓库，而是平台级聚合项目。

## 平台规范化说明
本仓库按“平台工程规范”组织，采用以下约定：

- `alpha/`：平台核心框架（角色、动作、工具、记忆、策略等）
- `src/`：平台知识库与团队配置
- `config/`：平台运行配置
- `upload/`：任务上传文件目录
- `data/`：标准数据目录（平台规范补齐）
- `model/`：标准模型目录（平台规范补齐）
- `results/`：标准结果目录（平台规范补齐）
- `main.py`：服务主入口（FastAPI + WebSocket）
- `file.json`：结构描述文件（机器可读）

## 快速启动
### 1. 安装依赖
可使用现有依赖文件：

```bash
pip install -r pip_requirements.txt
```

### 2. 设置环境变量
项目通过 `.env` 读取运行参数（如 `PORT`）。

### 3. 启动服务
```bash
python main.py
```

或使用：
```bash
bash start.sh
```

## 主要接口（以代码为准）
- `GET /`：服务状态检查
- `GET /roles`：获取平台角色
- `POST /uploadFile`：上传任务文件
- `POST /files`：列出任务文件
- `WS /start`：启动任务协作会话

## 注意事项
- 本项目为平台工程，面向多案例协作，不建议按“单案例训练项目”方式改造。
- 若要做单案例规范化，请从平台中抽取具体案例到独立目录，再应用 `quick/train` 入口规范。
