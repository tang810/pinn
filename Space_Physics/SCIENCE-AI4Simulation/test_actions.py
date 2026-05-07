# -*- coding: utf-8 -*-
import asyncio
import os
import sys

# 确保能导入 src.team_config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.team_config.actions import GNN4SInferenceAction, DLSurrogateInferenceAction

class MockWebSocket:
    async def send_text(self, text):
        print(f"[WS] {text}", end="")

async def test_gnn4s(target="airfoil"):
    print(f"\n=== Testing GNN4S ({target}) ===")
    action = GNN4SInferenceAction()
    ws = MockWebSocket()
    instruction = f"run GNN4S {target} demo"
    result = await action.run(instruction, ws, "test_user", "task_001")
    print(f"\nResult: {result}")

async def test_dlsurrogate():
    print("\n=== Testing DLSurrogate (Shape Opt) ===")
    action = DLSurrogateInferenceAction()
    ws = MockWebSocket()
    result = await action.run("run shape optimization using DLSurrogate", ws, "test_user", "task_002")
    print(f"\nResult: {result}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1].lower()
        if target == "all":
            async def run_all():
                for demo in ["airfoil", "cardiovascular", "cylinder_flow", "flag_simple"]:
                    await test_gnn4s(demo)
            asyncio.run(run_all())
        elif target == "full":
            async def run_full():
                for demo in ["airfoil", "cardiovascular", "cylinder_flow", "flag_simple"]:
                    await test_gnn4s(demo)
                await test_dlsurrogate()
            asyncio.run(run_full())
        # 兼容性处理：如果包含 dls/surrogate，跑 DLSurrogate
        elif "dls" in target or "surrogate" in target:
            asyncio.run(test_dlsurrogate())
        # 否则尝试作为 GNN 任务处理 (支持 gnn 关键字或直接的任务名)
        else:
            demo_type = "airfoil"
            if "cardio" in target: demo_type = "cardiovascular"
            elif "cylinder" in target or "flow" in target: demo_type = "cylinder_flow"
            elif "flag" in target or "simple" in target: demo_type = "flag_simple"
            asyncio.run(test_gnn4s(demo_type))
    else:
        print("Usage: python test_actions.py [gnn|dls]")
