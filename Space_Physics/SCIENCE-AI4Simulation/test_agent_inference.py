import asyncio
import os
import sys

# 将 src 目录添加到路径，以便导入 team_config 和 storage_utils
sys.path.append(os.path.join(os.getcwd(), "src"))

from dotenv import load_dotenv
load_dotenv()

from src.team_config import DiT4ScienceAction

class MockWebSocket:
    async def send_text(self, text):
        print(f"[Log] {text}", end="")

async def test_inference():
    action = DiT4ScienceAction()
    ws = MockWebSocket()
    
    print("开始测试 DiT4Science 推理 Action...")
    result = await action.run(
        "执行推理测试",
        ws,           # websocket
        "test_user",  # user_name
        "task_001"    # taskid
    )
    print(f"\n最终结果: {result}")

if __name__ == "__main__":
    asyncio.run(test_inference())
