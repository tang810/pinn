# -*- coding: utf-8 -*-
import os
import sys
import asyncio
import datetime
from dotenv import load_dotenv
from alpha.team import Team
from alpha.logs import logger
from alpha.actions import UserRequirement
from src.llm_utils import SeLLM, load_config
from .roles import XIMU_AI4Simulations

load_dotenv()
today = datetime.datetime.now().strftime("%Y%m%d")
logger.configure(handlers=[
    {"sink": sys.stdout, "level": "INFO"},
    {"sink": "logs/{time:YYYYMMDD}.txt", "level": "INFO", "enqueue": True}
])

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:10240"

# Configuration
server_base = os.getenv('server_base')
config = load_config("config/config.yaml")

async def start(user_request: str, n_round: int = 2):
    team = Team()
    team.hire([XIMU_AI4Simulations()])
    requirement = UserRequirement(instruction=user_request)
    team.add_req(requirement)
    await team.run(n_round=n_round)

async def main():
    while True:
        user_input = input("\n\n老板，您好（输入 '结束' 或 'exit' 退出）：")
        if user_input in ("结束", "exit"):
            print("再见！")
            break
        else:
            await start(user_input, n_round=3)

if __name__ == "__main__":
    asyncio.run(main())
