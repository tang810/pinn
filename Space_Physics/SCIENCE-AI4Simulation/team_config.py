"""
Created Date: Sat May 25 03:50:52 UTC 2024
"""
import os
import re
import sys
import fire
import pandas as pd

from alpha.team import Team
from alpha.roles import Role
from alpha.logs import logger
from alpha.schema import Message
from alpha.actions import Action, UserRequirement

from src.team_config import XIMU_AI4Simulations

from dotenv import load_dotenv
load_dotenv()
server_base = os.getenv('server_base')
handler = {"sink": sys.stdout, "level": "ERROR"}
logger.configure(handlers=[handler])


async def start(
    idea: str = "",
    investment: float = 0,
    n_round: int = 1,
    add_human: bool = True,
):

    team = Team()
    team.hire(
        [
            XIMU_AI4Simulations(),
        ]
    )


    team.run_project(idea)
    await team.run(n_round=n_round)

async def main():
    while True:
        userInput = input("\n\n老板，您好：").encode('utf-8').decode('utf-8')
        if userInput=="结束" or userInput=="exit":
            break
        else:
            await start(userInput)



if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
