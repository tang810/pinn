"""
Filename: test_xddpm.py
Created Date: Sat May 25 03:50:52 UTC 2024
Author: Hui Xiang
"""
import re, os, sys
import fire

from alpha.actions import Action, UserRequirement
from alpha.logs import logger
from alpha.roles import Role
from alpha.schema import Message
from alpha.team import Team
from dit4science.inference import inference

import torch

from dit4science.model.dit_models import DiT_models
from dit4science.data.dataset_dit import TurbDataset

# sys.path.append(os.path.dirname(__file__))
from dit4science.download import find_model  # noqa
from dit4science.model.autoencoder import AutoencoderKL



handler = {"sink": sys.stdout, "level": "ERROR"}
logger.configure(handlers=[handler])

channel_idx = -1
data_path = "/data/sda/chunyang/DiT4Science/test/"
prop = None
dataset = TurbDataset(dataProp=prop, dataDir=data_path, gaussian_norm=False, channel_idx=channel_idx)

seed = 1
image_size = 128
model = f"DiT-B/2"  # noqa
num_sampling_steps = 250

ckpt = "/data/sda/chunyang/DiT4Science/weights/6300000.pt"  # on 6k, idx=-1, fully-latent space infrerence

num_classes = 1000
cfg_scale = 40

device = "cuda" if torch.cuda.is_available() else "cpu"

# ########### VAE init ###################
# Load model:
latent_size = image_size // 8
model = DiT_models[model](
        input_size=latent_size,
        # input_size=args.image_size,
        num_classes=num_classes,
).to(device)
# Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
ckpt_path = ckpt
state_dict = find_model(ckpt_path)
model.load_state_dict(state_dict)
# model.eval()  # important!

# Autoencoder config
config = {
    "monitor": "val/rec_loss",
    "embed_dim": 4,
    "lossconfig": {
        "target": "dit4science.loss.LPIPSWithDiscriminator",
        "params": {
            "disc_start": 50001,
            "kl_weight": 0.000001,
            "disc_weight": 0.5
        }
    },
    "ddconfig": {
        "double_z": True,
        "z_channels": 4,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 2, 4, 4],  # num_down = len(ch_mult) - 1
        "num_res_blocks": 2,
        "attn_resolutions": [],
        "dropout": 0.0
    }
}
vae_ckpt_cond = "/data/sda/chunyang/DiT4Science/weights/cond_all_epoch=152-step=167840.ckpt"
vae_ckpt_res = "/data/sda/chunyang/DiT4Science/weights/res_all_epoch=109-step=120669.ckpt"

vae_cond = AutoencoderKL(**config).to(device)
vae_cond.init_from_ckpt(vae_ckpt_cond)

vae_res = AutoencoderKL(**config).to(device)
vae_res.init_from_ckpt(vae_ckpt_res)


class Physics_Analysis(Action):
    name: str = "物理分析"
    desc: str = "物理分析"
    PROMPT_TEMPLATE: str  = """
    你是物理学家。你擅长解决物理学问题，你的的任务是根据用户提出的问题明确该问题类型，比如是正问题还是反问题，然后进行数学建模，给出控制方程的目标函数、初始条件、边界条件，并分析其物理意义。
    需要解决的问题是：{instruction}
    请给出你的分析和建模：
    """

    async def run(self, instruction: str):
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)
        analysis = await self._aask(prompt)
        return "用户的问题和计划：{}\n 给出的物理建模：{}".format(instruction,analysis)


class Physicist(Role):
    name: str = "惜惜"
    profile: str = "物理学家"
    def __init__(
        self, 
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._watch([UserRequirement])
        self.set_actions([Physics_Analysis])


class Solve(Action):
    name: str = "算法分析与工程实现"
    desc: str = "算法分析与工程实现"
    PROMPT_TEMPLATE: str  = """
    你是算法专家。你的任务是对物理学家提出的控制方程进行思路和算法分析并构建代码实现，请细化求解该问题所需要的必要和详细步骤，拆解参数，列出使用PINN算法来求解的过程。最后附上github参考链接：https://github.com/Scien42/NSFnet
    物理学家的建模信息为：{context}
    请给出你的算法分析与工程实现：
    """


    async def run(self, context: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context)
        content = await self._aask(prompt)
        return "之前的信息是：{}\n 得到的伪代码：{}".format(context, content)


class Solver(Role):
    name: str = "慕慕"
    profile: str = "算法专家"
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._watch([Physics_Analysis])
        self.set_actions([Solve])


class CodeParsing(Action):
    name: str = "提取PINN所需要的推理参数"
    prefix:str="提取PINN所需要的推理参数"
    desc: str = "提取PINN所需要的推理参数"
    PROMPT_TEMPLATE: str  = """
    你是代码分析工具。你的任务是提取推理函数的注释与参数信息，分析老板的输入之中是否包含所需要的参数信息，如果不满足或者未提供，返回函数默认注释，否则按照注释格式提取老板的参数信息送入求解器求解。
    之前的信息是：{context}
    请给出你的参数信息:
    """
    async def run(self, context: str):
        info = ''
        print(info)
        return "之前的参数：{}\n 得到的函数参数信息是：{}".format(context, info)
    
class CodeParser(Role):
    name: str = "夕夕"
    profile: str = "系统工程师"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([Solve])
        self.set_actions([CodeParsing])


class PINN_Inference(Action):
    name: str = ""
    prefix:str="解偏微分方程"
    desc: str = "解偏微分方程"
    PROMPT_TEMPLATE: str  = """
    你是PINN推理求解器。你的任务是从之前的信息中提取和分析推理函数所需要的参数，并检查老板的输入是否给出相关参数信息，如果不满足或者未提供，明确指出将按照注释提到的默认案例来求解作为参考，否则按照注释格式提取出老板的参数信息送入求解器求解。
    之前的信息是：{context}
    请给出你的求解:
    """
    async def run(self, context: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context)
        content = await self._aask(prompt)
        inference(dataset, model, vae_cond, vae_res, latent_size=latent_size)
        return "之前的参数：{}\n 得到的参数：{}".format(context, content)
    
class PINN_Solver(Role):
    name: str = "沐沐"
    profile: str = "PINN求解器"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([CodeParsing])
        self.set_actions([PINN_Inference])


async def start(
    idea: str = "",
    investment: float = 0,
    n_round: int = 4,
    add_human: bool = False,
):
    team = Team()
    team.hire(
        [
            Physicist(),
            Solver(),
            CodeParser(),
            PINN_Solver(),
        ]
    )


    team.run_project(idea)
    await team.run(n_round=n_round)

async def main():
    # inference()
    while True:
        userInput = input("\n\n老板，您好：").encode('utf-8').decode('utf-8')
        if userInput=="结束" or userInput=="exit":
            break
        else:
            await start(userInput)



if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
