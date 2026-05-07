# -*- coding: utf-8 -*-
import asyncio
import json
import re
from typing import AsyncIterator, Optional, Union

from openai import APIConnectionError, AsyncOpenAI, AsyncStream
from openai._base_client import AsyncHttpxClientWrapper
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from alpha.configs.llm_config import LLMConfig, LLMType
from alpha.logs import log_llm_stream, logger
from alpha.provider.base_llm import BaseLLM
from alpha.provider.constant import GENERAL_FUNCTION_SCHEMA
from alpha.provider.llm_provider_registry import register_provider
from alpha.schema import Message
from alpha.utils.common import CodeParser, decode_image
from alpha.utils.cost_manager import CostManager, Costs, TokenCostManager
from alpha.utils.exceptions import handle_exception
from alpha.utils.token_counter import (
    count_message_tokens,
    count_string_tokens,
    get_max_completion_tokens,
)

import yaml
def load_config(file_path):
    """Load configuration from a YAML file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


class SeLLM(BaseLLM):
    """Check https://platform.openai.com/examples for examples"""

    def __init__(self, api_key,base_url):
        self._init_client(api_key,base_url)
        self.auto_max_tokens = False

    def _init_client(self,api_key,base_url):
        """https://github.com/openai/openai-python#async-usage"""
        kwargs = self._make_client_kwargs(api_key,base_url)
        self.aclient = AsyncOpenAI(**kwargs)

    def _make_client_kwargs(self,api_key,base_url) -> dict:
        kwargs = {"api_key": api_key, "base_url": base_url}

        return kwargs

    
    async def _achat_completion_stream(self, messages: list[dict],max_tokens,model,temperature, timeout=3) -> str:
        
        response: AsyncStream[ChatCompletionChunk] = await self.aclient.chat.completions.create(
            **self._cons_kwargs(messages,model,max_tokens,temperature, timeout=timeout), stream=True
        )
        return response

   
    def _cons_kwargs(self, messages: list[dict],model,max_tokens,temperature,timeout=3, **extra_kwargs) -> dict:
        kwargs = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "model": model,
            "timeout": timeout,
        }
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return kwargs

    async def _achat_completion(self, messages: list[dict],max_tokens,model,temperature,  timeout=3) -> ChatCompletion:
        kwargs = self._cons_kwargs(messages, max_tokens=max_tokens,model=model,temperature=temperature, timeout=timeout)
        rsp: ChatCompletion = await self.aclient.chat.completions.create(**kwargs)
        # self._update_costs(rsp.usage)
        return rsp

    async def acompletion(self, messages: list[dict], timeout=3) -> ChatCompletion:
        return await self._achat_completion(messages, timeout=timeout)

    async def acompletion_text(self, messages: list[dict],max_tokens=8196,model="SE_V0.0",temperature=0.7,stream=True, timeout=3) -> str:
        """when streaming, print each token in place."""
        if stream:
            rsp =  await self._achat_completion_stream(messages,max_tokens,model,temperature, timeout=timeout)
            return rsp
        else:
            rsp1 = await self._achat_completion(messages,max_tokens,model,temperature, timeout=timeout)
            return self.get_choice_text(rsp1)
    
    def get_choice_text(self, rsp: ChatCompletion) -> str:
        """Required to provide the first text of choice"""
        return rsp.choices[0].message.content if rsp.choices else ""


async def main():
    llm=SeLLM(
        #base_url="http://192.168.6.204:40200/v1",
        base_url="http://www.science42.vip:40101/v1",
        api_key="xxx"

    )

    prompt = ""
    message = [llm._default_system_msg()]
    message.append(llm._user_msg(prompt))
    response = await llm.acompletion_text(message,temperature=0.7,model="SE_V0.0",timeout=3)

    collected_messages = []
    async for chunk in response:
        chunk_message = chunk.choices[0].delta.content or "" if chunk.choices else ""
        collected_messages.append(chunk_message)
        print(chunk_message, end='', flush=True)  # 实时输出每个chunk

if __name__ == "__main__":
    asyncio.run(main())

