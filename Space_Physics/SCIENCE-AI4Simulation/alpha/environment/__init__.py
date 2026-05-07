#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   :

from alpha.environment.base_env import Environment
from alpha.environment.android_env.android_env import AndroidEnv
from alpha.environment.mincraft_env.mincraft_env import MincraftExtEnv
from alpha.environment.werewolf_env.werewolf_env import WerewolfEnv
from alpha.environment.stanford_town_env.stanford_town_env import StanfordTownEnv
from alpha.environment.software_env.software_env import SoftwareEnv


__all__ = ["AndroidEnv", "MincraftExtEnv", "WerewolfEnv", "StanfordTownEnv", "SoftwareEnv", "Environment"]
