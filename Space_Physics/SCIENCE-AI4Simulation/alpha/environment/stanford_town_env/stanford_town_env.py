#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : MG StanfordTown Env

from alpha.environment.base_env import Environment
from alpha.environment.stanford_town_env.stanford_town_ext_env import (
    StanfordTownExtEnv,
)


class StanfordTownEnv(Environment, StanfordTownExtEnv):
    pass
