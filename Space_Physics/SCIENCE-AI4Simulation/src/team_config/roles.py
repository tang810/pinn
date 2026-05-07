# -*- coding: utf-8 -*-
from alpha.roles import Role
from alpha.actions import UserRequirement
from .actions import Coding, DiT4ScienceAction, ACDM4ScienceAction, GNN4SInferenceAction, DLSurrogateInferenceAction

class XIMU_AI4Simulations(Role):
    name: str = "RoleName_CN/EN/Dev"
    profile: str = "功能Subscription"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([UserRequirement])
        self.set_actions([Coding, DiT4ScienceAction, ACDM4ScienceAction, GNN4SInferenceAction, DLSurrogateInferenceAction])
