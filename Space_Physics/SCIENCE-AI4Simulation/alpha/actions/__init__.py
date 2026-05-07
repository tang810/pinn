#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:44
@Author  : alexanderwu
@File    : __init__.py
"""
from enum import Enum

from alpha.actions.action import Action
from alpha.actions.action_output import ActionOutput
from alpha.actions.add_requirement import UserRequirement
from alpha.actions.debug_error import DebugError
from alpha.actions.design_api import WriteDesign
from alpha.actions.design_api_review import DesignReview
from alpha.actions.project_management import WriteTasks
from alpha.actions.research import CollectLinks, WebBrowseAndSummarize, ConductResearch
from alpha.actions.run_code import RunCode
from alpha.actions.search_and_summarize import SearchAndSummarize
from alpha.actions.write_code import WriteCode
from alpha.actions.write_code_review import WriteCodeReview
from alpha.actions.write_prd import WritePRD
from alpha.actions.write_prd_review import WritePRDReview
from alpha.actions.write_test import WriteTest
from alpha.actions.mi.execute_nb_code import ExecuteNbCode
from alpha.actions.mi.write_analysis_code import WriteCodeWithoutTools, WriteCodeWithTools
from alpha.actions.mi.write_plan import WritePlan


class ActionType(Enum):
    """All types of Actions, used for indexing."""

    ADD_REQUIREMENT = UserRequirement
    WRITE_PRD = WritePRD
    WRITE_PRD_REVIEW = WritePRDReview
    WRITE_DESIGN = WriteDesign
    DESIGN_REVIEW = DesignReview
    WRTIE_CODE = WriteCode
    WRITE_CODE_REVIEW = WriteCodeReview
    WRITE_TEST = WriteTest
    RUN_CODE = RunCode
    DEBUG_ERROR = DebugError
    WRITE_TASKS = WriteTasks
    SEARCH_AND_SUMMARIZE = SearchAndSummarize
    COLLECT_LINKS = CollectLinks
    WEB_BROWSE_AND_SUMMARIZE = WebBrowseAndSummarize
    CONDUCT_RESEARCH = ConductResearch
    EXECUTE_NB_CODE = ExecuteNbCode
    WRITE_CODE_WITHOUT_TOOLS = WriteCodeWithoutTools
    WRITE_CODE_WITH_TOOLS = WriteCodeWithTools
    WRITE_PLAN = WritePlan


__all__ = [
    "ActionType",
    "Action",
    "ActionOutput",
]
