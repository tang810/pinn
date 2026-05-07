#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : __init__.py
"""

from alpha.roles.role import Role
from alpha.roles.architect import Architect
from alpha.roles.project_manager import ProjectManager
from alpha.roles.product_manager import ProductManager
from alpha.roles.engineer import Engineer
from alpha.roles.qa_engineer import QaEngineer
from alpha.roles.searcher import Searcher
from alpha.roles.sales import Sales
from alpha.roles.customer_service import CustomerService


__all__ = [
    "Role",
    "Architect",
    "ProjectManager",
    "ProductManager",
    "Engineer",
    "QaEngineer",
    "Searcher",
    "Sales",
    "CustomerService",
]
