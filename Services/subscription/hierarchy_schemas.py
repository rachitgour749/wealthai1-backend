"""Pydantic schemas for hierarchy API"""
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime


class AddChildRequest(BaseModel):
    """Request to add a child user"""
    parent_email: EmailStr
    child_email: EmailStr
    child_name: str
    role: Optional[str] = None


class ChildUserResponse(BaseModel):
    """Response for child user information"""
    user_email: str
    user_name: Optional[str]
    hierarchy_level: int
    created_at: Optional[str]


class HierarchyTreeNode(BaseModel):
    """Hierarchical tree node"""
    email: str
    name: Optional[str]
    level: int
    children: List['HierarchyTreeNode'] = []


class AccessibleUsersResponse(BaseModel):
    """Response for accessible users"""
    user_email: str
    accessible_users: List[str]
    count: int


class HierarchyResponse(BaseModel):
    """Generic hierarchy response"""
    success: bool
    message: str
    data: Optional[dict] = None


# Enable forward references for recursive model
HierarchyTreeNode.model_rebuild()
