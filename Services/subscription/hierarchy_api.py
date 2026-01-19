"""FastAPI router for user hierarchy management"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import logging

from Databases.app_data_db_connection import get_db
from .hierarchy_service import HierarchyService
from .hierarchy_schemas import (
    AddChildRequest,
    ChildUserResponse,
    HierarchyTreeNode,
    AccessibleUsersResponse,
    HierarchyResponse
)
from .subscription_models import UserDetails

logger = logging.getLogger(__name__)

hierarchy_router = APIRouter(prefix="/api/hierarchy", tags=["User Hierarchy"])


# ============================================================================
# HIERARCHY MANAGEMENT ENDPOINTS
# ============================================================================

@hierarchy_router.post("/add-child", response_model=HierarchyResponse)
async def add_child_user(
    request: AddChildRequest,
    db: Session = Depends(get_db)
):
    """
    Add a child user under a parent in the hierarchy
    
    Creates a parent-child relationship where the parent can access
    and manage the child's strategies and data.
    
    **Example:**
    ```json
    {
        "parent_email": "anjan@gmail.com",
        "child_email": "rachit.gour@gmail.com",
        "child_name": "Rachit Gour"
    }
    ```
    """
    try:
        # Verify parent exists
        parent = db.query(UserDetails).filter(
            UserDetails.user_email == request.parent_email
        ).first()
        
        if not parent:
            raise HTTPException(
                status_code=404,
                detail=f"Parent user {request.parent_email} not found"
            )
        
        # Add user to hierarchy
        hierarchy = HierarchyService.add_user_to_hierarchy(
            parent_email=request.parent_email,
            child_email=request.child_email,
            child_name=request.child_name,
            created_by=request.parent_email,
            db=db,
            role=request.role
        )
        
        if not hierarchy:
            raise HTTPException(
                status_code=400,
                detail="Failed to add user to hierarchy. User may already exist."
            )
        
        return HierarchyResponse(
            success=True,
            message=f"Successfully added {request.child_email} as child of {request.parent_email}",
            data={
                "user_email": hierarchy.user_email,
                "parent_email": hierarchy.parent_email,
                "hierarchy_level": hierarchy.hierarchy_level
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding child user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@hierarchy_router.get("/children/{user_email}", response_model=List[ChildUserResponse])
async def get_children(
    user_email: str,
    db: Session = Depends(get_db)
):
    """
    Get direct children of a user
    
    Returns only the immediate children (one level down) of the specified user.
    
    **Example Response:**
    ```json
    [
        {
            "user_email": "rachit.gour@gmail.com",
            "user_name": "Rachit Gour",
            "hierarchy_level": 1,
            "created_at": "2024-01-15T10:30:00"
        }
    ]
    ```
    """
    try:
        children = HierarchyService.get_direct_children(user_email, db)
        return children
        
    except Exception as e:
        logger.error(f"Error getting children: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@hierarchy_router.get("/descendants/{user_email}", response_model=List[str])
async def get_all_descendants(
    user_email: str,
    db: Session = Depends(get_db)
):
    """
    Get all descendants (recursive) of a user
    
    Returns all users in the subtree below the specified user,
    including children, grandchildren, etc.
    
    **Example Response:**
    ```json
    [
        "rachit.gour@gmail.com",
        "vishal.chauhan@gmail.com",
        "shourya@gmail.com"
    ]
    ```
    """
    try:
        descendants = HierarchyService.get_all_descendants(user_email, db)
        return descendants
        
    except Exception as e:
        logger.error(f"Error getting descendants: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@hierarchy_router.get("/tree/{user_email}", response_model=HierarchyTreeNode)
async def get_hierarchy_tree(
    user_email: str,
    db: Session = Depends(get_db)
):
    """
    Get full hierarchy tree starting from a user
    
    Returns a nested tree structure showing the user and all their descendants.
    
    **Example Response:**
    ```json
    {
        "email": "anjan@gmail.com",
        "name": "Anjan",
        "level": 0,
        "children": [
            {
                "email": "rachit.gour@gmail.com",
                "name": "Rachit Gour",
                "level": 1,
                "children": [
                    {
                        "email": "vishal.chauhan@gmail.com",
                        "name": "Vishal Chauhan",
                        "level": 2,
                        "children": []
                    }
                ]
            }
        ]
    }
    ```
    """
    try:
        tree = HierarchyService.get_hierarchy_tree(user_email, db)
        
        if not tree:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_email} not found"
            )
        
        return tree
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting hierarchy tree: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@hierarchy_router.get("/accessible-users/{user_email}", response_model=AccessibleUsersResponse)
async def get_accessible_users(
    user_email: str,
    db: Session = Depends(get_db)
):
    """
    Get all users accessible to the specified user
    
    Returns the user themselves plus all their descendants.
    Useful for filtering strategies and data.
    
    **Example Response:**
    ```json
    {
        "user_email": "anjan@gmail.com",
        "accessible_users": [
            "anjan@gmail.com",
            "rachit.gour@gmail.com",
            "vishal.chauhan@gmail.com",
            "shourya@gmail.com"
        ],
        "count": 4
    }
    ```
    """
    try:
        accessible = HierarchyService.get_accessible_users(user_email, db)
        
        return AccessibleUsersResponse(
            user_email=user_email,
            accessible_users=accessible,
            count=len(accessible)
        )
        
    except Exception as e:
        logger.error(f"Error getting accessible users: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@hierarchy_router.get("/verify-access/{accessor_email}/{target_email}")
async def verify_access(
    accessor_email: str,
    target_email: str,
    db: Session = Depends(get_db)
):
    """
    Verify if accessor can access target's data
    
    Returns true if the target is the accessor themselves or a descendant.
    
    **Example Response:**
    ```json
    {
        "accessor": "anjan@gmail.com",
        "target": "rachit.gour@gmail.com",
        "has_access": true
    }
    ```
    """
    try:
        has_access = HierarchyService.verify_access(accessor_email, target_email, db)
        
        return {
            "accessor": accessor_email,
            "target": target_email,
            "has_access": has_access
        }
        
    except Exception as e:
        logger.error(f"Error verifying access: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@hierarchy_router.delete("/remove-child/{child_email}", response_model=HierarchyResponse)
async def remove_child(
    child_email: str,
    db: Session = Depends(get_db)
):
    """
    Remove a child user from the hierarchy
    
    Performs a soft delete by setting is_active to False.
    The user's data is preserved but they are no longer in the hierarchy.
    
    **Note:** This does not delete the user account, only the hierarchy relationship.
    """
    try:
        success = HierarchyService.remove_user_from_hierarchy(child_email, db)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"User {child_email} not found in hierarchy"
            )
        
        return HierarchyResponse(
            success=True,
            message=f"Successfully removed {child_email} from hierarchy"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing child: {e}")
        raise HTTPException(status_code=500, detail=str(e))
