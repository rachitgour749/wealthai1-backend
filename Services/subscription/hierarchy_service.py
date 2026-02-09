"""Hierarchy service for managing user parent-child relationships"""
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Optional
import logging

from .subscription_models import UserHierarchy, UserDetails

logger = logging.getLogger(__name__)


class HierarchyService:
    """Service for managing user hierarchy operations"""
    
    @staticmethod
    def get_all_descendants(user_email: str, db: Session) -> List[str]:
        """
        Recursively get all descendant emails using PostgreSQL recursive CTE
        
        Args:
            user_email: Parent user email
            db: Database session
        
        Returns:
            List of descendant user emails
        """
        try:
            # Recursive CTE query to get all descendants
            query = text("""
                WITH RECURSIVE descendants AS (
                    -- Base case: direct children
                    SELECT user_email, parent_email, hierarchy_level
                    FROM user_hierarchy
                    WHERE parent_email = :user_email AND is_active = TRUE
                    
                    UNION ALL
                    
                    -- Recursive case: children of children
                    SELECT h.user_email, h.parent_email, h.hierarchy_level
                    FROM user_hierarchy h
                    INNER JOIN descendants d ON h.parent_email = d.user_email
                    WHERE h.is_active = TRUE
                )
                SELECT user_email FROM descendants;
            """)
            
            result = db.execute(query, {"user_email": user_email})
            descendants = [row[0] for row in result.fetchall()]
            
            logger.info(f"Found {len(descendants)} descendants for {user_email}")
            return descendants
            
        except Exception as e:
            logger.error(f"Error getting descendants: {e}")
            # Rollback the transaction to prevent "transaction is aborted" errors
            db.rollback()
            return []
    
    @staticmethod
    def get_direct_children(user_email: str, db: Session) -> List[Dict]:
        """
        Get direct children of a user
        
        Args:
            user_email: Parent user email
            db: Database session
        
        Returns:
            List of child user dictionaries with details
        """
        try:
            children = db.query(UserHierarchy).filter(
                UserHierarchy.parent_email == user_email,
                UserHierarchy.is_active == True
            ).all()
            
            result = []
            for child in children:
                # Get user details
                user_details = db.query(UserDetails).filter(
                    UserDetails.user_email == child.user_email
                ).first()
                
                result.append({
                    'user_email': child.user_email,
                    'user_name': user_details.user_name if user_details else None,
                    'hierarchy_level': child.hierarchy_level,
                    'created_at': child.created_at.isoformat() if child.created_at else None
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting direct children: {e}")
            return []
    
    @staticmethod
    def get_hierarchy_tree(user_email: str, db: Session) -> Dict:
        """
        Build hierarchical tree structure starting from user
        
        Args:
            user_email: Root user email
            db: Database session
        
        Returns:
            Nested dictionary representing tree structure
        """
        try:
            # Get user details
            user = db.query(UserDetails).filter(
                UserDetails.user_email == user_email
            ).first()
            
            if not user:
                return {}
            
            # Get hierarchy info
            hierarchy = db.query(UserHierarchy).filter(
                UserHierarchy.user_email == user_email
            ).first()
            
            tree = {
                'email': user_email,
                'name': user.user_name,
                'level': hierarchy.hierarchy_level if hierarchy else 0,
                'children': []
            }
            
            # Get direct children
            children = HierarchyService.get_direct_children(user_email, db)
            
            # Recursively build tree for each child
            for child in children:
                child_tree = HierarchyService.get_hierarchy_tree(child['user_email'], db)
                tree['children'].append(child_tree)
            
            return tree
            
        except Exception as e:
            logger.error(f"Error building hierarchy tree: {e}")
            return {}
    
    @staticmethod
    def verify_access(accessor_email: str, target_email: str, db: Session) -> bool:
        """
        Verify if accessor has permission to access target's data
        
        Args:
            accessor_email: User requesting access
            target_email: User whose data is being accessed
            db: Database session
        
        Returns:
            True if access is allowed, False otherwise
        """
        # User can always access their own data
        if accessor_email == target_email:
            return True
        
        # Check if target is a descendant of accessor
        descendants = HierarchyService.get_all_descendants(accessor_email, db)
        return target_email in descendants
    
    @staticmethod
    def get_accessible_users(user_email: str, db: Session) -> List[str]:
        """
        Get all users accessible to the given user (self + descendants)
        
        Args:
            user_email: User email
            db: Database session
        
        Returns:
            List of accessible user emails
        """
        descendants = HierarchyService.get_all_descendants(user_email, db)
        accessible = [user_email] + descendants
        return accessible
    
    @staticmethod
    def add_user_to_hierarchy(
        parent_email: str,
        child_email: str,
        child_name: str,
        created_by: str,
        db: Session,
        role: Optional[str] = None
    ) -> Optional[UserHierarchy]:
        """
        Add a new user under a parent in the hierarchy
        
        Args:
            parent_email: Parent user email
            child_email: Child user email
            child_name: Child user name
            created_by: Email of user creating this relationship
            db: Database session
            role: Role of the child user (e.g., RM, Client)
        
        Returns:
            UserHierarchy object if successful, None otherwise
        """
        try:
            # Check if child already exists in hierarchy
            existing = db.query(UserHierarchy).filter(
                UserHierarchy.user_email == child_email
            ).first()
            
            if existing:
                logger.warning(f"User {child_email} already exists in hierarchy")
                return None
            
            # Determine hierarchy level based on role
            role_level_mapping = {
                'BRANCH': 0,
                'RM': 1,
                'CLIENT': 2
            }
            
            if role and role.upper() in role_level_mapping:
                child_level = role_level_mapping[role.upper()]
                logger.info(f"Assigned level {child_level} based on role {role}")
            else:
                # Fallback to parent_level + 1 if role is not recognized
                parent_hierarchy = db.query(UserHierarchy).filter(
                    UserHierarchy.user_email == parent_email
                ).first()
                parent_level = parent_hierarchy.hierarchy_level if parent_hierarchy else 0
                child_level = parent_level + 1
                logger.info(f"Fallback to incremental level {child_level} (parent level: {parent_level})")
            
            # Create or update user_details entry
            user_details = db.query(UserDetails).filter(
                UserDetails.user_email == child_email
            ).first()
            
            if not user_details:
                user_details = UserDetails(
                    user_email=child_email,
                    user_name=child_name,
                    status='TRIAL',
                    role=role.upper() if role else 'CLIENT'
                )
                db.add(user_details)
            else:
                # Update role if provided
                if role:
                    user_details.role = role.upper()
                    logger.info(f"Updated role for {child_email} to {role.upper()}")
            
            # Create hierarchy entry
            hierarchy = UserHierarchy(
                user_email=child_email,
                parent_email=parent_email,
                hierarchy_level=child_level,
                created_by=created_by,
                is_active=True
            )
            
            db.add(hierarchy)
            db.commit()
            db.refresh(hierarchy)
            
            logger.info(f"Added {child_email} as child of {parent_email} at level {child_level}")
            return hierarchy
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error adding user to hierarchy: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def remove_user_from_hierarchy(child_email: str, db: Session) -> bool:
        """
        Remove a user from the hierarchy
        
        Args:
            child_email: User email to remove
            db: Database session
        
        Returns:
            True if successful, False otherwise
        """
        try:
            hierarchy = db.query(UserHierarchy).filter(
                UserHierarchy.user_email == child_email
            ).first()
            
            if not hierarchy:
                logger.warning(f"User {child_email} not found in hierarchy")
                return False
            
            # Soft delete by setting is_active to False
            hierarchy.is_active = False
            db.commit()
            
            logger.info(f"Removed {child_email} from hierarchy")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error removing user from hierarchy: {e}")
            return False
