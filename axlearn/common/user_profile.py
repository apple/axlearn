# Copyright © 2025 Apple Inc.

"""User profile module for learning user preferences, habits, and interests.

This module provides functionality to track and learn from user behavior in AXLearn,
including commonly used commands, preferred configurations, model types, and areas of interest.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class UserInterest:
    """Represents a user's interest in a specific area."""
    
    category: str  # e.g., "nlp", "vision", "speech", "multimodal"
    subcategories: List[str] = field(default_factory=list)  # e.g., ["gpt", "bert", "t5"]
    frequency: int = 0  # How often this interest is engaged with
    last_accessed: Optional[str] = None  # ISO timestamp of last access
    
    def update_access(self):
        """Update the last accessed timestamp and increment frequency."""
        self.last_accessed = datetime.now(timezone.utc).isoformat()
        self.frequency += 1


@dataclass
class CommandHistory:
    """Tracks command usage history."""
    
    command: str
    count: int = 0
    last_used: Optional[str] = None
    
    def record_use(self):
        """Record a command usage."""
        self.count += 1
        self.last_used = datetime.now(timezone.utc).isoformat()


@dataclass
class UserProfile:
    """Main user profile class that tracks user preferences, habits, and interests."""
    
    # Basic profile info
    profile_version: str = "1.0.0"
    created_at: Optional[str] = None
    last_updated: Optional[str] = None
    
    # User interests and preferences
    interests: Dict[str, UserInterest] = field(default_factory=dict)
    
    # Command usage tracking
    command_history: Dict[str, CommandHistory] = field(default_factory=dict)
    
    # Preferred configurations
    favorite_configs: List[str] = field(default_factory=list)
    
    # Model and experiment preferences
    frequently_used_models: Dict[str, int] = field(default_factory=dict)  # model_type -> count
    experiment_patterns: List[str] = field(default_factory=list)  # Common experiment types
    
    # Learning preferences
    preferred_accelerator: Optional[str] = None  # "tpu", "gpu", etc.
    typical_scale: Optional[str] = None  # "small", "medium", "large"
    
    def __post_init__(self):
        """Initialize timestamps if not set."""
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if self.last_updated is None:
            self.last_updated = datetime.now(timezone.utc).isoformat()
    
    def track_command(self, command: str):
        """Track a command execution.
        
        Args:
            command: The command that was executed.
        """
        if command not in self.command_history:
            self.command_history[command] = CommandHistory(command=command)
        self.command_history[command].record_use()
        self._update_timestamp()
    
    def track_interest(self, category: str, subcategory: Optional[str] = None):
        """Track user interest in a specific area.
        
        Args:
            category: Main category of interest (e.g., "nlp", "vision").
            subcategory: Optional subcategory (e.g., "gpt", "transformer").
        """
        if category not in self.interests:
            self.interests[category] = UserInterest(category=category)
        
        interest = self.interests[category]
        interest.update_access()
        
        if subcategory and subcategory not in interest.subcategories:
            interest.subcategories.append(subcategory)
        
        self._update_timestamp()
    
    def track_model_usage(self, model_type: str):
        """Track usage of a specific model type.
        
        Args:
            model_type: Type of model being used (e.g., "gpt", "bert", "vit").
        """
        if model_type not in self.frequently_used_models:
            self.frequently_used_models[model_type] = 0
        self.frequently_used_models[model_type] += 1
        self._update_timestamp()
    
    def add_favorite_config(self, config_name: str):
        """Add a configuration to favorites.
        
        Args:
            config_name: Name of the configuration.
        """
        if config_name not in self.favorite_configs:
            self.favorite_configs.append(config_name)
            self._update_timestamp()
    
    def set_preference(self, key: str, value: Any):
        """Set a general preference.
        
        Args:
            key: Preference key (e.g., "preferred_accelerator", "typical_scale").
            value: Preference value.
        """
        if hasattr(self, key):
            setattr(self, key, value)
            self._update_timestamp()
    
    def get_top_interests(self, limit: int = 5) -> List[UserInterest]:
        """Get top interests by frequency.
        
        Args:
            limit: Maximum number of interests to return.
            
        Returns:
            List of UserInterest objects sorted by frequency.
        """
        sorted_interests = sorted(
            self.interests.values(),
            key=lambda x: x.frequency,
            reverse=True
        )
        return sorted_interests[:limit]
    
    def get_top_commands(self, limit: int = 10) -> List[CommandHistory]:
        """Get most frequently used commands.
        
        Args:
            limit: Maximum number of commands to return.
            
        Returns:
            List of CommandHistory objects sorted by usage count.
        """
        sorted_commands = sorted(
            self.command_history.values(),
            key=lambda x: x.count,
            reverse=True
        )
        return sorted_commands[:limit]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the user profile.
        
        Returns:
            Dictionary containing profile summary.
        """
        return {
            "profile_version": self.profile_version,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "total_interests": len(self.interests),
            "top_interests": [
                {"category": i.category, "frequency": i.frequency}
                for i in self.get_top_interests(3)
            ],
            "total_commands": len(self.command_history),
            "top_commands": [
                {"command": c.command, "count": c.count}
                for c in self.get_top_commands(5)
            ],
            "favorite_configs": self.favorite_configs,
            "top_models": sorted(
                self.frequently_used_models.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "preferred_accelerator": self.preferred_accelerator,
            "typical_scale": self.typical_scale,
        }
    
    def _update_timestamp(self):
        """Update the last_updated timestamp."""
        self.last_updated = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary.
        
        Returns:
            Dictionary representation of the profile.
        """
        return {
            "profile_version": self.profile_version,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "interests": {
                k: asdict(v) for k, v in self.interests.items()
            },
            "command_history": {
                k: asdict(v) for k, v in self.command_history.items()
            },
            "favorite_configs": self.favorite_configs,
            "frequently_used_models": self.frequently_used_models,
            "experiment_patterns": self.experiment_patterns,
            "preferred_accelerator": self.preferred_accelerator,
            "typical_scale": self.typical_scale,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Create profile from dictionary.
        
        Args:
            data: Dictionary representation of the profile.
            
        Returns:
            UserProfile instance.
        """
        # Reconstruct interests
        interests = {
            k: UserInterest(**v) for k, v in data.get("interests", {}).items()
        }
        
        # Reconstruct command history
        command_history = {
            k: CommandHistory(**v) for k, v in data.get("command_history", {}).items()
        }
        
        return cls(
            profile_version=data.get("profile_version", "1.0.0"),
            created_at=data.get("created_at"),
            last_updated=data.get("last_updated"),
            interests=interests,
            command_history=command_history,
            favorite_configs=data.get("favorite_configs", []),
            frequently_used_models=data.get("frequently_used_models", {}),
            experiment_patterns=data.get("experiment_patterns", []),
            preferred_accelerator=data.get("preferred_accelerator"),
            typical_scale=data.get("typical_scale"),
        )


class UserProfileManager:
    """Manager class for loading, saving, and managing user profiles."""
    
    DEFAULT_PROFILE_DIR = Path.home() / ".axlearn"
    DEFAULT_PROFILE_FILE = "user_profile.json"
    
    def __init__(self, profile_path: Optional[Path] = None):
        """Initialize the profile manager.
        
        Args:
            profile_path: Optional custom path for the profile file.
        """
        if profile_path is None:
            profile_path = self.DEFAULT_PROFILE_DIR / self.DEFAULT_PROFILE_FILE
        
        self.profile_path = Path(profile_path)
        self._profile: Optional[UserProfile] = None
    
    def load_profile(self) -> UserProfile:
        """Load user profile from disk.
        
        Returns:
            UserProfile instance.
        """
        if self.profile_path.exists():
            try:
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._profile = UserProfile.from_dict(data)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                # If profile is corrupted, create a new one
                print(f"Warning: Could not load profile ({e}). Creating new profile.")
                self._profile = UserProfile()
        else:
            self._profile = UserProfile()
        
        return self._profile
    
    def save_profile(self, profile: Optional[UserProfile] = None):
        """Save user profile to disk.
        
        Args:
            profile: Profile to save. If None, uses the currently loaded profile.
        """
        if profile is None:
            profile = self._profile
        
        if profile is None:
            raise ValueError("No profile to save")
        
        # Ensure directory exists
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save profile
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, indent=2)
    
    def get_profile(self) -> UserProfile:
        """Get the current profile, loading it if necessary.
        
        Returns:
            UserProfile instance.
        """
        if self._profile is None:
            self.load_profile()
        return self._profile
    
    def reset_profile(self):
        """Reset the user profile to a new empty profile."""
        self._profile = UserProfile()
        self.save_profile()
    
    def delete_profile(self):
        """Delete the user profile file."""
        if self.profile_path.exists():
            self.profile_path.unlink()
        self._profile = None


# Global profile manager instance
_global_profile_manager: Optional[UserProfileManager] = None


def get_profile_manager() -> UserProfileManager:
    """Get the global profile manager instance.
    
    Returns:
        UserProfileManager instance.
    """
    global _global_profile_manager
    if _global_profile_manager is None:
        _global_profile_manager = UserProfileManager()
    return _global_profile_manager


def get_user_profile() -> UserProfile:
    """Get the current user profile.
    
    Returns:
        UserProfile instance.
    """
    manager = get_profile_manager()
    return manager.get_profile()


def save_user_profile(profile: Optional[UserProfile] = None):
    """Save the user profile.
    
    Args:
        profile: Profile to save. If None, saves the current profile.
    """
    manager = get_profile_manager()
    manager.save_profile(profile)
