# Copyright © 2025 Apple Inc.

"""Tests for user profile module."""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from axlearn.common.user_profile import (
    CommandHistory,
    UserInterest,
    UserProfile,
    UserProfileManager,
)


class TestUserInterest(unittest.TestCase):
    """Tests for UserInterest class."""
    
    def test_initialization(self):
        """Test UserInterest initialization."""
        interest = UserInterest(category="nlp")
        self.assertEqual(interest.category, "nlp")
        self.assertEqual(interest.subcategories, [])
        self.assertEqual(interest.frequency, 0)
        self.assertIsNone(interest.last_accessed)
    
    def test_update_access(self):
        """Test update_access method."""
        interest = UserInterest(category="vision")
        initial_frequency = interest.frequency
        
        interest.update_access()
        
        self.assertEqual(interest.frequency, initial_frequency + 1)
        self.assertIsNotNone(interest.last_accessed)
        
        # Verify timestamp is valid ISO format
        datetime.fromisoformat(interest.last_accessed)


class TestCommandHistory(unittest.TestCase):
    """Tests for CommandHistory class."""
    
    def test_initialization(self):
        """Test CommandHistory initialization."""
        cmd = CommandHistory(command="train")
        self.assertEqual(cmd.command, "train")
        self.assertEqual(cmd.count, 0)
        self.assertIsNone(cmd.last_used)
    
    def test_record_use(self):
        """Test record_use method."""
        cmd = CommandHistory(command="eval")
        initial_count = cmd.count
        
        cmd.record_use()
        
        self.assertEqual(cmd.count, initial_count + 1)
        self.assertIsNotNone(cmd.last_used)
        
        # Verify timestamp is valid ISO format
        datetime.fromisoformat(cmd.last_used)


class TestUserProfile(unittest.TestCase):
    """Tests for UserProfile class."""
    
    def test_initialization(self):
        """Test UserProfile initialization."""
        profile = UserProfile()
        
        self.assertEqual(profile.profile_version, "1.0.0")
        self.assertIsNotNone(profile.created_at)
        self.assertIsNotNone(profile.last_updated)
        self.assertEqual(len(profile.interests), 0)
        self.assertEqual(len(profile.command_history), 0)
        self.assertEqual(profile.favorite_configs, [])
        self.assertEqual(profile.frequently_used_models, {})
    
    def test_track_command(self):
        """Test track_command method."""
        profile = UserProfile()
        
        profile.track_command("train")
        self.assertIn("train", profile.command_history)
        self.assertEqual(profile.command_history["train"].count, 1)
        
        profile.track_command("train")
        self.assertEqual(profile.command_history["train"].count, 2)
        
        profile.track_command("eval")
        self.assertIn("eval", profile.command_history)
        self.assertEqual(profile.command_history["eval"].count, 1)
    
    def test_track_interest(self):
        """Test track_interest method."""
        profile = UserProfile()
        
        profile.track_interest("nlp")
        self.assertIn("nlp", profile.interests)
        self.assertEqual(profile.interests["nlp"].frequency, 1)
        
        profile.track_interest("nlp", "gpt")
        self.assertEqual(profile.interests["nlp"].frequency, 2)
        self.assertIn("gpt", profile.interests["nlp"].subcategories)
        
        profile.track_interest("vision", "vit")
        self.assertIn("vision", profile.interests)
        self.assertEqual(profile.interests["vision"].frequency, 1)
        self.assertIn("vit", profile.interests["vision"].subcategories)
    
    def test_track_model_usage(self):
        """Test track_model_usage method."""
        profile = UserProfile()
        
        profile.track_model_usage("gpt")
        self.assertEqual(profile.frequently_used_models["gpt"], 1)
        
        profile.track_model_usage("gpt")
        self.assertEqual(profile.frequently_used_models["gpt"], 2)
        
        profile.track_model_usage("bert")
        self.assertEqual(profile.frequently_used_models["bert"], 1)
    
    def test_add_favorite_config(self):
        """Test add_favorite_config method."""
        profile = UserProfile()
        
        profile.add_favorite_config("config1")
        self.assertIn("config1", profile.favorite_configs)
        
        # Adding same config again should not duplicate
        profile.add_favorite_config("config1")
        self.assertEqual(profile.favorite_configs.count("config1"), 1)
        
        profile.add_favorite_config("config2")
        self.assertEqual(len(profile.favorite_configs), 2)
    
    def test_set_preference(self):
        """Test set_preference method."""
        profile = UserProfile()
        
        profile.set_preference("preferred_accelerator", "tpu")
        self.assertEqual(profile.preferred_accelerator, "tpu")
        
        profile.set_preference("typical_scale", "large")
        self.assertEqual(profile.typical_scale, "large")
    
    def test_get_top_interests(self):
        """Test get_top_interests method."""
        profile = UserProfile()
        
        # Add interests with different frequencies
        for _ in range(5):
            profile.track_interest("nlp")
        for _ in range(3):
            profile.track_interest("vision")
        for _ in range(1):
            profile.track_interest("speech")
        
        top_interests = profile.get_top_interests(2)
        self.assertEqual(len(top_interests), 2)
        self.assertEqual(top_interests[0].category, "nlp")
        self.assertEqual(top_interests[0].frequency, 5)
        self.assertEqual(top_interests[1].category, "vision")
        self.assertEqual(top_interests[1].frequency, 3)
    
    def test_get_top_commands(self):
        """Test get_top_commands method."""
        profile = UserProfile()
        
        # Track commands with different frequencies
        for _ in range(10):
            profile.track_command("train")
        for _ in range(5):
            profile.track_command("eval")
        for _ in range(2):
            profile.track_command("test")
        
        top_commands = profile.get_top_commands(2)
        self.assertEqual(len(top_commands), 2)
        self.assertEqual(top_commands[0].command, "train")
        self.assertEqual(top_commands[0].count, 10)
        self.assertEqual(top_commands[1].command, "eval")
        self.assertEqual(top_commands[1].count, 5)
    
    def test_get_summary(self):
        """Test get_summary method."""
        profile = UserProfile()
        
        profile.track_command("train")
        profile.track_interest("nlp", "gpt")
        profile.track_model_usage("transformer")
        profile.add_favorite_config("config1")
        profile.set_preference("preferred_accelerator", "gpu")
        
        summary = profile.get_summary()
        
        self.assertIn("profile_version", summary)
        self.assertIn("created_at", summary)
        self.assertIn("last_updated", summary)
        self.assertEqual(summary["total_interests"], 1)
        self.assertEqual(summary["total_commands"], 1)
        self.assertEqual(summary["favorite_configs"], ["config1"])
        self.assertEqual(summary["preferred_accelerator"], "gpu")
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        profile = UserProfile()
        
        # Add some data
        profile.track_command("train")
        profile.track_interest("nlp", "gpt")
        profile.track_model_usage("transformer")
        profile.add_favorite_config("config1")
        profile.set_preference("preferred_accelerator", "tpu")
        
        # Convert to dict
        data = profile.to_dict()
        
        # Verify dict structure
        self.assertIn("profile_version", data)
        self.assertIn("interests", data)
        self.assertIn("command_history", data)
        self.assertIn("favorite_configs", data)
        
        # Recreate from dict
        profile2 = UserProfile.from_dict(data)
        
        # Verify data is preserved
        self.assertEqual(profile2.profile_version, profile.profile_version)
        self.assertIn("nlp", profile2.interests)
        self.assertEqual(profile2.interests["nlp"].frequency, 1)
        self.assertIn("train", profile2.command_history)
        self.assertEqual(profile2.command_history["train"].count, 1)
        self.assertIn("config1", profile2.favorite_configs)
        self.assertEqual(profile2.frequently_used_models["transformer"], 1)
        self.assertEqual(profile2.preferred_accelerator, "tpu")


class TestUserProfileManager(unittest.TestCase):
    """Tests for UserProfileManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.profile_path = Path(self.temp_dir) / "test_profile.json"
    
    def test_initialization(self):
        """Test UserProfileManager initialization."""
        manager = UserProfileManager(self.profile_path)
        self.assertEqual(manager.profile_path, self.profile_path)
    
    def test_load_and_save_profile(self):
        """Test loading and saving profile."""
        manager = UserProfileManager(self.profile_path)
        
        # Load profile (should create new one)
        profile = manager.load_profile()
        self.assertIsNotNone(profile)
        self.assertIsInstance(profile, UserProfile)
        
        # Add some data
        profile.track_command("test")
        profile.track_interest("vision")
        
        # Save profile
        manager.save_profile(profile)
        
        # Verify file exists
        self.assertTrue(self.profile_path.exists())
        
        # Create new manager and load profile
        manager2 = UserProfileManager(self.profile_path)
        profile2 = manager2.load_profile()
        
        # Verify data is preserved
        self.assertIn("test", profile2.command_history)
        self.assertIn("vision", profile2.interests)
    
    def test_get_profile(self):
        """Test get_profile method."""
        manager = UserProfileManager(self.profile_path)
        
        # First call should load profile
        profile1 = manager.get_profile()
        self.assertIsNotNone(profile1)
        
        # Second call should return same profile
        profile2 = manager.get_profile()
        self.assertIs(profile1, profile2)
    
    def test_reset_profile(self):
        """Test reset_profile method."""
        manager = UserProfileManager(self.profile_path)
        
        # Load and modify profile
        profile = manager.get_profile()
        profile.track_command("test")
        manager.save_profile()
        
        # Reset profile
        manager.reset_profile()
        
        # Verify profile is reset
        new_profile = manager.get_profile()
        self.assertEqual(len(new_profile.command_history), 0)
        self.assertEqual(len(new_profile.interests), 0)
    
    def test_delete_profile(self):
        """Test delete_profile method."""
        manager = UserProfileManager(self.profile_path)
        
        # Create and save profile
        profile = manager.get_profile()
        manager.save_profile(profile)
        self.assertTrue(self.profile_path.exists())
        
        # Delete profile
        manager.delete_profile()
        self.assertFalse(self.profile_path.exists())
    
    def test_load_corrupted_profile(self):
        """Test loading a corrupted profile file."""
        manager = UserProfileManager(self.profile_path)
        
        # Create corrupted profile file
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.profile_path, "w") as f:
            f.write("invalid json content {{{")
        
        # Loading should create new profile instead of crashing
        profile = manager.load_profile()
        self.assertIsNotNone(profile)
        self.assertIsInstance(profile, UserProfile)
    
    def test_profile_persistence(self):
        """Test that profile changes persist across saves."""
        manager = UserProfileManager(self.profile_path)
        
        # Create profile and add data
        profile = manager.get_profile()
        profile.track_command("cmd1")
        profile.track_interest("nlp")
        manager.save_profile()
        
        # Add more data
        profile.track_command("cmd2")
        profile.track_interest("vision")
        manager.save_profile()
        
        # Load fresh and verify all data
        manager2 = UserProfileManager(self.profile_path)
        profile2 = manager2.load_profile()
        
        self.assertIn("cmd1", profile2.command_history)
        self.assertIn("cmd2", profile2.command_history)
        self.assertIn("nlp", profile2.interests)
        self.assertIn("vision", profile2.interests)


if __name__ == "__main__":
    unittest.main()
