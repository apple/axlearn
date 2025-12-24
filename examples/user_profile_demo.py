#!/usr/bin/env python3
# Copyright © 2025 Apple Inc.

"""Example demonstrating the AXLearn user profile system.

This script shows how the user profile system can learn about your preferences,
habits, and interests while using AXLearn.
"""

import tempfile
from pathlib import Path

from axlearn.common.user_profile import UserProfileManager


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main():
    """Run the user profile demonstration."""
    
    # For demo purposes, use a temporary profile
    # In normal use, the profile would be stored in ~/.axlearn/user_profile.json
    temp_dir = tempfile.mkdtemp()
    profile_path = Path(temp_dir) / "demo_profile.json"
    
    print_section("AXLearn User Profile System Demo")
    print("This demo shows how the system learns from your activity.\n")
    print(f"Demo profile location: {profile_path}")
    
    # Initialize profile manager
    manager = UserProfileManager(profile_path)
    profile = manager.get_profile()
    
    print_section("Simulating User Activity")
    
    # Simulate working on NLP projects
    print("📚 Working on NLP projects...")
    for _ in range(5):
        profile.track_interest("nlp", "gpt")
    profile.track_interest("nlp", "bert")
    profile.track_interest("nlp", "t5")
    profile.track_model_usage("transformer")
    profile.track_model_usage("transformer")
    profile.track_model_usage("transformer")
    profile.track_command("train")
    profile.track_command("train")
    profile.add_favorite_config("configs/nlp/gpt_large.py")
    print("  ✓ Tracked NLP work: GPT models, transformers, training runs")
    
    # Simulate working on vision projects
    print("\n🖼️  Working on computer vision projects...")
    for _ in range(3):
        profile.track_interest("vision", "vit")
    profile.track_interest("vision", "detection")
    profile.track_model_usage("vit")
    profile.track_model_usage("cnn")
    profile.track_command("eval")
    profile.add_favorite_config("configs/vision/vit_base.py")
    print("  ✓ Tracked vision work: ViT models, CNN, evaluation runs")
    
    # Set some preferences
    print("\n⚙️  Setting preferences...")
    profile.set_preference("preferred_accelerator", "tpu")
    profile.set_preference("typical_scale", "large")
    print("  ✓ Preferences set: TPU, large scale")
    
    # Save the profile
    manager.save_profile()
    print("\n💾 Profile saved!")
    
    print_section("What the System Learned")
    
    # Display top interests
    print("🎯 Top Interests:")
    top_interests = profile.get_top_interests(limit=5)
    for i, interest in enumerate(top_interests, 1):
        subcats = ", ".join(interest.subcategories[:3]) if interest.subcategories else "none"
        print(f"  {i}. {interest.category.upper()}")
        print(f"     - Frequency: {interest.frequency} accesses")
        print(f"     - Subcategories: {subcats}")
    
    # Display top commands
    print("\n⚡ Top Commands:")
    top_commands = profile.get_top_commands(limit=5)
    for i, cmd in enumerate(top_commands, 1):
        print(f"  {i}. '{cmd.command}' - used {cmd.count} times")
    
    # Display model preferences
    print("\n🤖 Model Usage:")
    sorted_models = sorted(
        profile.frequently_used_models.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for i, (model, count) in enumerate(sorted_models[:5], 1):
        print(f"  {i}. {model}: {count} times")
    
    # Display favorite configs
    print("\n⭐ Favorite Configs:")
    for config in profile.favorite_configs:
        print(f"  - {config}")
    
    # Display preferences
    print("\n🎛️  Preferences:")
    print(f"  - Preferred Accelerator: {profile.preferred_accelerator}")
    print(f"  - Typical Scale: {profile.typical_scale}")
    
    print_section("Profile Summary")
    
    summary = profile.get_summary()
    print(f"📊 Statistics:")
    print(f"  - Total Interests Tracked: {summary['total_interests']}")
    print(f"  - Total Commands Tracked: {summary['total_commands']}")
    print(f"  - Favorite Configs: {len(summary['favorite_configs'])}")
    print(f"  - Model Types Used: {len(profile.frequently_used_models)}")
    
    print_section("How This Helps You")
    
    print("The system can now:")
    print("  • Suggest relevant documentation for NLP and vision tasks")
    print("  • Recommend configurations similar to your favorites")
    print("  • Show examples using transformers and ViT models")
    print("  • Optimize for TPU and large-scale workloads")
    print("  • Prioritize content about GPT and ViT architectures")
    
    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)
    print(f"\nYour actual profile would be stored at:")
    print(f"  ~/.axlearn/user_profile.json")
    print("\nTo view your profile:")
    print("  python -m axlearn.cli.profile --action=summary")
    print("\n")


if __name__ == "__main__":
    main()
