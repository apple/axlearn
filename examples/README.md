# AXLearn Examples

This directory contains example scripts demonstrating various AXLearn features.

## User Profile Demo

The `user_profile_demo.py` script demonstrates the user profile system that learns about your preferences, habits, and interests.

### Running the Demo

```bash
# From the repository root
PYTHONPATH=/path/to/axlearn python3 examples/user_profile_demo.py
```

Or if you have AXLearn installed:

```bash
python3 examples/user_profile_demo.py
```

### What the Demo Shows

The demo simulates typical user activity with AXLearn and shows how the system learns from:
- Working on NLP projects (GPT, BERT, T5)
- Working on computer vision projects (ViT, detection)
- Training and evaluation runs
- Model usage patterns
- Configuration preferences
- Hardware and scale preferences

### Output

The demo provides a comprehensive view of what the system learns, including:
- Top interests by frequency
- Most used commands
- Model usage statistics
- Favorite configurations
- User preferences

### Learn More

See the [User Profile Documentation](../docs/user_profile.md) for complete details on:
- Using the profile system programmatically
- CLI commands for managing your profile
- Profile structure and data
- Privacy considerations
- Integration opportunities
