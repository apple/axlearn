# User Profile System

The AXLearn user profile system helps the library learn about your preferences, habits, and interests to provide a personalized experience.

## Overview

The user profile system tracks:
- **Commands**: Which AXLearn commands you use most frequently
- **Interests**: Your areas of focus (NLP, vision, speech, etc.)
- **Models**: Types of models you work with regularly
- **Configurations**: Your favorite configuration files
- **Preferences**: Your typical accelerator choice and scale preferences

All profile data is stored locally in `~/.axlearn/user_profile.json`.

## Usage

### Programmatic API

```python
from axlearn.common.user_profile import get_user_profile, save_user_profile

# Get the current user profile
profile = get_user_profile()

# Track command usage
profile.track_command("train")

# Track interests
profile.track_interest("nlp", subcategory="gpt")
profile.track_interest("vision", subcategory="vit")

# Track model usage
profile.track_model_usage("transformer")

# Add favorite configurations
profile.add_favorite_config("my_gpt_config.py")

# Set preferences
profile.set_preference("preferred_accelerator", "tpu")
profile.set_preference("typical_scale", "large")

# Save the profile
save_user_profile()

# Get insights
summary = profile.get_summary()
print(f"Total interests: {summary['total_interests']}")
print(f"Top commands: {summary['top_commands']}")

# Get top interests
top_interests = profile.get_top_interests(limit=5)
for interest in top_interests:
    print(f"{interest.category}: {interest.frequency} accesses")
```

### CLI Commands

View profile summary:
```bash
python -m axlearn.cli.profile --action=summary
```

View detailed profile:
```bash
python -m axlearn.cli.profile --action=view
```

Track activity manually:
```bash
python -m axlearn.cli.profile --action=track --category=nlp --subcategory=gpt
python -m axlearn.cli.profile --action=track --command=train
python -m axlearn.cli.profile --action=track --model=transformer
```

Export profile as JSON:
```bash
python -m axlearn.cli.profile --action=export
```

Reset profile:
```bash
python -m axlearn.cli.profile --action=reset
```

## Profile Structure

The user profile includes:

### Basic Information
- `profile_version`: Version of the profile format
- `created_at`: When the profile was created
- `last_updated`: Last time the profile was modified

### Interests
Each interest tracks:
- `category`: Main area (e.g., "nlp", "vision", "speech")
- `subcategories`: Specific topics within the category
- `frequency`: Number of times accessed
- `last_accessed`: Timestamp of last access

### Command History
Each command tracks:
- `command`: The command name
- `count`: Number of times used
- `last_used`: Timestamp of last use

### Model Usage
Dictionary mapping model types to usage counts

### Preferences
- `favorite_configs`: List of favorite configuration files
- `preferred_accelerator`: Preferred hardware ("tpu", "gpu", etc.)
- `typical_scale`: Typical project scale ("small", "medium", "large")
- `experiment_patterns`: Common experiment types

## Privacy

All profile data is stored locally on your machine. No data is transmitted or shared with external services. You can:
- View your profile at any time
- Export your data as JSON
- Reset your profile to clear all data
- Manually delete `~/.axlearn/user_profile.json`

## Integration

The profile system is designed to be:
- **Non-intrusive**: Tracking is optional and doesn't affect functionality
- **Lightweight**: Minimal performance overhead
- **Extensible**: Easy to add new tracking capabilities

Future enhancements could include:
- Personalized recommendations based on usage patterns
- Automatic configuration suggestions
- Learning from successful experiments
- Context-aware help and documentation

## Example: Learning from Usage

Here's how the system learns from your activities:

```python
from axlearn.common.user_profile import UserProfileManager

manager = UserProfileManager()
profile = manager.get_profile()

# As you work with AXLearn...
profile.track_command("train")
profile.track_interest("nlp", "transformer")
profile.track_model_usage("gpt")

# Over time, the system learns:
top_commands = profile.get_top_commands(limit=5)
# Shows your most used commands

top_interests = profile.get_top_interests(limit=3)
# Shows your primary areas of interest

# This can inform:
# - Personalized documentation
# - Relevant examples
# - Configuration suggestions
# - Tool recommendations
```

## API Reference

### UserProfile Class

**Methods:**
- `track_command(command: str)`: Record command usage
- `track_interest(category: str, subcategory: Optional[str])`: Record interest
- `track_model_usage(model_type: str)`: Record model usage
- `add_favorite_config(config_name: str)`: Add to favorites
- `set_preference(key: str, value: Any)`: Set a preference
- `get_top_interests(limit: int)`: Get top interests by frequency
- `get_top_commands(limit: int)`: Get most used commands
- `get_summary()`: Get profile summary
- `to_dict()`: Export as dictionary

### UserProfileManager Class

**Methods:**
- `load_profile()`: Load profile from disk
- `save_profile(profile)`: Save profile to disk
- `get_profile()`: Get current profile (loading if needed)
- `reset_profile()`: Reset to empty profile
- `delete_profile()`: Delete profile file

### Helper Functions

- `get_user_profile()`: Get the global user profile
- `save_user_profile(profile)`: Save the global user profile
- `get_profile_manager()`: Get the global profile manager
