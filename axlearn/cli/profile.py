# Copyright © 2025 Apple Inc.

"""CLI commands for managing user profiles."""

import json
import sys

from absl import app, flags

from axlearn.common.user_profile import UserProfileManager, get_user_profile, save_user_profile


FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "action",
    None,
    ["view", "summary", "reset", "track", "export"],
    "Action to perform: view (detailed profile), summary (brief overview), "
    "reset (clear profile), track (manually track activity), export (JSON output)",
    required=True,
)

flags.DEFINE_string(
    "category",
    None,
    "Category to track (for --action=track). E.g., 'nlp', 'vision', 'speech'",
)

flags.DEFINE_string(
    "subcategory",
    None,
    "Subcategory to track (for --action=track). E.g., 'gpt', 'bert', 'vit'",
)

flags.DEFINE_string(
    "command",
    None,
    "Command to track (for --action=track). E.g., 'train', 'eval'",
)

flags.DEFINE_string(
    "model",
    None,
    "Model type to track (for --action=track). E.g., 'transformer', 'vit'",
)


def _print_summary(profile):
    """Print a summary of the user profile."""
    summary = profile.get_summary()
    
    print("\n=== User Profile Summary ===\n")
    print(f"Profile Version: {summary['profile_version']}")
    print(f"Created: {summary['created_at']}")
    print(f"Last Updated: {summary['last_updated']}")
    
    print(f"\n--- Preferences ---")
    print(f"Preferred Accelerator: {summary['preferred_accelerator'] or 'Not set'}")
    print(f"Typical Scale: {summary['typical_scale'] or 'Not set'}")
    
    print(f"\n--- Interests ({summary['total_interests']} total) ---")
    if summary['top_interests']:
        interest_data = [
            [i['category'], i['frequency']] 
            for i in summary['top_interests']
        ]
        print(tabulate(interest_data, headers=['Category', 'Frequency'], tablefmt='simple'))
    else:
        print("No interests tracked yet.")
    
    print(f"\n--- Commands ({summary['total_commands']} total) ---")
    if summary['top_commands']:
        command_data = [
            [c['command'], c['count']] 
            for c in summary['top_commands']
        ]
        print(tabulate(command_data, headers=['Command', 'Count'], tablefmt='simple'))
    else:
        print("No commands tracked yet.")
    
    print(f"\n--- Favorite Configs ---")
    if summary['favorite_configs']:
        for config in summary['favorite_configs']:
            print(f"  - {config}")
    else:
        print("No favorite configs yet.")
    
    print(f"\n--- Top Models ---")
    if summary['top_models']:
        model_data = [[model, count] for model, count in summary['top_models']]
        print(tabulate(model_data, headers=['Model', 'Usage Count'], tablefmt='simple'))
    else:
        print("No models tracked yet.")
    
    print()


def _print_detailed_view(profile):
    """Print detailed profile information."""
    print("\n=== Detailed User Profile ===\n")
    
    data = profile.to_dict()
    
    print(f"Profile Version: {data['profile_version']}")
    print(f"Created: {data['created_at']}")
    print(f"Last Updated: {data['last_updated']}")
    
    print("\n--- All Interests ---")
    if data['interests']:
        for category, interest in data['interests'].items():
            subcats = ', '.join(interest['subcategories']) if interest['subcategories'] else 'None'
            print(f"  {category}:")
            print(f"    Frequency: {interest['frequency']}")
            print(f"    Subcategories: {subcats}")
            print(f"    Last Accessed: {interest['last_accessed']}")
    else:
        print("No interests tracked yet.")
    
    print("\n--- All Command History ---")
    if data['command_history']:
        cmd_data = [
            [cmd_info['command'], cmd_info['count'], cmd_info['last_used']]
            for cmd_info in data['command_history'].values()
        ]
        cmd_data.sort(key=lambda x: x[1], reverse=True)
        print(tabulate(cmd_data, headers=['Command', 'Count', 'Last Used'], tablefmt='simple'))
    else:
        print("No commands tracked yet.")
    
    print("\n--- Favorite Configurations ---")
    if data['favorite_configs']:
        for config in data['favorite_configs']:
            print(f"  - {config}")
    else:
        print("No favorite configs yet.")
    
    print("\n--- Model Usage ---")
    if data['frequently_used_models']:
        model_data = [
            [model, count] 
            for model, count in sorted(
                data['frequently_used_models'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
        ]
        print(tabulate(model_data, headers=['Model Type', 'Usage Count'], tablefmt='simple'))
    else:
        print("No models tracked yet.")
    
    print("\n--- Preferences ---")
    print(f"Preferred Accelerator: {data['preferred_accelerator'] or 'Not set'}")
    print(f"Typical Scale: {data['typical_scale'] or 'Not set'}")
    print()


def main(_):
    """Main function for profile CLI."""
    try:
        # Handle tabulate import for nice tables
        global tabulate
        try:
            from tabulate import tabulate
        except ImportError:
            # Fallback to simple printing if tabulate not available
            def simple_table(data, headers, tablefmt=None):
                """Simple table fallback."""
                print("  ".join(headers))
                for row in data:
                    print("  ".join(str(cell) for cell in row))
            tabulate = simple_table
        
        manager = UserProfileManager()
        
        if FLAGS.action == "view":
            profile = manager.load_profile()
            _print_detailed_view(profile)
        
        elif FLAGS.action == "summary":
            profile = manager.load_profile()
            _print_summary(profile)
        
        elif FLAGS.action == "reset":
            confirm = input("Are you sure you want to reset your profile? (yes/no): ")
            if confirm.lower() in ["yes", "y"]:
                manager.reset_profile()
                print("Profile has been reset.")
            else:
                print("Reset cancelled.")
        
        elif FLAGS.action == "track":
            profile = manager.get_profile()
            
            if FLAGS.category:
                profile.track_interest(FLAGS.category, FLAGS.subcategory)
                msg = f"Tracked interest: {FLAGS.category}"
                if FLAGS.subcategory:
                    msg += f" > {FLAGS.subcategory}"
                print(msg)
            
            if FLAGS.command:
                profile.track_command(FLAGS.command)
                print(f"Tracked command: {FLAGS.command}")
            
            if FLAGS.model:
                profile.track_model_usage(FLAGS.model)
                print(f"Tracked model: {FLAGS.model}")
            
            if not (FLAGS.category or FLAGS.command or FLAGS.model):
                print("Error: Please specify --category, --command, or --model to track")
                return 1
            
            manager.save_profile()
            print("Profile updated.")
        
        elif FLAGS.action == "export":
            profile = manager.load_profile()
            print(json.dumps(profile.to_dict(), indent=2))
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    app.run(main)
