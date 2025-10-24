"""
Utility Functions Module
Contains helper functions for the project
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


def print_header(text, width=70, char='='):
    """
    Print a formatted header
    
    Args:
        text (str): Header text
        width (int): Width of the header
        char (str): Character to use for border
    """
    print("\n" + char * width)
    print(text.center(width))
    print(char * width)


def print_section(text, width=70, char='-'):
    """
    Print a formatted section header
    
    Args:
        text (str): Section text
        width (int): Width of the section
        char (str): Character to use for border
    """
    print("\n" + char * width)
    print(text)
    print(char * width)


def save_results(results, filename='results.json'):
    """
    Save results to JSON file
    
    Args:
        results (dict): Results dictionary
        filename (str): Output filename
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    filepath = os.path.join('results', filename)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        return obj
    
    # Convert all values
    converted_results = {k: convert_types(v) for k, v in results.items()}
    
    with open(filepath, 'w') as f:
        json.dump(converted_results, f, indent=4, default=str)
    
    print(f"✓ Results saved to '{filepath}'")


def load_results(filename='results.json'):
    """
    Load results from JSON file
    
    Args:
        filename (str): Input filename
    
    Returns:
        dict: Results dictionary
    """
    filepath = os.path.join('results', filename)
    
    if not os.path.exists(filepath):
        print(f"✗ File '{filepath}' not found")
        return None
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    print(f"✓ Results loaded from '{filepath}'")
    return results


def create_project_structure():
    """Create necessary project directories"""
    directories = ['data/raw', 'data/processed', 'results', 'models']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Project structure created")


def get_dataset_info(data):
    """
    Get comprehensive dataset information
    
    Args:
        data: DataFrame
    
    Returns:
        dict: Dataset information
    """
    info = {
        'n_rows': len(data),
        'n_columns': len(data.columns),
        'columns': data.columns.tolist(),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'memory_usage': data.memory_usage(deep=True).sum() / 1024**2,  # MB
    }
    
    return info


def print_dataset_summary(data):
    """
    Print a comprehensive dataset summary
    
    Args:
        data: DataFrame
    """
    print_header("DATASET SUMMARY")
    
    print(f"\nShape: {data.shape[0]} rows × {data.shape[1]} columns")
    print(f"Memory Usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\nData Types:")
    print(data.dtypes.value_counts())
    
    print("\nMissing Values:")
    missing = data.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values ✓")
    else:
        print(missing[missing > 0])
    
    print("\nNumeric Columns Summary:")
    print(data.describe())


def format_metrics(metrics):
    """
    Format metrics dictionary for display
    
    Args:
        metrics (dict): Metrics dictionary
    
    Returns:
        str: Formatted metrics string
    """
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{key}: {value:.4f}")
        else:
            formatted.append(f"{key}: {value}")
    
    return "\n".join(formatted)


def get_timestamp():
    """
    Get current timestamp string
    
    Returns:
        str: Timestamp in format YYYY-MM-DD_HH-MM-SS
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def export_to_csv(data, filename, folder='results'):
    """
    Export DataFrame to CSV
    
    Args:
        data: DataFrame
        filename (str): Output filename
        folder (str): Output folder
    """
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    data.to_csv(filepath, index=False)
    print(f"✓ Data exported to '{filepath}'")


def calculate_percentage(part, total):
    """
    Calculate percentage
    
    Args:
        part: Part value
        total: Total value
    
    Returns:
        float: Percentage
    """
    if total == 0:
        return 0
    return (part / total) * 100


def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """
    Print a progress bar
    
    Args:
        iteration (int): Current iteration
        total (int): Total iterations
        prefix (str): Prefix string
        suffix (str): Suffix string
        length (int): Bar length
        fill (str): Fill character
    """
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()


def validate_data(data, required_columns=None):
    """
    Validate dataset
    
    Args:
        data: DataFrame
        required_columns (list): List of required column names
    
    Returns:
        bool: True if valid, False otherwise
    """
    if data is None or data.empty:
        print("✗ Data is empty or None")
        return False
    
    if required_columns:
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            print(f"✗ Missing required columns: {missing_cols}")
            return False
    
    print("✓ Data validation passed")
    return True


def get_stress_level_name(level):
    """
    Get stress level name from numeric value
    
    Args:
        level (int): Stress level (0, 1, 2)
    
    Returns:
        str: Stress level name
    """
    stress_levels = {
        0: "Low Stress",
        1: "Moderate Stress",
        2: "High Stress"
    }
    return stress_levels.get(level, "Unknown")


def print_menu(options):
    """
    Print a menu with options
    
    Args:
        options (list): List of menu options
    """
    print("\n" + "="*70)
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    print("="*70)


def get_user_choice(min_choice, max_choice):
    """
    Get user choice from menu
    
    Args:
        min_choice (int): Minimum valid choice
        max_choice (int): Maximum valid choice
    
    Returns:
        int: User's choice
    """
    while True:
        try:
            choice = int(input(f"\nEnter your choice ({min_choice}-{max_choice}): "))
            if min_choice <= choice <= max_choice:
                return choice
            else:
                print(f"✗ Please enter a number between {min_choice} and {max_choice}")
        except ValueError:
            print("✗ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n✗ Operation cancelled by user")
            return None


def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def pause():
    """Pause execution and wait for user input"""
    input("\nPress Enter to continue...")
