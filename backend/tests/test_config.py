import os
import pytest
from pathlib import Path

def test_env_file_exists():
    """Test that .env file exists in project root"""
    env_file = Path(__file__).parent.parent.parent / ".env"
    assert env_file.exists(), ".env file must exist in project root"

def test_env_example_exists():
    """Test that .env.example exists in project root"""
    env_example = Path(__file__).parent.parent.parent / ".env.example"
    assert env_example.exists(), ".env.example must exist in project root"

def test_all_env_example_variables_exist_in_env():
    """Test that all variables defined in .env.example are set in .env"""
    env_example = Path(__file__).parent.parent.parent / ".env.example"
    env_file = Path(__file__).parent.parent.parent / ".env"
    
    # Read all variables from .env.example
    example_vars = {}
    with open(env_example, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key = line.split('=')[0]
                example_vars[key] = True
    
    # Read all variables from .env
    env_vars = {}
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key = line.split('=')[0]
                    env_vars[key] = True
    
    # Check that all example variables exist in .env
    missing_vars = []
    for var in example_vars:
        if var not in env_vars:
            missing_vars.append(var)
    
    assert not missing_vars, f"Missing environment variables in .env: {', '.join(missing_vars)}"
