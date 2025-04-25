#!/usr/bin/env python
"""
Environment setup script for the EU Legal Recommender system.

This script helps set up the environment for the recommender system by:
1. Creating necessary directories
2. Setting up configuration files
3. Checking dependencies
4. Validating the environment
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logger

# Set up logger
logger = setup_logger('setup', log_file='logs/setup.log')


def check_python_version() -> bool:
    """
    Check if the Python version is compatible.
    
    Returns:
        True if compatible, False otherwise
    """
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version >= required_version:
        logger.info(f"Python version {'.'.join(map(str, current_version))} is compatible")
        return True
    else:
        logger.error(
            f"Python version {'.'.join(map(str, current_version))} is not compatible. "
            f"Required: {'.'.join(map(str, required_version))} or higher"
        )
        return False


def check_dependencies() -> Tuple[bool, List[str]]:
    """
    Check if all required dependencies are installed.
    
    Returns:
        Tuple of (success, missing_packages)
    """
    required_packages = [
        'numpy',
        'torch',
        'sentence-transformers',
        'pinecone-client',
        'python-dotenv',
        'scikit-learn',
        'tqdm',
        'PyYAML',
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"Package {package} is installed")
        except ImportError:
            logger.warning(f"Package {package} is not installed")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages


def install_dependencies(packages: List[str]) -> bool:
    """
    Install missing dependencies.
    
    Args:
        packages: List of packages to install
        
    Returns:
        True if successful, False otherwise
    """
    if not packages:
        return True
    
    logger.info(f"Installing missing packages: {', '.join(packages)}")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
        logger.info("All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install packages: {e}")
        return False


def create_directories() -> bool:
    """
    Create necessary directories.
    
    Returns:
        True if successful, False otherwise
    """
    directories = [
        'logs',
        'profiles',
        'data',
        'cache',
    ]
    
    try:
        for directory in directories:
            path = Path(__file__).parent.parent / directory
            path.mkdir(exist_ok=True)
            logger.info(f"Created directory: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        return False


def create_env_file(api_key: Optional[str] = None, environment: Optional[str] = None) -> bool:
    """
    Create or update .env file.
    
    Args:
        api_key: Pinecone API key
        environment: Pinecone environment
        
    Returns:
        True if successful, False otherwise
    """
    env_path = Path(__file__).parent.parent / '.env'
    
    # Read existing .env if it exists
    env_vars = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    env_vars[key] = value
    
    # Update with new values
    if api_key:
        env_vars['PINECONE_API_KEY'] = api_key
    if environment:
        env_vars['PINECONE_ENVIRONMENT'] = environment
    
    # Add default values if not present
    if 'DEVICE' not in env_vars:
        env_vars['DEVICE'] = 'cpu'
    if 'CACHE_DIR' not in env_vars:
        env_vars['CACHE_DIR'] = str(Path(__file__).parent.parent / 'cache')
    
    # Write to .env file
    try:
        with open(env_path, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        logger.info(f"Created/updated .env file at {env_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create .env file: {e}")
        return False


def check_environment() -> Dict[str, bool]:
    """
    Check the environment for compatibility.
    
    Returns:
        Dictionary with check results
    """
    results = {}
    
    # Check Python version
    results['python_version'] = check_python_version()
    
    # Check operating system
    system = platform.system()
    logger.info(f"Operating system: {system}")
    results['operating_system'] = system in ['Linux', 'Darwin', 'Windows']
    
    # Check dependencies
    dependencies_ok, missing_packages = check_dependencies()
    results['dependencies'] = dependencies_ok
    
    # Check for GPU
    try:
        import torch
        results['gpu_available'] = torch.cuda.is_available()
        if results['gpu_available']:
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("No GPU available, using CPU")
    except ImportError:
        results['gpu_available'] = False
        logger.warning("PyTorch not installed, cannot check GPU availability")
    
    return results


def setup_environment(
    install_missing: bool = False,
    api_key: Optional[str] = None,
    environment: Optional[str] = None
) -> bool:
    """
    Set up the environment.
    
    Args:
        install_missing: Whether to install missing dependencies
        api_key: Pinecone API key
        environment: Pinecone environment
        
    Returns:
        True if setup was successful, False otherwise
    """
    logger.info("Starting environment setup")
    
    # Check environment
    env_checks = check_environment()
    if not env_checks['python_version']:
        logger.error("Python version check failed")
        return False
    
    # Create directories
    if not create_directories():
        logger.error("Failed to create directories")
        return False
    
    # Install missing dependencies if requested
    if not env_checks['dependencies'] and install_missing:
        _, missing_packages = check_dependencies()
        if not install_dependencies(missing_packages):
            logger.error("Failed to install dependencies")
            return False
    
    # Create .env file
    if not create_env_file(api_key, environment):
        logger.error("Failed to create .env file")
        return False
    
    logger.info("Environment setup completed successfully")
    return True


def main():
    """Run the setup script from command line."""
    parser = argparse.ArgumentParser(description='Set up the environment for the EU Legal Recommender system')
    parser.add_argument('--install-missing', action='store_true', help='Install missing dependencies')
    parser.add_argument('--api-key', help='Pinecone API key')
    parser.add_argument('--environment', help='Pinecone environment')
    
    args = parser.parse_args()
    
    success = setup_environment(
        install_missing=args.install_missing,
        api_key=args.api_key,
        environment=args.environment
    )
    
    if success:
        print("Environment setup completed successfully")
        sys.exit(0)
    else:
        print("Environment setup failed, see logs for details")
        sys.exit(1)


if __name__ == '__main__':
    main()
