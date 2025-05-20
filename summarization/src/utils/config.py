"""
Centralized Configuration Management for Summarization Module

This module provides a unified interface for loading and accessing configuration
settings from YAML files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Default paths
DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "config" / "summarisation_config.yaml"
DEFAULT_DB_PATH = Path(__file__).parents[3] / "scraper" / "data" / "eurlex.db"
LEGACY_DB_PATH = Path(__file__).parents[2] / "data" / "processed_documents.db"

class ConfigManager:
    """
    Centralized configuration manager for the summarization module.
    
    Handles loading configuration from YAML files and environment variables.
    Provides a unified interface for accessing configuration settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to a YAML configuration file.
                         If not provided, uses the default path or environment variable.
        """
        # Load configuration path from environment variable or use default
        self.config_path = Path(
            config_path or 
            os.environ.get("SUMMARIZATION_CONFIG_PATH") or 
            DEFAULT_CONFIG_PATH
        )
        
        # Load database paths from environment variables or use defaults
        self.db_paths = {
            "consolidated": Path(
                os.environ.get("EURLEX_DB_PATH") or 
                DEFAULT_DB_PATH
            ),
            "legacy": Path(
                os.environ.get("LEGACY_SUMMARY_DB_PATH") or 
                LEGACY_DB_PATH
            )
        }
        
        # Load configuration
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration settings
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Add database paths to configuration
            config["database"] = {
                "paths": {
                    "consolidated": str(self.db_paths["consolidated"]),
                    "legacy": str(self.db_paths["legacy"])
                }
            }
            
            return config
        except Exception as e:
            raise ValueError(f"Error loading configuration from {self.config_path}: {str(e)}")
    
    def get_db_path(self, db_type: str = "consolidated") -> str:
        """
        Get the path to the specified database.
        
        Args:
            db_type: Type of database ('consolidated' or 'legacy')
            
        Returns:
            Path to the database as a string
        """
        if db_type not in self.db_paths:
            raise ValueError(f"Unknown database type: {db_type}")
        
        return str(self.db_paths[db_type])
    
    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration settings.
        
        Args:
            section: Optional section of the configuration to retrieve
            
        Returns:
            Dictionary containing configuration settings
        """
        if section:
            if section not in self.config:
                raise ValueError(f"Configuration section not found: {section}")
            return self.config[section]
        
        return self.config

# Create a singleton instance
config_manager = ConfigManager()

def get_config(section: Optional[str] = None) -> Dict[str, Any]:
    """
    Get configuration settings.
    
    Args:
        section: Optional section of the configuration to retrieve
        
    Returns:
        Dictionary containing configuration settings
    """
    return config_manager.get_config(section)

def get_db_path(db_type: str = "consolidated") -> str:
    """
    Get the path to the specified database.
    
    Args:
        db_type: Type of database ('consolidated' or 'legacy')
        
    Returns:
        Path to the database as a string
    """
    return config_manager.get_db_path(db_type)

def initialize_config(config_path: Optional[str] = None) -> None:
    """
    Initialize the configuration manager with a specific configuration file.
    
    Args:
        config_path: Path to a YAML configuration file
    """
    global config_manager
    config_manager = ConfigManager(config_path)
