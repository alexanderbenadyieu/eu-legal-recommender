"""
Configuration Management Module for EUR-Lex Web Scraper

This module provides a robust configuration management system for the 
EUR-Lex web scraper. It handles loading, parsing, and accessing 
configuration settings from a YAML file, with support for dynamic 
configuration updates and error handling.

Key Features:
- YAML-based configuration loading
- Flexible configuration path specification
- Logging of configuration loading events
- Secure configuration parsing
- Modular configuration access methods
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger


class ConfigManager:
    """
    A comprehensive configuration management class for the EUR-Lex web scraper.

    Manages loading, parsing, and accessing configuration settings from a YAML file.
    Provides methods to retrieve specific configuration sections and update 
    configurations dynamically.

    Attributes:
        config_path (Path): Path to the configuration YAML file
        config (Dict[str, Any]): Loaded configuration dictionary

    Raises:
        Exception: If configuration loading or parsing fails
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the configuration manager with optional custom configuration path.

        Args:
            config_path (Optional[Path], optional): 
                Custom path to the configuration file. 
                Defaults to 'config/config.yaml' in the project root.

        Notes:
            - If no path is provided, uses a default path relative to the project structure
            - Automatically loads configuration upon initialization
            - Logs successful configuration loading or any loading errors
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
            
        self.config_path = config_path
        self.config: Dict[str, Any] = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load and parse configuration from the specified YAML file.

        Reads the configuration file using safe YAML loading to prevent 
        potential code execution vulnerabilities.

        Returns:
            Dict[str, Any]: Parsed configuration dictionary

        Raises:
            Exception: If file reading or parsing fails
        """
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def get_scraping_config(self) -> Dict[str, Any]:
        """
        Retrieve scraping-related configuration settings.

        Returns configuration parameters specific to web scraping, 
        such as base URLs, request timeouts, and language settings.

        Returns:
            Dict[str, Any]: Scraping configuration dictionary
        """
        return self.config["scraping"]
    
    def get_storage_config(self) -> Dict[str, Any]:
        """
        Retrieve storage-related configuration settings.

        Returns configuration parameters for data storage, including 
        base directories, file formats, and storage strategies.

        Returns:
            Dict[str, Any]: Storage configuration dictionary
        """
        return self.config["storage"]
    
    def get_metrics_config(self) -> Dict[str, Any]:
        """
        Retrieve metrics-related configuration settings.

        Returns configuration for metrics collection and export, 
        with a fallback to an empty dictionary if no metrics config exists.

        Returns:
            Dict[str, Any]: Metrics configuration dictionary
        """
        return self.config.get("metrics", {})
    
    def get_validation_config(self) -> Dict[str, Any]:
        """
        Retrieve validation-related configuration settings.

        Returns configuration parameters for document metadata validation, 
        including schema definitions and validation rules.

        Returns:
            Dict[str, Any]: Validation configuration dictionary
        """
        return self.config["validation"]
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Dynamically update the current configuration.

        Allows runtime modification of configuration settings, 
        with optional validation and persistence mechanisms.

        Args:
            new_config (Dict[str, Any]): Dictionary of configuration updates

        Notes:
            - Performs shallow merge of configuration dictionaries
            - Does not persist changes to the original YAML file
            - Recommended for runtime configuration adjustments
        """
        self.config.update(new_config)
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info("Configuration updated successfully")
        except Exception as e:
            logger.error(f"Failed to update configuration: {str(e)}")
            raise
