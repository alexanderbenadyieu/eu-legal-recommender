from pathlib import Path

from loguru import logger

from config_manager import ConfigManager
from logging_config import setup_logging


def test_configuration():
    """Test configuration and logging setup."""
    # Setup logging
    setup_logging()
    logger.info("Testing configuration setup...")
    
    try:
        # Initialize config manager
        config_manager = ConfigManager()
        
        # Test accessing different config sections
        scraping_config = config_manager.get_scraping_config()
        storage_config = config_manager.get_storage_config()
        metrics_config = config_manager.get_metrics_config()
        
        # Log some config values
        logger.info(f"Base URL: {scraping_config['base_url']}")
        logger.info(f"Storage base directory: {storage_config['base_dir']}")
        logger.info(f"Metrics enabled: {metrics_config['enabled']}")
        
        # Test directory creation
        base_dir = Path(__file__).parent.parent
        for dir_path in [
            base_dir / storage_config['base_dir'],
            base_dir / storage_config['metrics_dir'],
            base_dir / storage_config['recovery_dir'],
            base_dir / 'logs'
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory created/verified: {dir_path}")
        
        logger.success("Configuration test completed successfully")
        
    except Exception as e:
        logger.error(f"Configuration test failed: {str(e)}")
        raise


if __name__ == "__main__":
    test_configuration()
