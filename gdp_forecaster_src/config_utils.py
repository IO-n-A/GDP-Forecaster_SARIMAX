# gdp_forecaster_src/config_utils.py

import argparse
import logging

logger = logging.getLogger(__name__)

# Initialize the global configuration manager
config_manager = None
CONFIG_AVAILABLE = False
try:
    from config import get_config  # use project configuration manager
    CONFIG_AVAILABLE = True
    logger.info("Configuration system detected: get_config")
except Exception as e:
    CONFIG_AVAILABLE = False
    logger.warning("Configuration system NOT detected: %s - using defaults", e)

def initialize_config():
    """
    Initializes the global configuration manager.
    This function discovers and validates the project's configuration files. If the configuration
    is not available or fails to initialize, it logs the error and proceeds with default settings.
    """
    global config_manager
    if CONFIG_AVAILABLE and config_manager is None:
        try:
            config_manager = get_config()
            validation_errors = config_manager.validate_configuration()
            if validation_errors:
                logger.warning("Configuration validation warnings: %s", validation_errors)
        except Exception as e:
            logger.error("Failed to initialize configuration: %s. Using defaults.", e)
            config_manager = None

def get_config_value(key_path: str, default=None, args=None, cli_param=None):
    """
    Retrieves a configuration value, providing support for command-line overrides.
    The function prioritizes values in the following order:
    1. CLI argument (if provided)
    2. Configuration file
    3. Default value
    """
    # First priority: CLI argument
    if args and cli_param and hasattr(args, cli_param):
        cli_value = getattr(args, cli_param)
        if cli_value is not None:
            return cli_value
    
    # Second priority: Configuration file
    if config_manager:
        try:
            config_value = config_manager.get(key_path, default)
            if config_value is not None:
                return config_value
        except Exception as e:
            logger.debug("Error accessing config key '%s': %s", key_path, e)
    
    # Third priority: Default value
    return default