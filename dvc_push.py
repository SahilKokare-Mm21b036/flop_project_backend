import subprocess
import sys
import configparser
import logging
import os
from logging.handlers import RotatingFileHandler # Good for managing log file size

# --- Configuration ---
CONFIG_FILE = 'config.ini'
# Define the sections and keys we need from the config file
CONFIG_GENERAL_SECTION = 'General'
CONFIG_LOGGING_SECTION = 'Logging'
CONFIG_DATA_DIR_KEY = 'data_dir'
CONFIG_LOG_DIR_KEY = 'log_dir'
CONFIG_LOG_LEVEL_KEY = 'log_level'

# --- Logging Setup ---
# Placeholder for logger, will be configured after reading config
logger = None

def setup_logging(log_dir, log_level_str='INFO'):
    """
    Configures the root logger to write to a file in the specified directory.
    Creates the log directory if it doesn't exist.
    """
    # Map string log levels from config to logging module constants
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level = log_levels.get(log_level_str.upper(), logging.INFO) # Default to INFO

    # Ensure the log directory exists
    try:
        os.makedirs(log_dir, exist_ok=True)
        # Use exist_ok=True to prevent an error if the directory already exists
    except OSError as e:
        # If we can't create the log directory, we have a serious problem.
        # Fallback to printing an error, as logging setup hasn't finished.
        print(f"FATAL ERROR: Could not create log directory {log_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    # Define the log file path
    log_file_path = os.path.join(log_dir, 'dvc_push.log')

    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        # We will add handlers manually to control where output goes
    )

    # Prevent logs from propagating to the default console handler if one exists
    # This helps ensure output only goes to our file handler
    logging.getLogger().handlers = []

    # Create a file handler for logging to a file
    # Use RotatingFileHandler to prevent log files from growing indefinitely
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=1024 * 1024 * 5, # 5 MB per log file
        backupCount=5 # Keep up to 5 backup files
    )
    file_handler.setLevel(log_level) # Set handler level
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the root logger
    logging.getLogger().addHandler(file_handler)

    # Return the configured logger instance
    return logging.getLogger(__name__) # Use __name__ for a logger specific to this module

def read_config(config_file):
    """
    Reads configuration from the specified ini file.
    Returns a dictionary with required config values.
    Exits if the config file or required sections/keys are missing.
    """
    config = configparser.ConfigParser()

    try:
        config.read(config_file)
    except configparser.Error as e:
        print(f"FATAL ERROR: Could not read config file {config_file}: {e}", file=sys.stderr)
        sys.exit(1)

    config_values = {}
    required_keys = {
        CONFIG_GENERAL_SECTION: [CONFIG_DATA_DIR_KEY, CONFIG_LOG_DIR_KEY],
        CONFIG_LOGGING_SECTION: [CONFIG_LOG_LEVEL_KEY]
    }

    for section, keys in required_keys.items():
        if not config.has_section(section):
            print(f"FATAL ERROR: Config file missing section: [{section}]", file=sys.stderr)
            sys.exit(1)
        for key in keys:
            if not config.has_option(section, key):
                 print(f"FATAL ERROR: Config file missing option '{key}' in section [{section}]", file=sys.stderr)
                 sys.exit(1)
            config_values[key] = config.get(section, key)

    return config_values

def run_dvc_commands(data_dir):
    """
    Runs dvc add and dvc push commands using the specified data directory.
    Logs command output and errors.
    """
    commands = [
        ["dvc", "add", data_dir], # Use the data_dir read from config
        ["dvc", "push"]
    ]

    for command in commands:
        command_str = ' '.join(command)
        logger.info(f"Running command: {command_str}")

        try:
            # Run the command
            # Using Popen allows us to capture stdout and stderr separately
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # communicate() waits for the process to terminate and returns stdout/stderr
            stdout, stderr = process.communicate()

            # Log output
            if stdout:
                logger.info("STDOUT:\n%s", stdout.decode('utf-8'))
            if stderr:
                logger.error("STDERR:\n%s", stderr.decode('utf-8')) # stderr usually indicates errors

            # Check for errors using the return code
            if process.returncode != 0:
                logger.error(f"Command failed with return code {process.returncode}: {command_str}")
                # Exit early if a DVC command fails, as subsequent commands might depend on it
                sys.exit(1)
            else:
                logger.info(f"Command completed successfully: {command_str}")


        except FileNotFoundError:
            logger.critical(f"Error: Command '{command[0]}' not found. Is DVC installed and in your PATH?")
            sys.exit(1) # Cannot proceed if dvc command is not found
        except Exception as e:
            logger.critical(f"An unexpected error occurred while running command '{command_str}': {e}", exc_info=True) # Log exception details
            sys.exit(1)

# --- Main Execution ---
if __name__ == "__main__":
    # Read configuration first
    config_settings = read_config(CONFIG_FILE)

    # Setup logging based on config
    # We need the log_dir before setting up logging
    log_directory = config_settings.get(CONFIG_LOG_DIR_KEY, './logs') # Use default if somehow missed
    log_level_setting = config_settings.get(CONFIG_LOG_LEVEL_KEY, 'INFO') # Use default if somehow missed
    logger = setup_logging(log_directory, log_level_setting)

    logger.info("Script started.")
    logger.info(f"Reading data directory from config: {config_settings.get(CONFIG_DATA_DIR_KEY)}")
    logger.info(f"Logging to directory: {log_directory}")
    logger.info(f"Logging level set to: {log_level_setting}")

    # Get the data directory path from the config
    data_directory_path = config_settings.get(CONFIG_DATA_DIR_KEY)

    # Run the DVC commands
    # Pass the data directory path to the function
    run_dvc_commands(data_directory_path)

    print("Script finished successfully.")
