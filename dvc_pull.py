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
CONFIG_LOG_DIR_KEY = 'log_dir'      # Key for log directory in [General]
CONFIG_LOG_LEVEL_KEY = 'log_level'  # Key for log level in [Logging]

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
    # Get the log level, defaulting to INFO if the string is unrecognized
    log_level = log_levels.get(log_level_str.upper(), logging.INFO)

    # Ensure the log directory exists. If not, create it.
    try:
        os.makedirs(log_dir, exist_ok=True)
        # exist_ok=True prevents an error if the directory already exists
    except OSError as e:
        # If we can't even create the log directory, we have a critical problem.
        # This error occurs before logging is fully set up, so print to stderr as a fallback.
        print(f"FATAL ERROR: Could not create log directory {log_dir}: {e}", file=sys.stderr)
        sys.exit(1) # Cannot continue without a log directory

    # Define the full path for the log file
    log_file_path = os.path.join(log_dir, 'dvc_pull.log') # Using the same log file name

    # Configure the root logger
    # We won't specify stream handlers here as we want output ONLY to the file
    logging.basicConfig(
        level=logging.DEBUG, # Set base level to DEBUG to capture all messages, handler will filter
        # format is applied by the handler
    )

    # Clear any default handlers that might send output to the console
    # This ensures our custom handler is the only one processing messages
    logging.getLogger().handlers = []

    # Create a file handler for writing logs to a file
    # Using RotatingFileHandler to automatically manage log file size and rotate old logs
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=1024 * 1024 * 5, # Max size of each log file (e.g., 5 MB)
        backupCount=5 # Number of backup log files to keep
    )
    file_handler.setLevel(log_level) # Set the level for this handler (read from config)

    # Define the format for log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the configured file handler to the root logger
    logging.getLogger().addHandler(file_handler)

    # Get a logger instance for this specific module (__name__)
    # This is good practice as it allows for more granular logging control later if needed
    return logging.getLogger(__name__)

def read_config(config_file):
    """
    Reads configuration from the specified ini file.
    Returns a dictionary with required config values (log_dir, log_level).
    Exits if the config file or required sections/keys are missing.
    """
    config = configparser.ConfigParser()

    try:
        # Read the configuration file
        config.read(config_file)
    except configparser.Error as e:
        # If reading the config file fails, print a fatal error and exit
        print(f"FATAL ERROR: Could not read config file {config_file}: {e}", file=sys.stderr)
        sys.exit(1)

    config_values = {}
    # Define the required sections and keys for this script (log_dir and log_level)
    required_keys = {
        CONFIG_GENERAL_SECTION: [CONFIG_LOG_DIR_KEY],
        CONFIG_LOGGING_SECTION: [CONFIG_LOG_LEVEL_KEY]
    }

    # Check if all required sections and keys are present in the config
    for section, keys in required_keys.items():
        if not config.has_section(section):
            print(f"FATAL ERROR: Config file missing required section: [{section}]", file=sys.stderr)
            sys.exit(1)
        for key in keys:
            if not config.has_option(section, key):
                 print(f"FATAL ERROR: Config file missing required option '{key}' in section [{section}]", file=sys.stderr)
                 sys.exit(1)
            # Get the value for the key and store it
            config_values[key] = config.get(section, key)

    # Return the dictionary containing the retrieved configuration values
    return config_values

def run_dvc_pull():
    """
    Runs the 'dvc pull' command.
    Logs command execution, output, and errors using the configured logger.
    """
    command = ["dvc", "pull"]
    command_str = ' '.join(command)

    # Log the command being executed
    logger.info(f"Running command: {command_str}")

    try:
        # Execute the command using subprocess.run
        # capture_output=True captures stdout and stderr
        # text=True decodes output as text
        # check=True automatically raises CalledProcessError on non-zero exit status
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True # This will raise an exception if 'dvc pull' fails
        )

        # Log standard output from the command
        if result.stdout:
            logger.info("STDOUT:\n%s", result.stdout.strip()) # Use strip() to avoid leading/trailing whitespace

        # Log standard error from the command
        # DVC often uses stderr for progress/info, but log it as INFO unless check=True caught an error
        if result.stderr:
             # If check=True didn't raise an error, stderr might just be info/progress
             logger.info("STDERR:\n%s", result.stderr.strip())

        # If we reached here, check=True did not raise an exception, meaning the command succeeded
        logger.info(f"Command completed successfully: {command_str}")


    except FileNotFoundError:
        # This occurs if the 'dvc' executable is not found in the system's PATH
        logger.critical(f"Error: Command '{command[0]}' not found. Is DVC installed and in your PATH?")
        sys.exit(1) # Critical error, cannot run DVC

    except subprocess.CalledProcessError as e:
        # This block catches errors where 'dvc pull' was executed but returned a non-zero exit code
        # This indicates a failure in the DVC operation itself (e.g., remote connection error, data not found)
        logger.error(f"Command failed with return code {e.returncode}: {command_str}")
        logger.error(f"STDOUT:\n%s", e.stdout.strip())
        logger.error(f"STDERR:\n%s", e.stderr.strip()) # Log stderr as error when command failed
        sys.exit(e.returncode) # Exit with the same error code as the failed command

    except Exception as e:
        # Catch any other unexpected exceptions
        logger.critical(f"An unexpected error occurred while running command '{command_str}': {e}", exc_info=True) # Log exception details
        sys.exit(1) # Exit with a general error code

# --- Main Execution ---
if __name__ == "__main__":
    # Read configuration from the specified file
    config_settings = read_config(CONFIG_FILE)

    # Get the log directory and level from the configuration
    # Use get() with a default in case read_config somehow missed a required key (though it shouldn't)
    log_directory = config_settings.get(CONFIG_LOG_DIR_KEY, './logs')
    log_level_setting = config_settings.get(CONFIG_LOG_LEVEL_KEY, 'INFO')

    # Set up the global logger instance based on the configuration
    logger = setup_logging(log_directory, log_level_setting)

    # Log startup information
    logger.info("Script started: Running dvc pull.")
    logger.info(f"Logging to directory: {log_directory}")
    logger.info(f"Logging level set to: {log_level_setting}")

    # Run the dvc pull command
    run_dvc_pull()

    # Log successful completion
    print("Script finished successfully.")
