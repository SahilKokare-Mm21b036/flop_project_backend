import asyncio
import os
import pickle
import configparser
import re
import logging
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright
import subprocess
import sys

# --- Combined Imports ---
# Imports from init_scraper_module.txt:
# import subprocess
# import logging
# import configparser
# import pathlib # Used via Path
# import sys

# Imports from image_scraper_v2_2.txt:
# import asyncio
# import os
# import pickle
# import configparser
# import re
# import logging
# from datetime import datetime
# from pathlib import Path
# from playwright.async_api import async_playwright

# All necessary imports are now included

# --- Configuration File (Common) ---
CONFIG_FILE = 'config.ini'

# --- State File Configuration (from image_scraper_v2_2.txt) ---
STATE_FILE_NAME = "scroll_state.pkl"

# --- Log File Names (from both scripts, separated for clarity) ---
INIT_LOG_FILE_NAME = 'init_module.log' # For the initial setup/playwright install
SCRAPER_LOG_FILE_NAME = "image_scraper.log" # For the main scraping process

# --- Flag File for First Run Detection ---
PLAYWRIGHT_INSTALLED_FLAG = ".playwright_installed_flag"


# --- Helper Functions (from image_scraper_v2_2.txt) ---

def sanitize_name(name):
    """Sanitizes a string to be safe for use in a filename or directory name."""
    name = name.replace(" ", "_")
    # Remove characters that are not alphanumeric, underscores, hyphens, or periods
    name = re.sub(r'[^\w.-]', '', name)
    # Remove leading/trailing underscores or hyphens
    name = name.strip('_-')
    # Ensure it's not empty after sanitizing
    if not name:
        name = "untitled"
    return name

# NOTE: Keeping the load_config and setup_logging from image_scraper_v2_2.txt
# as they are more comprehensive for the scraping process.
# The initial setup phase will perform a basic config read just for its logging.

def load_config(config_path=CONFIG_FILE):
    """Loads configuration from the config.ini file."""
    config = configparser.ConfigParser()
    if not config.read(config_path):
        # Note: This load_config is used by the main scraping logic.
        # Initial config loading for the init part is handled separately in __main__.
        # Use root logger here as scraper logger might not be set up yet
        logging.error(f"Configuration file not found at {config_path}")
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    # --- Parse General Settings ---
    tmp_dir = Path(config.get('General', 'tmp_dir', fallback='./tmp')).resolve()
    data_dir = Path(config.get('General', 'data_dir', fallback='./data')).resolve()
    log_dir = Path(config.get('General', 'log_dir', fallback='./logs')).resolve()

    # --- Parse Scrolling Settings ---
    scroll_increment_factor = config.getfloat('Scrolling', 'scroll_increment_factor', fallback=0.9)
    min_image_size_bytes = config.getint('Scrolling', 'min_image_size_bytes', fallback=1024)

    # --- Parse Image Types ---
    mime_types_str = config.get('ImageTypes', 'mime_types', fallback='image/jpeg, image/png').replace(" ", "")
    image_mime_types = [m.strip() for m in mime_types_str.split(',') if m.strip()]

    # --- Parse Logging Settings ---
    log_level_str = config.get('Logging', 'log_level', fallback='INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # --- Parse Websites ---
    websites = dict(config.items('Websites'))
    if not websites:
        logging.warning("Warning: No websites found in the [Websites] section of config.ini.")

    return {
        'tmp_dir': tmp_dir,
        'data_dir': data_dir,
        'log_dir': log_dir,
        'state_file_path': tmp_dir / STATE_FILE_NAME, # Path object for state file
        'scroll_increment_factor': scroll_increment_factor,
        'scroll_wait_time_s': 1, # Hardcoded wait time in seconds as per image_scraper_v2_2.txt
        'min_image_size_bytes': min_image_size_bytes,
        'image_mime_types': image_mime_types,
        'log_level': log_level,
        'websites': websites
    }

def setup_logging(log_dir, log_level, log_file_name):
    """
    Sets up logging to a specific file and disables console output.
    Log level from config controls the minimum level that gets processed.
    The file handler captures DEBUG and above.
    Takes a log_file_name parameter to distinguish between init and scraper logs.
    """
    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / log_file_name

    # Get a logger instance (using the root logger name '')
    # We will configure the root logger here. The init phase will configure it first,
    # and the scraper phase will reconfigure it.
    logger = logging.getLogger('')
    # Prevent adding handlers multiple times if called repeatedly
    if logger.handlers:
         # Clear existing handlers from root logger to ensure only the desired file handler is active
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close() # Close the handler to free up resources

    # Set the logger's level based on config - messages below this level are ignored
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create file handler - Use 'a' mode for append
    file_handler = logging.FileHandler(str(log_file_path), mode='a') # Use str() for compatibility
    # Set handler level to DEBUG to capture all messages that pass the logger's level filter
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # NO Console Handler is added explicitly here. Logging will go to the file.

    # Log a startup message (this will only go to the file)
    logger.info("-" * 30)
    logger.info(f"Script module ({log_file_name}) started.")
    logger.info(f"Log file: {log_file_path}")
    logger.info(f"Effective logger level: {logging.getLevelName(logger.level)}")
    logger.info("-" * 30)

    # No need to return logger as we configured the root logger

def load_state(state_file_path: Path):
    """Loads the persistent state (scroll counts, execution number, downloaded URLs) from a pickle file."""
    # Added 'downloaded_image_urls' to the initial state
    initial_state = {'execution_counter': 0, 'url_scroll_state': {}, 'downloaded_image_urls': set()}
    if state_file_path.exists():
        try:
            with open(state_file_path, 'rb') as f:
                state = pickle.load(f)
            # Basic validation - check for required keys
            if not isinstance(state, dict) or 'execution_counter' not in state or 'url_scroll_state' not in state:
                logging.warning(f"Invalid state data found in {state_file_path}. Starting fresh.")
                return initial_state
            # Ensure url_scroll_state is a dict
            if not isinstance(state.get('url_scroll_state'), dict):
                logging.warning(f"Invalid url_scroll_state format in {state_file_path}. Resetting scroll state.")
                state['url_scroll_state'] = {}
            # Ensure downloaded_image_urls is a set
            if not isinstance(state.get('downloaded_image_urls'), set):
                logging.warning(f"Invalid downloaded_image_urls format in {state_file_path}. Resetting unique URL state.")
                state['downloaded_image_urls'] = set()

            logging.info(f"State loaded from {state_file_path}")
            logging.info(f"Currently tracking {len(state.get('downloaded_image_urls', set()))} previously downloaded unique image URLs.")
            return state
        except (EOFError, pickle.UnpicklingError, FileNotFoundError, AttributeError, Exception) as e: # Added generic Exception catch
            logging.error(f"Error loading state from {state_file_path} ({e}). Starting fresh.")
            return initial_state
    else:
        logging.info(f"State file not found at {state_file_path}. Starting fresh.")
        return initial_state

def save_state(state, state_file_path: Path):
    """Saves the current state to a pickle file."""
    try:
        # Ensure the directory exists
        state_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(state_file_path, 'wb') as f:
            pickle.dump(state, f)
        logging.info(f"State saved to {state_file_path}. Total unique images tracked: {len(state.get('downloaded_image_urls', set()))}")
    except Exception as e:
        logging.error(f"Error saving state to {state_file_path}: {e}")

# --- Core Download Function (Enhanced for State and Scrolling and Uniqueness) ---

# Pass the full state dictionary to process_website
async def process_website(page, config, state, execution_data_dir: Path, website_name: str, website_url: str, start_scroll_count: int):
    """
    Navigates to a website, scrolls step-by-step to the previous position,
    then scrolls 4 additional steps to trigger lazy loading, and downloads *unique* images.
    Starts scrolling from start_scroll_count and performs 4 new scroll increments.
    Saves images to a subdirectory within the execution directory.
    Returns the final total scroll count achieved for this URL (start_count + new scrolls performed).
    Updates the persistent state with downloaded unique URLs.
    """
    # Access the state object directly - it's a parameter, no need for nonlocal

    scroll_increment_factor = config['scroll_increment_factor']
    scroll_wait_time_s = config['scroll_wait_time_s'] # Use the hardcoded wait time
    image_mime_types = config['image_mime_types']
    min_image_size_bytes = config['min_image_size_bytes']

    # Construct the save directory: execution_data_dir / sanitized_website_name
    sanitized_website_name = sanitize_name(website_name)
    save_dir = execution_data_dir / sanitized_website_name
    save_dir.mkdir(parents=True, exist_ok=True) # Ensure this website-specific directory exists

    logging.info(f"\n--- Processing Website: '{website_name}' ({website_url}) ---")
    logging.info(f"Saving images to: {save_dir}")
    logging.info(f"Previously completed scroll increments: {start_scroll_count}")
    logging.info(f"Will perform 4 new scroll increments this execution.")
    logging.info(f"Currently tracking {len(state.get('downloaded_image_urls', set()))} unique image URLs across all executions.")


    downloaded_count_this_run = 0 # Count of images downloaded *in this specific run* for this website
    # Use a set scoped to this function call to track URLs processed *in this run* for this website.
    # This is separate from the persistent state and prevents redundant checks/work within one site's processing.
    seen_urls_this_run = set()

    # --- The Request Interceptor ---
    # This asynchronous function will be called by Playwright for each completed request
    async def handle_request(request):
        # Access downloaded_count_this_run and seen_urls_this_run from the outer function's scope
        nonlocal downloaded_count_this_run
        nonlocal seen_urls_this_run
        # Access state parameter directly - it's visible in the outer scope

        resource_url = request.url
        resource_type = request.resource_type

        # Primary uniqueness check: Is this URL already in our persistent state?
        if resource_type == 'image' and resource_url not in state.get('downloaded_image_urls', set()):
            # Secondary check: Have we processed this URL already in *this run* for *this website*?
            if resource_url not in seen_urls_this_run:
                seen_urls_this_run.add(resource_url) # Mark this URL as seen for this run

                try:
                    response = await request.response()
                    if not response:
                        logging.debug(f"  No response for {resource_url}")
                        return

                    content_type = response.headers.get('content-type', '')
                    image_data = await response.body()

                    # Check if the content type is one we want and the size is sufficient
                    if any(mime in content_type for mime in image_mime_types) and len(image_data) >= min_image_size_bytes:
                        # Create a safe filename
                        filename = os.path.basename(resource_url)
                        if '?' in filename:
                            filename = filename.split('?')[0] # Remove query parameters
                        filename = sanitize_name(filename) # Sanitize filename just in case URL path is weird

                        # Construct the full save path
                        save_path = save_dir / f"{downloaded_count_this_run:04d}_{filename}"

                        # Final check before saving: ensure the file doesn't already exist
                        if not save_path.exists():
                            logging.info(f"  Downloading NEW unique image: {resource_url} to {save_path.name}")
                            try:
                                with open(save_path, 'wb') as f:
                                    f.write(image_data)
                                downloaded_count_this_run += 1 # Increment count for this run

                                # --- IMPORTANT: Add the URL to the persistent state ---
                                # Access state directly here
                                state.get('downloaded_image_urls', set()).add(resource_url)
                                logging.debug(f"  Added {resource_url} to persistent unique URL list.")

                            except IOError as io_e:
                                logging.error(f"  Error saving file {save_path}: {io_e}")


                except Exception as e:
                    # Catch potential errors during response processing
                    logging.debug(f"  Error processing image request {resource_url}: {e}")


    # --- Setup Request Interception ---
    page.on("requestfinished", handle_request)

    # --- Navigate to the page ---
    try:
        logging.info("Going to page and waiting for initial load...")
        await page.goto(website_url, wait_until="networkidle", timeout=60000)
        logging.info("Initial load complete.")
    except Exception as e:
        logging.error(f"Error navigating to {website_url}: {e}")
        return start_scroll_count

    # --- Perform Step-by-Step Scrolling ---
    logging.info(f"Starting step-by-step scrolling process...")

    scrolls_performed_in_this_run = 0 # Counter for the new scrolls actually performed in this run

    try:
        viewport_height = await page.evaluate("window.innerHeight") or (page.viewport_size['height'] if page.viewport_size else 800)
        scroll_step_amount = int(viewport_height * scroll_increment_factor)
        if scroll_step_amount <= 0: scroll_step_amount = viewport_height

        current_scroll_position_px = 0 # Assume starting at top after goto

        # --- Scroll step-by-step TO the bookmark (previous total scroll position) ---
        if start_scroll_count > 0:
            logging.info(f"Scrolling step-by-step to reach the position corresponding to {start_scroll_count} previous increments...")
            for i in range(start_scroll_count):
                # Wait *before* scrolling to the bookmark as requested
                logging.debug(f"  Waiting {scroll_wait_time_s}s before scroll step {i+1}/{start_scroll_count} (reaching bookmark)...")
                await asyncio.sleep(scroll_wait_time_s)

                current_scroll_position_px = (i + 1) * scroll_step_amount
                logging.debug(f"  Scrolling to {current_scroll_position_px}px (bookmark step {i+1})...")
                await page.evaluate(f"window.scrollTo(0, {current_scroll_position_px});")

                # Check if reached bottom during this initial scroll phase
                try:
                    actual_current_scroll = await page.evaluate("window.pageYOffset")
                    page_height = await page.evaluate("document.body.scrollHeight")
                    viewport_height_after_scroll = await page.evaluate("window.innerHeight") or viewport_height
                    if actual_current_scroll + viewport_height_after_scroll >= page_height - 10:
                        logging.info(f"  Reached bottom while scrolling towards bookmark. Cannot perform new scrolls.")
                        scrolls_performed_in_this_run = 0 # No *new* scrolls performed
                        break # Exit the bookmark scrolling loop
                except Exception as e:
                    logging.debug(f"  Error checking scroll position during bookmark scroll {i+1}: {e}. Continuing.")

        # --- Perform the NEXT 4 scrolls (step-by-step) ---
        scrolls_to_perform_this_run = 4 # Hardcoded number of *new* scrolls as requested
        can_perform_new_scrolls = True
        # Check if we hit bottom in the previous loop before starting new scrolls
        try:
            actual_current_scroll_at_bookmark_end = await page.evaluate("window.pageYOffset")
            page_height_at_bookmark_end = await page.evaluate("document.body.scrollHeight")
            viewport_height_at_bookmark_end = await page.evaluate("window.innerHeight") or viewport_height

            if actual_current_scroll_at_bookmark_end + viewport_height_at_bookmark_end >= page_height_at_bookmark_end - 10:
                logging.info("Already at the bottom after reaching the bookmark position. No new scrolls performed.")
                can_perform_new_scrolls = False

        except Exception as check_bottom_e:
            logging.debug(f"  Error checking bottom status before new scrolls: {check_bottom_e}. Assuming not at bottom.")
            can_perform_new_scrolls = True # Assume we can continue if check fails

        if can_perform_new_scrolls:
            logging.info(f"Performing the next {scrolls_to_perform_this_run} *new* scroll increments...")
            # Start these 4 scrolls from the position reached by the bookmark scrolls
            # Use the actual pageYOffset after reaching the bookmark position as the base for new scrolls
            start_pos_for_new_scrolls_px = await page.evaluate("window.pageYOffset") # Get actual position after phase 1

            for i in range(scrolls_to_perform_this_run):
                # Calculate the target absolute position for the next step relative to the start of NEW scrolls
                next_scroll_pos_px = start_pos_for_new_scrolls_px + (i + 1) * scroll_step_amount

                logging.debug(f"  Scrolling to {next_scroll_pos_px}px (new scroll step {i+1})...")
                await page.evaluate(f"window.scrollTo(0, {next_scroll_pos_px});")
                current_scroll_position_px = next_scroll_pos_px # Update tracker

                # Wait *after* scrolling for the new scrolls
                logging.debug(f"  Waiting {scroll_wait_time_s}s after new scroll step {i+1}/{scrolls_to_perform_this_run}...")
                await asyncio.sleep(scroll_wait_time_s)

                scrolls_performed_in_this_run += 1 # Count this NEW scroll attempt

                try:
                    actual_current_scroll = await page.evaluate("window.pageYOffset")
                    page_height = await page.evaluate("document.body.scrollHeight")
                    viewport_height_after_scroll = await page.evaluate("window.innerHeight") or viewport_height
                    if actual_current_scroll + viewport_height_after_scroll >= page_height - 10:
                        logging.info(f"  Reached bottom of page during new scrolls. Stopping scroll early.")
                        break # Exit the new scrolling loop
                except Exception as e:
                    logging.debug(f"  Error checking scroll position during new scroll {i+1}: {e}. Continuing scrolling.")


    except Exception as general_scroll_e:
        # Catch any unhandled errors during the entire scrolling try block
        logging.error(f"An unhandled error occurred during the scrolling process for {website_name}: {general_scroll_e}", exc_info=True)
        # scrolls_performed_in_this_run will hold the count up to the error point if it occurred in the second loop.
        pass # Continue to final logging and return


    logging.info(f"Finished step-by-step scrolling process for '{website_name}'. Performed {scrolls_performed_in_this_run} *new* scrolls in this run.")

    # Final wait to capture any last-minute requests triggered by the scrolling
    logging.info("Giving network a moment to settle after scrolling...")
    await page.wait_for_load_state('networkidle', timeout=60000)

    # Calculate the total scrolls for this URL across all runs
    # This is the starting point PLUS the number of *new* scrolls successfully performed in this run.
    final_total_scrolls = start_scroll_count + scrolls_performed_in_this_run
    logging.info(f"Finished processing '{website_name}'. Downloaded {downloaded_count_this_run} NEW unique images in this run. Final total scroll count for URL: {final_total_scrolls}.")

    return final_total_scrolls # Return the *total* scrolls performed for this URL


# --- Main Execution Function (from image_scraper_v2_2.txt) ---
async def main():
    """Main function to load config, state, setup logging, and process websites."""
    # This main function is called *after* the initial setup and playwright install.
    # It loads config, sets up its own logging, and proceeds with scraping.

    config = None
    try:
        # Use the load_config defined in this script (from image_scraper_v2_2.txt)
        config = load_config()
    except FileNotFoundError as e:
        # Critical error if config is missing at this stage
        logging.critical(f"Error: {e}\nPlease ensure config.ini exists.")
        return # Exit if config loading fails

    try:
        # Setup logging specifically for the scraper part
        # This will get the root logger and configure it for the scraper log file.
        # Any prior handlers (like the init logger's file handler) will be removed.
        setup_logging(config['log_dir'], config['log_level'], SCRAPER_LOG_FILE_NAME)
        # Messages will now go to image_scraper.log.
    except Exception as e:
        # Critical error if scraper logging cannot be set up
        # Print to stderr as logging setup failed
        print(f"Critical Error: Failed to set up scraper logging to directory {config['log_dir']}: {e}", file=sys.stderr)
        return # Exit if logging setup fails

    # --- Load Persistent State ---
    state = load_state(config['state_file_path']) # Load the state, including unique URLs

    # --- Increment Execution Counter and Setup Directories ---
    state['execution_counter'] += 1
    config['current_execution_number'] = state['execution_counter'] # Store in config

    logging.info(f"\n--- Starting Scraper Execution Number {state['execution_counter']} ---")

    try:
        config['tmp_dir'].mkdir(parents=True, exist_ok=True)
        config['data_dir'].mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Error creating tmp or data directories: {e}")
        # Attempt to save state before exiting
        save_state(state, config['state_file_path'])
        return

    execution_data_dir = config['data_dir'] / f"{config['current_execution_number']:04d}"
    try:
        execution_data_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving data for this execution under: {execution_data_dir}")
    except Exception as e:
        logging.error(f"Error creating execution data directory {execution_data_dir}: {e}")
        # Attempt to save state before exiting
        save_state(state, config['state_file_path'])
        return

    # --- Process Each Website ---
    if not config['websites']:
        logging.info("No websites configured to process. Exiting.")
        # Save state before exiting
        save_state(state, config['state_file_path'])
        return

    browser = None
    try:
        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch(headless=True) # Keep headless true usually
                logging.info("Browser launched.")
            except Exception as e:
                logging.error(f"Failed to launch browser: {e}", exc_info=True)
                return # Exit if browser fails to launch

            # --- Loop through websites with per-website error handling ---
            for website_name, website_url in config['websites'].items():
                logging.info(f"Attempting to process website: {website_name} ({website_url})")
                start_scroll_count = state['url_scroll_state'].get(website_url, 0)
                page = None

                try: # Inner try block for processing a single website
                    page = await browser.new_page()
                    logging.debug(f"New page created for {website_name}.")

                    # Call process_website and get its final scroll count
                    # Pass the state dictionary to process_website so it can update the unique URL set
                    final_total_scrolls_for_url = await process_website(
                        page,
                        config,
                        state, # Pass the state here
                        execution_data_dir,
                        website_name,
                        website_url,
                        start_scroll_count
                    )

                    # Update the state with the returned final count for this URL
                    state['url_scroll_state'][website_url] = final_total_scrolls_for_url

                    logging.info(f"Successfully completed processing and state update for '{website_name}'.")

                except Exception as website_e:
                    logging.error(f"An error occurred while processing {website_name} ({website_url}): {website_e}", exc_info=True)
                    # State for this URL remains the 'start_scroll_count' if an unhandled error occurred during processing.
                    # The unique URLs downloaded *before* the error are still in the state because process_website modifies it directly.
                finally:
                    if page:
                        try:
                            await page.close()
                        except Exception as close_page_e:
                            logging.error(f"Error closing page for {website_name}: {close_page_e}")
                            pass # Continue even if page close fails

            logging.info("Finished iterating through all configured websites.")

    except Exception as browser_e:
        logging.error(f"An unhandled error occurred during browser operations: {browser_e}", exc_info=True)

    finally:
        if browser:
            try:
                await browser.close()
                logging.info("Browser closed.")
            except Exception as close_e:
                logging.error(f"Error closing browser: {close_e}")

    # --- Save Persistent State ---
    # Save the state after all websites have been attempted and the browser is closed
    # This saves the updated scroll counts AND the accumulated set of downloaded unique URLs
    save_state(state, config['state_file_path'])

    logging.info("\nScript finished.")


# --- Combined Entry Point ---
if __name__ == "__main__":
    # This block now combines the logic of both original scripts' __main__ blocks

    # --- Phase 1: Initialization and Playwright Install (from init_scraper_module.txt, modified) ---

    # Initial check for config file using print before any config loading/logging setup
    if not Path(CONFIG_FILE).exists():
        print(f"Error: {CONFIG_FILE} not found. Please create it based on the example.", file=sys.stderr)
        sys.exit(1) # Exit immediately if config is missing

    # Basic config read just to get log directory and level for the init phase logger
    # and the temp directory for the flag file.
    init_config = configparser.ConfigParser()
    log_directory_str = None
    tmp_directory_str = None
    init_log_level = logging.INFO # Default level

    try:
        init_config.read(CONFIG_FILE)
        log_directory_str = init_config.get('General', 'log_dir')
        tmp_directory_str = init_config.get('General', 'tmp_dir', fallback='./tmp') # Get tmp_dir for flag file
        log_level_str = init_config.get('Logging', 'log_level', fallback='INFO').upper()
        init_log_level = getattr(logging, log_level_str, logging.INFO)

        # Construct the full path to the init log file and ensure directory exists
        log_directory_path = Path(log_directory_str)
        log_directory_path.mkdir(parents=True, exist_ok=True)

        # Construct the full path to the temp directory and the flag file within it
        tmp_directory_path = Path(tmp_directory_str).resolve()
        tmp_directory_path.mkdir(parents=True, exist_ok=True)
        playwright_flag_file_path = tmp_directory_path / PLAYWRIGHT_INSTALLED_FLAG


        # Setup logging for the initialization phase using the root logger
        setup_logging(log_directory_path, init_log_level, INIT_LOG_FILE_NAME)

        logging.info(f"Configuration read successfully for initialization phase.")
        logging.info(f"Base log directory: '{log_directory_str}'")
        logging.info(f"Temp directory: '{tmp_directory_str}'")
        logging.info(f"Log level: '{log_level_str}' (Effective level: {init_log_level})")
        logging.info(f"Initialization logging output directed to file: '{log_directory_path / INIT_LOG_FILE_NAME}'")
        logging.info(f"Playwright installation flag file: '{playwright_flag_file_path}'")


    except FileNotFoundError:
        # Should be caught by the initial check, but included for robustness
        print(f"FATAL ERROR: Configuration file not found: {CONFIG_FILE}.", file=sys.stderr)
        sys.exit(1)
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        print(f"FATAL ERROR: Missing config section or option in {CONFIG_FILE}: {e}.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR: An unexpected error occurred during initial config reading or log setup. Error: {e}", file=sys.stderr)
        sys.exit(1)


    # --- Execute Playwright Install Command (Conditionally) ---
    # Use the root logger configured above for this phase

    if playwright_flag_file_path.exists():
        logging.info("Playwright installation flag file found. Skipping browser installation.")
    else:
        logging.info("Playwright installation flag file NOT found. Attempting browser installation...")
        install_command = ["playwright", "install"]
        logging.info(f"Command to be executed: {' '.join(install_command)}")

        try:
            process = subprocess.run(
                install_command,
                capture_output=True,
                text=True,
                check=True # This will raise CalledProcessError if the command returns a non-zero exit code
            )

            logging.info("Playwright installation command executed successfully.")
            logging.info("--- Standard Output ---")
            for line in process.stdout.splitlines():
                logging.debug(f"STDOUT: {line}")

            logging.info("--- Standard Error ---")
            for line in process.stderr.splitlines():
                 logging.warning(f"STDERR: {line}")

            logging.info(f"Ritual complete. Exit code: {process.returncode}")

            # Create the flag file to indicate successful installation
            try:
                playwright_flag_file_path.touch()
                logging.info(f"Created Playwright installation flag file: {playwright_flag_file_path}")
            except Exception as e:
                logging.error(f"Error creating Playwright installation flag file {playwright_flag_file_path}: {e}")
                # Do not exit here, as the installation itself was successful.

        except FileNotFoundError:
            logging.error(
                "Execution Error: The 'playwright' command was not found. "
                "Is Playwright installed and in your system's PATH?"
            )
            logging.error(
                "Suggestion: You might need to install Playwright first: pip install playwright"
            )
            sys.exit(1) # Exit on command not found

        except subprocess.CalledProcessError as e:
            logging.error(f"Execution Error: Playwright installation command failed with exit code {e.returncode}.")
            logging.error("--- Failed Standard Output ---")
            for line in e.stdout.splitlines():
                logging.error(f"STDOUT: {line}")

            logging.error("--- Failed Standard Error ---")
            for line in e.stderr.splitlines():
                logging.error(f"STDERR: {line}")
            sys.exit(1) # Exit on command failure

        except Exception as e:
            logging.error(f"An unexpected error occurred during command execution: {e}")
            sys.exit(1) # Exit on any other execution error

    logging.info("Initialization phase completed successfully.")
    logging.info("-" * 30)


    # --- Phase 2: Image Scraping (from image_scraper_v2_2.txt main) ---
    # Only proceed if the initialization phase did NOT exit.
    logging.info("Proceeding to the image scraping phase.")

    try:
        # Run the async main function from image_scraper_v2_2.txt
        # This main function will load config and set up its own logger again (reconfiguring the root logger).
        asyncio.run(main())
    except Exception as e:
        # Catch any unhandled exceptions from the async main execution
        # This should be logged by the scraper's logger, but this is a final fallback.
        print(f"FATAL ERROR: An unhandled exception occurred during the scraping process: {e}", file=sys.stderr)
        sys.exit(1)

    print("Script execution finished.") # Final print for clarity outside logging
