import json
import time
import logging
from pathlib import Path
from phone_agent import PhoneAgent
from gemini_vl_agent import GeminiVLAgent
import os
from dotenv import load_dotenv

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('task_execution.log'),
            logging.StreamHandler()
        ]
    )

def load_config():
    """Load configuration from config.json."""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading config.json: {e}")
        return {
            'device_id': None,
            'screen_width': 1080,
            'screen_height': 2340,
            'omniparser_path': './omniparser',
            'screenshot_dir': './screenshots',
            'max_retries': 3,
            'adb_path': None
        }

def load_tasks():
    """Load tasks from tasks.json."""
    try:
        with open('tasks.json', 'r') as f:
            return json.load(f)['tasks']
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading tasks.json: {e}")
        return []

def execute_task(agent, task):
    """Execute a single task."""
    logging.info(f"Executing task {task['id']}: {task['description']}")
    try:
        # Set max_cycles to 1 for this task
        agent.max_cycles = 1
        success = agent.run(task['command'])
        
        if success:
            logging.info(f"Task {task['id']} completed successfully")
            return True
        else:
            logging.error(f"Task {task['id']} failed to complete")
            return False
            
    except Exception as e:
        logging.error(f"Error executing task {task['id']}: {e}")
        return False

def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()
    
    # Set up logging
    setup_logging()
    logging.info("Starting task execution")
    
    # Load configuration and tasks
    config = load_config()
    tasks = load_tasks()
    
    if not tasks:
        logging.error("No tasks found in tasks.json")
        return
    
    # Initialize agents
    try:
        vl_agent = GeminiVLAgent(api_key=os.getenv('GOOGLE_API_KEY'))
        phone_agent = PhoneAgent(
            vl_agent=vl_agent,
            config=config,
            max_cycles=1  # Set initial max_cycles to 1
        )
    except Exception as e:
        logging.error(f"Error initializing agents: {e}")
        return
    
    # Execute tasks in sequence
    for task in tasks:
        logging.info(f"Starting task {task['id']}: {task['description']}")
        
        # Try the task up to 3 times
        task_success = False
        for attempt in range(3):
            if execute_task(phone_agent, task):
                task_success = True
                break
            elif attempt < 2:  # Don't wait after the last attempt
                logging.info(f"Retrying task {task['id']}... (attempt {attempt + 2}/3)")
                time.sleep(2)
        
        if not task_success:
            logging.error(f"Task {task['id']} failed after all attempts. Stopping execution.")
            break
            
        # Wait between tasks
        if task['id'] < len(tasks):
            logging.info("Waiting 3 seconds before next task...")
            time.sleep(3)
    
    logging.info("All tasks completed")

if __name__ == "__main__":
    main() 