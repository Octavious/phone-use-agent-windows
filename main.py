#!/usr/bin/env python3
"""
Main entry point for the Phone Agent with Qwen2.5-VL.
This script sets up and runs the agent with command-line arguments.
"""

import os
import json
import argparse
import logging
from pathlib import Path

# Import our main agent class
from phone_agent import PhoneAgent

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"phone_agent_{Path(__file__).stem}.log"),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        logging.warning(f"Config file not found: {config_path}. Using defaults.")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing config file: {e}")
        return {}

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Phone Agent with Qwen2.5-VL')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--task', type=str, required=True,
                        help='Task to execute (e.g., "Open Chrome and search for weather")')
    parser.add_argument('--max-cycles', type=int, default=10,
                        help='Maximum number of interaction cycles')
    parser.add_argument('--device-id', type=str, default=None,
                        help='ADB device ID (optional, will use first device if not specified)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.device_id:
        config['device_id'] = args.device_id
    
    # Create and run the agent
    try:
        logging.info(f"Starting Phone Agent with task: {args.task}")
        agent = PhoneAgent(config)
        
        # Execute the task
        result = agent.execute_task(args.task, max_cycles=args.max_cycles)
        
        # Log the result
        logging.info(f"Task execution result: {result}")
        
        if result['success']:
            print(f"✅ Task completed successfully in {result['cycles']} cycles.")
        else:
            print(f"❌ Task failed after {result['cycles']} cycles.")
        
    except KeyboardInterrupt:
        logging.info("Operation canceled by user")
        print("\n⚠️ Operation canceled by user")
    except Exception as e:
        logging.error(f"Error executing task: {e}", exc_info=True)
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
