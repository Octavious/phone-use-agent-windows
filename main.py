#!/usr/bin/env python3
"""
Main entry point for the Phone Agent with Gemini 2.0 Flash.
This script sets up and runs the agent with command-line arguments.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Import our main agent class
from phone_agent import PhoneAgent
from gemini_vl_agent import GeminiVLAgent

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
    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Phone Agent with Gemini 2.0 Flash')
    parser.add_argument('--task', required=True, help='Task to perform')
    parser.add_argument('--max-cycles', type=int, default=int(os.getenv('MAX_CYCLES', 10)), help='Maximum number of execution cycles')
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    parser.add_argument('--api-key', help='Google API key for Gemini (overrides .env file)')
    parser.add_argument('--temperature', type=float, default=float(os.getenv('GEMINI_TEMPERATURE', 0.1)), help='Temperature for text generation')
    args = parser.parse_args()

    # Set up logging
    setup_logging()

    # Load configuration
    config = load_config(args.config)

    # Get API key from command line, environment variable, or .env file
    api_key = args.api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("API key not found. Please set GOOGLE_API_KEY in .env file or provide --api-key argument.")

    # Initialize the Gemini VL agent
    vl_agent = GeminiVLAgent(
        api_key=api_key,
        temperature=args.temperature
    )

    # Initialize and run the phone agent
    agent = PhoneAgent(
        vl_agent=vl_agent,
        config=config,
        max_cycles=args.max_cycles
    )

    try:
        agent.run(args.task)
    except KeyboardInterrupt:
        logging.info("Agent stopped by user")
    except Exception as e:
        logging.error(f"Error running agent: {e}")
        raise

if __name__ == "__main__":
    main()
