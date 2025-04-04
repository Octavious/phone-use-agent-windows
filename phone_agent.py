import os
import json
import time
import logging
import tempfile
import subprocess
import base64
import platform
from datetime import datetime
from pathlib import Path
import ast

# Import our Gemini VL integration
from gemini_vl_agent import GeminiVLAgent

class PhoneAgent:
    """
    A phone agent that uses Gemini 2.0 Flash to analyze screenshots and control a physical 
    Android phone via ADB.
    """
    
    def __init__(self, vl_agent, config=None, max_cycles=10):
        """
        Initialize the phone agent with configuration.
        
        Args:
            vl_agent: Vision-Language agent instance (GeminiVLAgent)
            config (dict): Configuration for the agent
            max_cycles (int): Maximum number of execution cycles
        """
        default_config = {
            'device_id': None,  # Will use first connected device if None
            'screen_width': 1080,  # Pixel 5 dimensions
            'screen_height': 2340,
            'omniparser_path': './omniparser',
            'screenshot_dir': './screenshots',
            'max_retries': 3,
            'adb_path': None  # Path to adb executable
        }
        
        # Update default config with provided config
        if config:
            for key, value in config.items():
                if key in default_config:
                    default_config[key] = value
                else:
                    logging.warning(f"Unknown config key: {key}")
            
        self.config = default_config
        self.vl_agent = vl_agent
        self.max_cycles = max_cycles
        
        self.context = {
            'previous_actions': [],
            'current_app': "Home",
            'current_state': "",
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # Set up directories and ADB
        self._setup_directories()
        self._setup_adb()
        
        # Verify ADB connection
        self._check_adb_connection()
        
        logging.info(f"Phone Agent initialized with device: {self.config['device_id']}")
    
    def _setup_directories(self):
        """Set up required directories."""
        # Create screenshots directory
        screenshots_dir = Path(self.config['screenshot_dir'])
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured directory exists: {screenshots_dir}")
    
    def _setup_adb(self):
        """Set up ADB connection."""
        if platform.system() == 'Windows':
            # Try to find ADB in common locations
            adb_paths = [
                self.config.get('adb_path'),
                os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Android', 'Sdk', 'platform-tools', 'adb.exe'),
                os.path.join(os.environ.get('ANDROID_HOME', ''), 'platform-tools', 'adb.exe'),
                'adb.exe'  # Try in PATH
            ]
            
            for path in adb_paths:
                if path and os.path.exists(path):
                    self.config['adb_path'] = path
                    break
                    
            if not self.config['adb_path']:
                raise RuntimeError("ADB not found. Please install Android SDK platform-tools or specify adb_path in config.")
        else:
            self.config['adb_path'] = 'adb'
    
    def _check_adb_connection(self):
        """Verify ADB connection to device."""
        try:
            # Get list of devices
            result = self._run_adb_command('devices')
            devices = [line.split('\t')[0] for line in result.split('\n')[1:] if line.strip()]
            
            if not devices:
                raise RuntimeError("No devices found. Please connect an Android device.")
            
            # Use specified device or first available
            if self.config['device_id']:
                if self.config['device_id'] not in devices:
                    raise RuntimeError(f"Specified device {self.config['device_id']} not found.")
                device_id = self.config['device_id']
            else:
                device_id = devices[0]
                self.config['device_id'] = device_id
            
            logging.info(f"Using device: {device_id}")
            
            # Verify device is responsive
            self._run_adb_command('shell echo "ADB connection verified"')
            logging.info("ADB connection verified")
            
        except Exception as e:
            logging.error(f"ADB connection error: {e}")
            raise
    
    def _run_adb_command(self, command):
        """Run an ADB command and return its output."""
        try:
            full_command = f"{self.config['adb_path']} -s {self.config['device_id']} {command}"
            result = subprocess.run(full_command, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"ADB command failed: {result.stderr}")
                
            return result.stdout.strip()
            
        except Exception as e:
            logging.error(f"Error running ADB command: {e}")
            raise
    
    def capture_screen(self):
        """Capture a screenshot of the device screen."""
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screen_{timestamp}_{int(time.time())}.png"
            screenshot_path = Path(self.config['screenshot_dir']) / filename
            
            # Capture screenshot
            self._run_adb_command(f'shell screencap -p /sdcard/{filename}')
            self._run_adb_command(f'pull /sdcard/{filename} "{screenshot_path}"')
            self._run_adb_command(f'shell rm /sdcard/{filename}')
            
            logging.info(f"Screenshot saved to: {screenshot_path}")
            return str(screenshot_path)
            
        except Exception as e:
            logging.error(f"Error capturing screen: {e}")
            raise
    
    def parse_screen(self, screenshot_path):
        """Parse the screen using OmniParser."""
        try:
            # Create temporary file for output
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
                output_path = tmp.name
            
            logging.info(f"Created temporary output file: {output_path}")
            
            # Run OmniParser
            command = f"python {Path(__file__).parent}/omniparser_runner.py --input {screenshot_path} --output {output_path} --use_paddleocr --box_threshold 0.05 --iou_threshold 0.1 --imgsz 640"
            logging.info(f"Running OmniParser with command: {command}")
            
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                logging.error(f"Error parsing screen with omniparser: {command}")
                logging.error(f"Command output:\n{result.stdout}")
                logging.error(f"Command error:\n{result.stderr}")
                raise RuntimeError(f"OmniParser failed with exit code {result.returncode}")
            
            logging.info(f"OmniParser command completed successfully")
            logging.info(f"Reading output from: {output_path}")
            
            # Read and parse output
            with open(output_path, 'r') as f:
                output_data = json.load(f)
            
            logging.info(f"Raw OmniParser output: {json.dumps(output_data, indent=2)}")
            
            if not output_data:
                logging.error("OmniParser returned empty output")
                return []
            
            # Save OmniParser output to a debug file
            debug_dir = Path(self.config['screenshot_dir']) / 'debug'
            debug_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file = debug_dir / f"omniparser_output_{timestamp}.json"
            
            with open(debug_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logging.info(f"Saved OmniParser output to: {debug_file}")
            
            # Clean up temporary file
            os.unlink(output_path)
            
            processed_elements = self._process_omniparser_output(output_data)
            logging.info(f"Processed elements: {json.dumps(processed_elements, indent=2)}")
            
            return processed_elements
            
        except Exception as e:
            logging.error(f"Error in parse_screen: {e}")
            logging.error(f"Error type: {type(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _process_omniparser_output(self, output):
        """Process OmniParser output into a structured format."""
        try:
            if not output or 'filtered_boxes' not in output:
                return []
            
            elements = []
            for box in output['filtered_boxes']:
                if not isinstance(box, dict):
                    continue
                    
                element = {
                    'type': box.get('type', 'unknown'),
                    'content': box.get('content', ''),
                    'bbox': box.get('bbox', []),
                    'interactivity': box.get('interactivity', False)
                }
                elements.append(element)
            
            return elements
            
        except Exception as e:
            logging.error(f"Error processing OmniParser output: {e}")
            return []
    
    def execute_action(self, action):
        """Execute an action on the device."""
        try:
            # Parse action
            if isinstance(action, str):
                action = ast.literal_eval(action)
            
            action_type = action.get('type', '').lower()
            target = action.get('target', {})
            
            if action_type == 'tap':
                # Convert normalized coordinates to actual coordinates
                x, y = self._translate_coordinates(target.get('x', 0), target.get('y', 0))
                self._run_adb_command(f'shell input tap {x} {y}')
                logging.info(f"Tapped at ({x}, {y})")
                
            elif action_type == 'type':
                text = target.get('text', '')
                self._run_adb_command(f'shell input text "{text}"')
                logging.info(f"Typed text: {text}")
                
            elif action_type == 'swipe':
                # Convert normalized coordinates
                start_x, start_y = self._translate_coordinates(target.get('start_x', 0), target.get('start_y', 0))
                end_x, end_y = self._translate_coordinates(target.get('end_x', 0), target.get('end_y', 0))
                duration = target.get('duration', 500)
                
                self._run_adb_command(f'shell input swipe {start_x} {start_y} {end_x} {end_y} {duration}')
                logging.info(f"Swiped from ({start_x}, {start_y}) to ({end_x}, {end_y})")
                
            elif action_type == 'press':
                key = target.get('key', '')
                self._run_adb_command(f'shell input keyevent {key}')
                logging.info(f"Pressed key: {key}")
                
            else:
                logging.warning(f"Unknown action type: {action_type}")
            
            # Update context
            self.context['previous_actions'].append(action)
            
        except Exception as e:
            logging.error(f"Error executing action: {e}")
            raise
    
    def _translate_coordinates(self, normalized_x, normalized_y):
        """Convert normalized coordinates to actual screen coordinates."""
        x = int(normalized_x * self.config['screen_width'])
        y = int(normalized_y * self.config['screen_height'])
        return x, y
    
    def execute_cycle(self, user_request):
        """Execute one cycle of the agent."""
        try:
            # Capture screen
            screenshot_path = self.capture_screen()
            
            # Parse screen
            screen_elements = self.parse_screen(screenshot_path)
            
            # Analyze with Gemini
            response = self.vl_agent.analyze_screenshot(
                screenshot_path,
                user_request,
                screen_elements,
                self.context
            )
            
            # Parse response and execute actions
            try:
                # Extract actions from response
                actions = self._parse_actions(response)
                
                # Execute each action
                for action in actions:
                    self.execute_action(action)
                    
                return True
                
            except Exception as e:
                logging.error(f"Error parsing or executing actions: {e}")
                return False
                
        except Exception as e:
            logging.error(f"Error in execution cycle: {e}")
            return False
    
    def _parse_actions(self, response):
        """Parse actions from Gemini's response."""
        try:
            # Split response into lines and look for action descriptions
            lines = response.split('\n')
            actions = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Look for action descriptions
                if line.lower().startswith('tap'):
                    # Parse coordinates
                    try:
                        coords_str = line.split('at')[1].strip().strip('()')
                        x, y = map(float, coords_str.split(','))
                        actions.append({
                            'type': 'tap',
                            'target': {'x': x, 'y': y}
                        })
                    except (IndexError, ValueError) as e:
                        logging.warning(f"Could not parse tap coordinates from: {line}")
                        continue
                        
                elif line.lower().startswith('type'):
                    # Extract text to type
                    try:
                        text = line.split('type')[1].strip().strip('"')
                        actions.append({
                            'type': 'type',
                            'target': {'text': text}
                        })
                    except IndexError as e:
                        logging.warning(f"Could not parse type text from: {line}")
                        continue
                    
                elif line.lower().startswith('swipe'):
                    # Parse swipe parameters
                    try:
                        params = line.split('from')[1].strip().split('to')
                        if len(params) == 2:
                            start = params[0].strip().strip('()').split(',')
                            end = params[1].strip().strip('()').split(',')
                            if len(start) == 2 and len(end) == 2:
                                actions.append({
                                    'type': 'swipe',
                                    'target': {
                                        'start_x': float(start[0]),
                                        'start_y': float(start[1]),
                                        'end_x': float(end[0]),
                                        'end_y': float(end[1]),
                                        'duration': 500
                                    }
                                })
                    except (IndexError, ValueError) as e:
                        logging.warning(f"Could not parse swipe parameters from: {line}")
                        continue
                        
                elif line.lower().startswith('press'):
                    # Parse key code
                    try:
                        key_code = int(line.split('press')[1].strip())
                        actions.append({
                            'type': 'press',
                            'target': {'key': key_code}
                        })
                    except (IndexError, ValueError) as e:
                        logging.warning(f"Could not parse key code from: {line}")
                        continue
            
            logging.info(f"Parsed actions: {json.dumps(actions, indent=2)}")
            return actions
            
        except Exception as e:
            logging.error(f"Error parsing actions: {e}")
            logging.error(f"Response text: {response}")
            return []
    
    def run(self, user_request):
        """Run the agent with the given request."""
        logging.info(f"Starting task: {user_request}")
        
        for cycle in range(self.max_cycles):
            logging.info(f"Starting cycle {cycle + 1}")
            
            try:
                success = self.execute_cycle(user_request)
                
                if success:
                    logging.info(f"Cycle {cycle + 1} completed successfully")
                    return True  # Return True if cycle is successful
                else:
                    logging.error(f"Cycle {cycle + 1} failed")
                    
            except Exception as e:
                logging.error(f"Error in cycle {cycle + 1}: {e}")
                logging.info("Retrying... (1/3)")
                
                # Wait before retrying
                time.sleep(2)
                
                try:
                    success = self.execute_cycle(user_request)
                    if success:
                        logging.info(f"Retry successful")
                        return True  # Return True if retry is successful
                    else:
                        logging.error(f"Retry failed")
                except Exception as e:
                    logging.error(f"Error in retry: {e}")
                    break
        
        # If we get here, all cycles failed
        return False


if __name__ == "__main__":
    # Example usage
    import json
    
    # Load configuration from file
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading config.json: {e}")
        print("Using default configuration...")
        config = {
            'device_id': None,  # Will use first connected device
            'screen_width': 1080,
            'screen_height': 2340,
            'omniparser_path': './omniparser',
            'screenshot_dir': './screenshots',
            'max_retries': 3,
            'adb_path': None
        }
    
    # Initialize Gemini agent
    from gemini_vl_agent import GeminiVLAgent
    vl_agent = GeminiVLAgent(api_key=os.getenv('GOOGLE_API_KEY'))
    
    # Create phone agent with loaded config
    agent = PhoneAgent(
        vl_agent=vl_agent,
        config=config,
        max_cycles=10
    )
    
    try:
        agent.run('Open Chrome and search for the weather in New York')
    except Exception as e:
        print(f"Task execution failed: {e}")
