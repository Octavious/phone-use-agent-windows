import os
import json
import time
import logging
import tempfile
import subprocess
import base64
from datetime import datetime
from pathlib import Path
import ast

# Import our Qwen VL integration
from qwen_vl_agent import QwenVLAgent

class PhoneAgent:
    """
    A phone agent that uses Qwen2.5-VL to analyze screenshots and control a physical 
    Android phone via ADB.
    """
    
    def __init__(self, config=None):
        """
        Initialize the phone agent with configuration.
        
        Args:
            config (dict): Configuration for the agent
        """
        default_config = {
            'device_id': None,  # Will use first connected device if None
            'screen_width': 1080,  # Pixel 5 dimensions
            'screen_height': 2340,
            'omniparser_path': './omniparser',
            'screenshot_dir': './screenshots',
            'max_retries': 3,
            'qwen_model_path': 'Qwen/Qwen2.5-VL-3B-Instruct',
            'use_gpu': True,
            'temperature': 0.1
        }
        
        self.config = default_config
        if config:
            self.config.update(config)
        
        self.context = {
            'previous_actions': [],
            'current_app': "Home",
            'current_state': "",
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"phone_agent_{self.context['session_id']}.log"),
                logging.StreamHandler()
            ]
        )

        # Initialize the Qwen VL model
        logging.info("Initializing Qwen VL agent... may take a while")
        self.vl_agent = QwenVLAgent(
            model_path=self.config['qwen_model_path'],
            use_gpu=self.config['use_gpu'],
            temperature=self.config['temperature']
        )
        
        self._setup_directories()
        self._check_adb_connection()
    
    def _setup_directories(self):
        """Create necessary directories for storing screenshots."""
        Path(self.config['screenshot_dir']).mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured directory exists: {self.config['screenshot_dir']}")
    
    def _check_adb_connection(self):
        """Verify ADB connection to the device."""
        try:
            # Check for connected devices
            result = subprocess.run(
                "adb devices",
                shell=True, check=True, capture_output=True, text=True
            )
            
            # Get the device ID
            if self.config['device_id'] is None:
                # Extract the first device from the list
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # First line is "List of devices attached"
                    device_info = lines[1].split('\t')
                    if len(device_info) > 0 and device_info[1].strip() == 'device':
                        self.config['device_id'] = device_info[0].strip()
                        logging.info(f"Using device: {self.config['device_id']}")
                    else:
                        raise Exception("No authorized device found")
                else:
                    raise Exception("No devices connected")
            
            # Test a simple ADB command to verify connection
            device_cmd_prefix = f"-s {self.config['device_id']}" if self.config['device_id'] else ""
            subprocess.run(
                f"adb {device_cmd_prefix} shell echo 'Connected'",
                shell=True, check=True, capture_output=True, text=True
            )
            
            logging.info("ADB connection verified")
        except subprocess.CalledProcessError as e:
            logging.error(f"ADB connection error: {e}")
            raise Exception("Failed to connect to device via ADB. Make sure USB debugging is enabled.")
    
    def _run_adb_command(self, command):
        """
        Run an ADB command on the connected device.
        
        Args:
            command (str): ADB command to run
            
        Returns:
            str: Command output
        """
        device_cmd_prefix = f"-s {self.config['device_id']}" if self.config['device_id'] else ""
        full_command = f"adb {device_cmd_prefix} {command}"
        
        try:
            result = subprocess.run(
                full_command,
                shell=True, check=True, capture_output=True, text=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            logging.error(f"ADB command error: {e}")
            logging.error(f"Command stderr: {e.stderr}")
            raise
    
    def capture_screen(self):
        """
        Capture screenshot from the device.
        
        Returns:
            str: Path to the saved screenshot
        """
        timestamp = int(time.time())
        screenshot_path = os.path.join(
            self.config['screenshot_dir'],
            f"screen_{self.context['session_id']}_{timestamp}.png"
        )
        
        try:
            # Use ADB to capture screenshot and save it locally
            # For a physical device, we use the following approach
            self._run_adb_command(f"shell screencap -p /sdcard/screenshot.png")
            self._run_adb_command(f"pull /sdcard/screenshot.png {screenshot_path}")
            self._run_adb_command(f"shell rm /sdcard/screenshot.png")
            
            logging.info(f"Screenshot saved to: {screenshot_path}")
            return screenshot_path
        except Exception as e:
            logging.error(f"Error capturing screenshot: {e}")
            raise
    
    def parse_screen(self, screenshot_path):
        """
        Run omniparser on the screenshot to extract UI elements.
        
        Args:
            screenshot_path (str): Path to the screenshot
            
        Returns:
            list: Processed screen elements
        """
        try:
            # Use our custom omniparser_runner.py script
            runner_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "omniparser_runner.py")
            
            # Create a temporary file for output
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
                output_path = tmp.name
            
            # Build the command with parameters from config
            cmd = f"python {runner_script} --input {screenshot_path} --output {output_path}"
            
            # Add optional parameters from config
            omniparser_config = self.config.get('omniparser_config', {})
            if omniparser_config.get('use_paddleocr', True):
                cmd += " --use_paddleocr"
            if 'box_threshold' in omniparser_config:
                cmd += f" --box_threshold {omniparser_config['box_threshold']}"
            if 'iou_threshold' in omniparser_config:
                cmd += f" --iou_threshold {omniparser_config['iou_threshold']}"
            if 'imgsz' in omniparser_config:
                cmd += f" --imgsz {omniparser_config['imgsz']}"
            
            # Log the command we're executing
            logging.info(f"Running OmniParser with command: {cmd}")
            
            # Execute the command
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            
            if result.stderr:
                logging.warning(f"Omniparser warnings: {result.stderr}")
            
            # Read and process the output
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    parsed_output = json.load(f)
                os.unlink(output_path)
                
                # Save the annotated image for debugging (optional)
                if 'annotated_image' in parsed_output:
                    debug_path = f"{os.path.splitext(screenshot_path)[0]}_annotated.png"
                    with open(debug_path, 'wb') as f:
                        f.write(base64.b64decode(parsed_output['annotated_image']))
                    logging.info(f"Saved annotated image to: {debug_path}")
                
                # Return the parsed elements
                elements = parsed_output.get('elements', [])
                logging.info(f"Detected {len(elements)} screen elements")
                
                # Convert elements to the expected format
                return self._process_omniparser_output("\n".join([f'icon {i}: {element}' for i, element in enumerate(elements)]))
            else:
                raise FileNotFoundError(f"Output file not found: {output_path}")
        
        except subprocess.CalledProcessError as e:
            logging.error(f"Error parsing screen with omniparser: {e}")
            logging.error(f"Command output: {e.stdout}")
            logging.error(f"Command error: {e.stderr}")
            raise
        except Exception as e:
            logging.error(f"Error in parse_screen: {e}")
            raise
    
    def _process_omniparser_output(self, output):
        """
        Process omniparser output and extract useful information.
        
        Args:
            output (str): Raw omniparser output
            
        Returns:
            list: Structured screen elements
        """
        screen_elements = []
        lines = output.strip().split('\n')
        
        for line in lines:
            if not line.startswith('icon '):
                continue
            
            try:
                # Extract the icon number and content
                icon_match = line.split(':', 1)
                if len(icon_match) != 2:
                    continue
                
                icon_id = icon_match[0].replace('icon ', '')
                
                # Convert the content to proper dict format
                content_text = icon_match[1].strip()
                # Convert single quotes to double quotes for JSON parsing
                content_text = content_text.replace("'", '"')
                content_text = content_text.replace('True', 'true').replace('False', 'false')
                
                try:
                    content = json.loads(content_text)
                except json.JSONDecodeError:
                    # Fallback to ast.literal_eval for handling Python dict representation
                    content = ast.literal_eval(icon_match[1].strip())
                
                # Add the icon id
                content['id'] = int(icon_id)
                
                # Calculate center point for interaction
                bbox = content['bbox']
                center = [
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2
                ]
                
                # Create a cleaner representation
                screen_element = {
                    'id': content['id'],
                    'type': content['type'],
                    'content': content['content'].strip(),
                    'interactivity': content.get('interactivity', False),
                    'position': {
                        'x': round(center[0], 3),
                        'y': round(center[1], 3)
                    },
                    'bbox': [round(coord, 3) for coord in bbox]
                }
                
                screen_elements.append(screen_element)
            except Exception as e:
                logging.error(f"Error processing line: {line}")
                logging.error(f"Exception: {e}")
                continue
        
        logging.info(f"Processed {len(screen_elements)} screen elements")
        return screen_elements
    
    def execute_action(self, action):
        """
        Execute the action on the device.
        
        Args:
            action (dict): Action to execute
            
        Returns:
            dict: Result of the action execution
        """
        try:
            # Manually convert 'click' to 'tap' for compatibility - sometimes the model will not strictly follow the prompt so we can force the correct action here 
            if action['action'] == 'click':
                logging.info(f"Converting 'click' action to 'tap'")
                action['action'] = 'tap'
            
            logging.info(f"Executing action: {action['action']} on element \"{action.get('elementName', 'unknown')}\"")

            # Fallback: Type → Tap (focus) if we never tapped it - prevents the model from trying to enter text before it has selected a text field
            if action['action'] == 'type':
                last_actions = self.context['previous_actions']
                
                # Make sure we at least have 'coordinates' in the original action JSON
                if 'coordinates' not in action:
                    # Without coordinates, we can’t convert "type" → "tap"
                    # so either skip or raise an error
                    msg = "No 'coordinates' given for a 'type' action. Skipping."
                    logging.info(msg)
                    return {
                        'success': False,
                        'error': msg,
                        'action': action
                    }
                
                # Also check if we tapped the same element last cycle
                if not last_actions or last_actions[-1].get('action') != 'tap' \
                or last_actions[-1].get('elementId') != action.get('elementId'):
                    
                    logging.info("Overriding 'type' action with a 'tap' first to focus text field.")
                    action = {
                        "action": "tap",
                        "elementId": action['elementId'],
                        "elementName": action.get('elementName', ''),
                        "coordinates": action['coordinates'],  # now we know 'coordinates' exist
                        "reasoning": "Need to tap first to focus the text field."
                    }

            # -------------------------------------------------------
            # Now handle whichever action we end up with
            # -------------------------------------------------------
            if action['action'] == 'tap':
                # Convert normalized coordinates to pixel coordinates
                x, y = self._translate_coordinates(
                    float(action['coordinates'][0]),
                    float(action['coordinates'][1])
                )
                # Execute the tap
                self._run_adb_command(f"shell input tap {x} {y}")
                logging.info(f"Tapped at coordinates ({x}, {y})")
            
            elif action['action'] == 'swipe':
                start_x = self.config['screen_width'] // 2
                start_y = self.config['screen_height'] // 2
                
                end_x, end_y = start_x, start_y
                direction = action['direction']
                
                if direction == 'up':
                    end_y = int(start_y * 0.3)
                elif direction == 'down':
                    end_y = int(start_y * 1.7)
                elif direction == 'left':
                    end_x = int(start_x * 0.3)
                elif direction == 'right':
                    end_x = int(start_x * 1.7)
                
                self._run_adb_command(f"shell input swipe {start_x} {start_y} {end_x} {end_y} 300")
                logging.info(f"Swiped {direction}")
            
            elif action['action'] == 'type':
                # Escape shell chars:
                escaped_text = action['text'].replace("'", "\\'").replace('"', '\\"')
                # Replace spaces with %s so the phone interprets them correctly - without this the text gets entered weird
                escaped_text = escaped_text.replace(" ", "%s")

                # Now run the command
                self._run_adb_command(f"shell input text \"{escaped_text}\"")
                logging.info(f"Typed text: \"{action['text']}\"")
            
            elif action['action'] == 'wait':
                wait_time = action.get('waitTime', 1000)  # Default to 1 second
                logging.info(f"Waiting for {wait_time}ms")
                time.sleep(wait_time / 1000)
            
            else:
                raise ValueError(f"Unknown action type: {action['action']}")
            
            return {
                'success': True,
                'action': action
            }
        
        except Exception as e:
            logging.error(f"Error executing action: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': action
            }

    
    def _translate_coordinates(self, normalized_x, normalized_y):
        """
        Translate normalized coordinates to actual pixel coordinates.
        
        Args:
            normalized_x (float): Normalized x coordinate (0-1)
            normalized_y (float): Normalized y coordinate (0-1)
            
        Returns:
            tuple: (x, y) pixel coordinates
        """
        x = int(normalized_x * self.config['screen_width'])
        y = int(normalized_y * self.config['screen_height'])
        return x, y
    
    def execute_cycle(self, user_request):
        """
        Execute a single interaction cycle.
        
        Args:
            user_request (str): User's request
            
        Returns:
            dict: Result of the action execution
        """
        try:
            # Step 1: Capture the screen
            screenshot_path = self.capture_screen()
            
            # Step 2: Parse the screen with omniparser
            screen_elements = self.parse_screen(screenshot_path)
            
            # Step 3: Use Qwen VL to analyze the screenshot and determine the action
            action = self.vl_agent.analyze_screenshot(
                screenshot_path,
                user_request,
                screen_elements,
                self.context
            )

            # 1) If there is an elementId but no coordinates, fill them from screen_elements
            if action.get("elementId") is not None and "coordinates" not in action:
                for elem in screen_elements:
                    if elem["id"] == action["elementId"]:
                        # Use the normalized position from OmniParser
                        x = elem["position"]["x"]
                        y = elem["position"]["y"]
                        action["coordinates"] = [x, y]
                        break
            
            if not action:
                raise Exception("Failed to determine action from screenshot")
            
            # Step 4: Execute the action on the device
            result = self.execute_action(action)
            
            # result['action'] is the final, possibly overridden action
            executed_action = result['action']

            # Update context with what was *actually* executed
            self.context['previous_actions'].append({
                'action': executed_action['action'],            # Store the final corrected action
                'elementId': executed_action.get('elementId'),  # etc.
                'elementName': executed_action.get('elementName', ''),
                'timestamp': time.time()
            })
            
            # If we're opening an app, update the context
            if (action['action'] == 'tap' and 
                action.get('elementName') and 
                action['elementName'] not in ['Circle', 'Dictate', 'Paste']):
                self.context['current_app'] = action['elementName']
            
            return result
        except Exception as e:
            logging.error(f"Error in execution cycle: {e}")
            raise
    
    def execute_task(self, user_request, max_cycles=10):
        """
        Execute a task by running multiple interaction cycles.
        
        Args:
            user_request (str): User's request
            max_cycles (int): Maximum number of cycles to execute
            
        Returns:
            dict: Result of the task execution
        """
        logging.info(f"Starting task: \"{user_request}\"")
        
        cycles = 0
        while cycles < max_cycles:
            try:
                result = self.execute_cycle(user_request)
                logging.info(f"Cycle {cycles + 1} completed: {result}")
                
                # Check if we should continue or if the task is complete
                # This could be determined by asking the LLM if the task is complete
                # or by checking for specific elements on the screen
                
                cycles += 1
                
                # Wait a moment for the UI to update
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error in cycle {cycles + 1}: {e}")
                
                # Retry logic
                if cycles < self.config['max_retries']:
                    logging.info(f"Retrying... ({cycles + 1}/{self.config['max_retries']})")
                    continue
                
                raise Exception(f"Failed to complete task after {cycles} cycles: {str(e)}")
        
        logging.info(f"Task completed after {cycles} cycles")
        return {
            'success': True,
            'cycles': cycles,
            'context': self.context
        }


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
            'qwen_model_path': 'Qwen/Qwen2.5-VL-3B-Instruct'
        }
    
    # Create agent with loaded config
    agent = PhoneAgent(config)
    
    try:
        result = agent.execute_task('Open Chrome and search for the weather in New York')
        print('Task execution result:', result)
    except Exception as e:
        print(f"Task execution failed: {e}")
