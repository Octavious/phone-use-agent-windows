import os
import base64
import logging
import platform
from pathlib import Path
from PIL import Image
import google.generativeai as genai

class GeminiVLAgent:
    """
    Vision-Language LLM integration for the phone agent using Gemini 2.0 Flash.
    This class handles the processing of screenshots and generation of actions.
    """
    
    def __init__(self, api_key=None, temperature=0.1):
        """
        Initialize the Gemini model.
        
        Args:
            api_key (str): Google API key for Gemini
            temperature (float): Sampling temperature for generation
        """
        # Configure API key
        if api_key:
            genai.configure(api_key=api_key)
        elif 'GOOGLE_API_KEY' in os.environ:
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        else:
            raise ValueError("API key not provided. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
            
        logging.info("Initializing Gemini model...")
        
        # Initialize the model
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.temperature = temperature
        
        # Configure generation parameters
        self.generation_config = {
            "temperature": temperature,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        logging.info("Gemini model initialized successfully.")
    
    def _encode_image(self, image_path):
        """Convert image to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logging.error(f"Error encoding image: {e}")
            return None
    
    def analyze_screenshot(self, screenshot_path, user_request, screen_elements, context=None):
        """
        Analyze a screenshot and generate appropriate actions.
        
        Args:
            screenshot_path (str): Path to the screenshot image
            user_request (str): User's request or task
            screen_elements (dict): Detected screen elements
            context (dict, optional): Additional context for the analysis
            
        Returns:
            str: Generated response with actions
        """
        try:
            logging.info(f"Starting screenshot analysis for: {screenshot_path}")
            logging.info(f"User request: {user_request}")
            logging.info(f"Screen elements: {screen_elements}")
            
            # Load and prepare the image
            image = Image.open(screenshot_path)
            logging.info(f"Image loaded successfully: {image.size}")
            
            # Prepare the prompt
            prompt = f"""You are a phone automation agent. Analyze this screenshot and help with the following request: {user_request}

Screen elements detected:
{self._format_screen_elements(screen_elements)}

Previous context:
{context if context else 'None'}

You must respond with a list of actions, one per line. Each action must follow EXACTLY one of these formats:

1. For tapping: "tap at (x, y)" where x and y are normalized coordinates between 0 and 1
2. For typing: 'type "text to type"'
3. For swiping: "swipe from (start_x, start_y) to (end_x, end_y)" where coordinates are normalized
4. For pressing keys: "press KEYCODE" where KEYCODE is the Android key code

Example actions:
tap at (0.5, 0.5)
type "weather in New York"
swipe from (0.5, 0.8) to (0.5, 0.2)
press 66

IMPORTANT:
1. Each action must be on a new line
2. Do not include any explanatory text or comments
3. Only output the actions in the exact format shown above
4. Use normalized coordinates (0-1) for all positions
5. For text input, use double quotes around the text
6. For key codes, use the standard Android key codes

Your response should ONLY contain the actions, one per line, nothing else."""

            logging.info("Sending request to Gemini API...")
            
            # Generate response
            response = self.model.generate_content(
                [prompt, image],
                generation_config=self.generation_config
            )
            
            logging.info("Received response from Gemini API")
            logging.info(f"Response: {response.text}")
            
            return response.text
            
        except Exception as e:
            logging.error(f"Error analyzing screenshot: {e}")
            logging.error(f"Error type: {type(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return f"Error analyzing screenshot: {str(e)}"
    
    def _format_screen_elements(self, elements):
        """Format screen elements for the prompt."""
        if not elements:
            return "No elements detected"
            
        formatted = []
        for i, element in enumerate(elements):
            if isinstance(element, dict):
                element_type = element.get('type', 'unknown')
                content = element.get('content', '')
                bbox = element.get('bbox', [])
                formatted.append(f"{i+1}. Type: {element_type}, Content: {content}, Position: {bbox}")
            else:
                formatted.append(f"{i+1}. {str(element)}")
                
        return "\n".join(formatted) 