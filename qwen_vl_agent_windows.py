import os
import base64
import logging
import platform
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq

class QwenVLAgent:
    """
    Vision-Language LLM integration for the phone agent using Qwen2.5-VL with standard transformers.
    This class handles the processing of screenshots and generation of actions.
    """
    
    def __init__(self, model_path, use_gpu=True, temperature=0.1):
        """
        Initialize the Qwen VL model with standard transformers.
        
        Args:
            model_path (str): Path to the Qwen model
            use_gpu (bool): Whether to use GPU acceleration
            temperature (float): Sampling temperature for generation
        """
        logging.info(f"Loading Qwen model from: {model_path} ...")
        
        # Configure device
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        # Load model and processor
        try:
            # First try loading as a vision-language model
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
                self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            except Exception as e:
                logging.warning(f"Failed to load as vision-language model: {e}")
                # Fallback to causal LM if needed
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
                self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
        
        # Set default parameters
        self.temperature = temperature
        self.max_tokens = 1024
        
        # System prompt
        self.system_prompt = """
        You are a phone UI automation agent. Your task is to analyze phone screenshots and determine
        the next action to take based on the user's request. You will be shown a single screenshot
        of a phone screen along with information about interactive elements.
        
        IMPORTANT UI RULES:
        1. If you need to enter text into a text field, you MUST first 'tap' that text field (even if it appears selected) in one cycle. 
        2. On the *next* cycle, you can 'type' into that field. Never 'type' without a prior 'tap' on the same element.

        IMPORTANT: 
        1. You must respond ONLY with a JSON object containing a single action to perform.
        2. Valid actions are 'tap', 'swipe', 'type', and 'wait'.
        3. For tap actions, you must include the element ID and coordinates.
        4. Include a brief reasoning explaining why you chose this action.
        """
        
        logging.info("Qwen model loaded successfully.")
    
    def _encode_image(self, image_path):
        """
        Encode an image to base64 for the model.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Base64-encoded image
        """
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    
    def analyze_screenshot(self, screenshot_path, user_request, screen_elements, context=None):
        """
        Analyze a screenshot and determine the next action.
        
        Args:
            screenshot_path (str): Path to the screenshot image
            user_request (str): The user's request
            screen_elements (list): List of interactive elements detected by omniparser
            context (dict): Current context information
            
        Returns:
            dict: Action to perform
        """
        # Encode the screenshot
        img_b64 = self._encode_image(screenshot_path)
        
        # Format the screen elements for the prompt
        formatted_elements = "\n".join([
            f"ID: {el['id']} | Type: {el['type']} | Content: \"{el['content']}\" | "
            f"Position: ({el['position']['x']}, {el['position']['y']}) | "
            f"Interactive: {el.get('interactivity', False)}"
            for el in screen_elements
        ])
        
        # Create context description
        if context:
            context_info = [
                f"Previous actions: {', '.join([str(a) for a in context.get('previous_actions', [])])}",
                f"Current app: {context.get('current_app', 'Unknown')}",
                f"Current state: {context.get('current_state', '')}"
            ]
            context_description = "\n".join(context_info)
        else:
            context_description = "No prior context"
        
        # Build the user message
        user_message = f"""
        # Phone Screen Analysis
        
        ## User Request
        "{user_request}"
        
        ## Context
        {context_description}
        
        ## Screen Elements
        {formatted_elements}
        
        ## Instructions
        Analyze the screen and determine a single action to take. Provide your response as a JSON object with the following structure:
        {{
          "action": "tap" | "swipe" | "type" | "wait",
          "elementId": number,  // The ID of the element to interact with (only for tap)
          "elementName": string,  // The name of the element (for reference)
          "coordinates": [x, y],  // For tap or swipe actions (normalized 0-1 coordinates)
          "direction": "up" | "down" | "left" | "right",  // Only for swipe
          "text": string,  // Only for type action
          "waitTime": number,  // In milliseconds, only for wait action
          "reasoning": string  // A brief explanation of why this action was chosen
        }}
        """
        
        # Create Qwen-style messages
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt.strip()}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/png;base64,{img_b64}",
                    },
                    {
                        "type": "text", 
                        "text": user_message
                    }
                ]
            }
        ]
        
        # Process the input
        inputs = self.processor(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32768
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode the response
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the JSON response
        try:
            # Find the JSON object in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                import json
                return json.loads(json_str)
            else:
                raise ValueError("No JSON object found in response")
        except Exception as e:
            logging.error(f"Error parsing model response: {e}")
            return {
                "action": "wait",
                "waitTime": 1000,
                "reasoning": "Error parsing model response, waiting for next cycle"
            } 