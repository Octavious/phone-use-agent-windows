import os
import base64
import logging
from pathlib import Path

# Import Qwen VL and vLLM dependencies
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

class QwenVLAgent:
    """
    Vision-Language LLM integration for the phone agent using Qwen2.5-VL with vLLM.
    This class handles the processing of screenshots and generation of actions.
    """
    
    def __init__(self, model_path, use_gpu=True, temperature=0.1):
        """
        Initialize the Qwen VL model with vLLM for optimal performance.
        
        Args:
            model_path (str): Path to the Qwen model
            use_gpu (bool): Whether to use GPU acceleration
            temperature (float): Sampling temperature for generation
        """
        logging.info(f"Loading vLLM-based Qwen model from: {model_path} ...")
        
        # Configure GPU usage
        gpu_config = {}
        if use_gpu:
            gpu_config = {
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.80 # This works on a 24gb card with Omniparser running as well
            }
        
        # Create the vLLM LLM instance
        self.llm = LLM(
            model=model_path,
            max_model_len=32768,
            limit_mm_per_prompt={"image": 1},  # We only need one screenshot at a time
            **gpu_config
        )
        
        # Create the processor for building prompts + handling images
        self.processor = AutoProcessor.from_pretrained(model_path)
        
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
        
        logging.info("vLLM Qwen model loaded successfully.")
    
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
        qwen_messages = [
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
        
        # Build the prompt using Qwen's apply_chat_template
        prompt = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Prepare multi-modal data
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            qwen_messages,
            return_video_kwargs=True
        )
        
        # Build dict for vLLM
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        
        # Construct final llm_inputs
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }
        
        # Build SamplingParams
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            repetition_penalty=1.05,
            top_p=0.9,
            top_k=40
        )
        
        # Run vLLM inference
        outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params)
        
        # Process the result
        if not outputs or not outputs[0].outputs:
            logging.error("No output generated by the model")
            return None
        
        generated_text = outputs[0].outputs[0].text
        
        # Parse the JSON response
        try:
            import json
            import re
            
            # Try to extract JSON if it's within a code block or other formatting
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', generated_text)
            if json_match:
                action_json = json.loads(json_match.group(1).strip())
            else:
                # Try to parse the whole text as JSON
                action_json = json.loads(generated_text.strip())
            
            logging.info(f"Generated action: {action_json}")
            return action_json
        except (json.JSONDecodeError, Exception) as e:
            logging.error(f"Error parsing model output: {e}")
            logging.error(f"Raw output: {generated_text}")
            return None
