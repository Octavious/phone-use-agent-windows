import os
import json
import logging
import gradio as gr
from pathlib import Path
from threading import Thread

from phone_agent import PhoneAgent

class UILogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []
    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)
        # Keep only last 100 entries
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]

def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

# Global shared state
current_screenshot = None
log_handler = None
is_running = False
agent = None

def setup_logging():
    global log_handler
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Clear out any old handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    # UI log handler
    log_handler = UILogHandler()
    log_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler.setFormatter(formatter)
    root_logger.addHandler(log_handler)
    
    # Also log to file
    file_handler = logging.FileHandler("phone_agent_ui.log")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

def execute_task(task, max_cycles=10):
    """
    Runs in a background thread. Clears logs and sets is_running=True
    until done, then sets is_running=False.
    """
    global current_screenshot, is_running, agent
    if log_handler:
        log_handler.logs = []
    is_running = True
    config = load_config()

    try:
        logging.info(f"Starting Phone Agent with task: '{task}'")
        agent = PhoneAgent(config)
        
        # Monkey-patch parse_screen to capture screenshots
        original_parse_screen = agent.parse_screen
        def parse_screen_with_capture(screenshot_path):
            result = original_parse_screen(screenshot_path)
            annotated_path = f"{os.path.splitext(screenshot_path)[0]}_annotated.png"
            if os.path.exists(annotated_path):
                global current_screenshot
                current_screenshot = annotated_path
            return result
        
        agent.parse_screen = parse_screen_with_capture
        
        # Execute
        result = agent.execute_task(task, max_cycles=max_cycles)
        
        if result['success']:
            logging.info(f"✅ Task completed successfully in {result['cycles']} cycles")
        else:
            logging.info(f"❌ Task failed after {result['cycles']} cycles")
    except Exception as e:
        logging.error(f"Error executing task: {e}")
    finally:
        is_running = False

def start_task(task, max_cycles):
    """
    1) If already running, do nothing.
    2) Otherwise, start the background thread for the agent.
    3) Return outputs for:
       - log_output
       - image_output
       - timer (to enable it)
    """
    global is_running
    if is_running:
        return (
            "Task is already running",  # log_output
            None,                      # image_output
            gr.update(active=False)    # Keep timer off if it's already running
        )
    
    # Convert cycles
    try:
        max_cycles = int(max_cycles)
    except ValueError:
        max_cycles = 10
    
    # Start the background thread
    thread = Thread(target=execute_task, args=(task, max_cycles))
    thread.daemon = True
    thread.start()
    
    # Return: log text, image, and turn timer on
    return (
        "Task started",   # log_output
        None,            # image_output
        gr.update(active=True)
    )

def update_ui(_=None):
    """
    Called on each Timer tick.  
    Returns:
      - current screenshot (if any)
      - concatenated logs
      - updated timer state (turn off if no longer running)
    """
    global current_screenshot, log_handler, is_running
    
    # 1) Gather screenshot
    screenshot = None
    if current_screenshot and os.path.exists(current_screenshot):
        screenshot = current_screenshot
    
    # 2) Gather logs
    logs = "\n".join(log_handler.logs) if log_handler else ""
    
    # 3) If the agent is done, disable the timer so it stops polling
    if not is_running:
        return (screenshot, logs, gr.update(active=False))
    
    # Otherwise, keep it active (unchanged)
    return (screenshot, logs, gr.update())

def create_ui():
    config = load_config()
    screenshot_dir = config.get('screenshot_dir', './screenshots')
    Path(screenshot_dir).mkdir(parents=True, exist_ok=True)
    
    with gr.Blocks(title="Phone Agent UI") as demo:
        gr.Markdown("# Phone Agent Control Panel")
        
        with gr.Row():
            with gr.Column(scale=3):
                task_input = gr.Textbox(
                    label="Task to execute",
                    placeholder="e.g., Open Chrome and check the weather",
                    lines=2
                )
                max_cycles = gr.Textbox(label="Maximum cycles", value="10")
                start_button = gr.Button("Start Task", variant="primary")
                
            with gr.Column(scale=5):
                image_output = gr.Image(label="Current Screen", type="filepath")

        log_output = gr.Textbox(
            label="Log Output",
            lines=20,
            max_lines=20,
            interactive=False
        )

        # This Timer starts OFF (active=False). We'll enable it when user starts a task.
        timer = gr.Timer(value=5, active=False)
        
        # 1) Button to start the agent -> enable the timer
        start_button.click(
            fn=start_task,
            inputs=[task_input, max_cycles],
            outputs=[log_output, image_output, timer]
        )
        
        # 2) Timer calls update_ui repeatedly
        #    We now return 3 outputs from update_ui:
        #      (screenshot, logs, timer-update)
        #    to allow turning timer off automatically
        timer.tick(
            fn=update_ui,
            inputs=None,
            outputs=[image_output, log_output, timer]
        )
        
        # 3) Manual refresh button (optional)
        refresh_button = gr.Button("Refresh Display")
        refresh_button.click(
            fn=update_ui,
            inputs=None,
            outputs=[image_output, log_output, timer]
        )
        
        gr.Markdown("""
        **Usage**  
        1. Enter your task above and click **Start Task**.  
        2. Once started, the Timer becomes active and auto-refreshes every 5s.  
        3. The Timer auto-stops after the task is finished, so you won't get timeouts.  
        4. You can still click **Refresh Display** manually any time.
        """)
        
    return demo

def main():
    setup_logging()
    demo = create_ui()
    # Optional concurrency management
    demo.queue()
    demo.launch(server_name="0.0.0.0", share=True)

if __name__ == "__main__":
    main()
