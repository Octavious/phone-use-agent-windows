# Phone Use Agent

An experimental Python agent that controls Android phones using Qwen2.5-VL, OmniParser, and ADB.

![Phone Use Agent Architecture](docs/workflow.png)

## Overview

The Phone Use Agent automates interactions with Android devices by:
- Taking screenshots via ADB
- Analyzing UI elements with OmniParser
- Making decisions with Qwen2.5-VL vision language model through vLLM
- Executing actions (tap, swipe, type) through ADB

## Requirements

- Python 3.10
- Linux operating system
- Android Debug Bridge (ADB)
- CUDA-capable GPU (Tested on 3xxx GPU with Cuda 12.4)
- Connected Android device with USB debugging enabled

## Installing ADB on Linux

ADB is required for the Phone Agent to communicate with your Android device. Install it on Linux with:

```bash
sudo apt update
sudo apt install adb
```

Verify the installation with:
```bash
adb version
```

## Setup with OmniParser

1. Clone this repository:
   ```bash
   git clone https://github.com/OminousIndustries/phone-use-agent.git
   cd phone-use-agent
   ```

2. Clone OmniParser into the phone-use-agent directory:
   ```bash
   git clone https://github.com/microsoft/OmniParser.git
   ```

3. Create and activate conda environment:
   ```bash
   conda create -n "phone_agent" python==3.10
   conda activate phone_agent
   ```

4. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Download OmniParser weights:
   ```bash
   cd OmniParser

   # Create a folder for icon_detect but NOT icon_caption_florence:
   mkdir -p weights/icon_detect

   # Download weights from HF
   for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do
       huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights
   done

   # Rename the icon_caption -> icon_caption_florence
   mv weights/icon_caption weights/icon_caption_florence
   ```

6. Return to main directory:
   ```bash
   cd ..
   ```

## Device Configuration

**Important:** You must set the correct screen resolution for your specific device in `config.json`. The default values are for a Pixel 5:

```json
{
  "screen_width": 1080,
  "screen_height": 2340
}
```

To find your device's resolution, run:
```bash
adb shell wm size
```

Update the values in `config.json` to match your device's resolution exactly. Incorrect resolution settings will cause the agent to tap in the wrong locations.

## Usage Options

### Command Line Interface

1. Connect your Android device via USB and enable USB debugging in Developer Options
2. Ensure conda environment is activated:
   ```bash
   conda activate phone_agent
   ```
3. Reccomended to run the first time through the CLI so we can see vLLM Qwen2.5VL download process
4. Run a task:
   ```bash
   python main.py --task "Open Chrome and search for weather in New York" --max-cycles 10
   ```

5. Additional options:
   ```bash
   python main.py --help
   ```

### Graphical User Interface

A simple Gradio UI is provided to visualize the agent's progress:

```bash
python ui.py
```

The UI provides:
- Input field for your task
- View of the phone's screen at screenshot intervals
- Log output
- Auto-refresh functionality

## Configuration

Edit `config.json` to configure:
- Device dimensions (must match your actual device)
- Model selection (3B vs 7B)
- OmniParser settings
- General execution parameters

```json
{
  "device_id": null,
  "screen_width": 1080,
  "screen_height": 2340,
  "omniparser_path": "./OmniParser",
  "screenshot_dir": "./screenshots",
  "max_retries": 3,
  "qwen_model_path": "Qwen/Qwen2.5-VL-3B-Instruct",
  "use_gpu": true,
  "temperature": 0.1,

  "omniparser_config": {
    "use_paddleocr": true,
    "box_threshold": 0.05,
    "iou_threshold": 0.1,
    "imgsz": 640
  }
}
```

## How It Works

The Phone Agent follows this workflow:

1. **User Request**: Define a task like "Open Chrome and search for weather"
2. **Capture**: Take a screenshot of the phone screen via ADB
3. **Analyze**: Use OmniParser to identify UI elements (buttons, text fields, icons)
4. **Decide**: Qwen2.5-VL analyzes screenshot and elements to determine next action
5. **Execute**: ADB performs the action (tap, swipe, type text)
6. **Repeat**: Continue the cycle until task completion or max cycles reached

The Main Controller manages execution cycles, tracks context between actions, handles errors, and implements retry logic when actions fail.

## Components

- **ADB Bridge**: Handles communication with the Android device
- **OmniParser**: Identifies interactive elements on the screen
- **Qwen VL Agent**: Makes decisions based on visual input and task context
- **Main Controller**: Orchestrates the execution cycles and manages state

## Troubleshooting

- **Wrong tap locations**: Verify your device resolution in `config.json` matches the actual device
- **ADB connection issues**: Make sure USB debugging is enabled and you've authorized the computer on your device
- **OmniParser errors**: Check that all model weights are correctly downloaded and placed in the proper directories
- **Gradio errors**: If using the UI, make sure you have gradio installed (`pip install gradio`)
- **OOM Errors from vLLM**: The Qwen2.5VL 3B and 7B models can take up a lot of memory. If you have a dual GPU setup, it is possible to set Omniparser to run on the second GPU to allow for more memory to run the model on the first GPU by uncommenting `# os.environ["CUDA_VISIBLE_DEVICES"] = "1"` on line 21 of omniparser_runner.py 

## Experimental Status

This project is experimental and intended for research purposes. It may not work perfectly for all devices or UI layouts.

