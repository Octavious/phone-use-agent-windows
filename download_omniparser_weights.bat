@echo off
REM Windows batch script to download OmniParser weights

REM Check if OmniParser directory exists
if not exist "OmniParser" (
    echo Error: OmniParser directory not found
    echo Current directory: %CD%
    pause
    exit /b 1
)

REM Change to OmniParser directory
cd OmniParser

REM Create the weights/icon_detect directory
mkdir weights\icon_detect 2>nul

REM Download files one by one (Windows doesn't support the Linux-style brace expansion)
echo Downloading icon_detect files...
huggingface-cli download microsoft/OmniParser-v2.0 "icon_detect/train_args.yaml" --local-dir weights
huggingface-cli download microsoft/OmniParser-v2.0 "icon_detect/model.pt" --local-dir weights
huggingface-cli download microsoft/OmniParser-v2.0 "icon_detect/model.yaml" --local-dir weights

echo Downloading icon_caption files...
huggingface-cli download microsoft/OmniParser-v2.0 "icon_caption/config.json" --local-dir weights
huggingface-cli download microsoft/OmniParser-v2.0 "icon_caption/generation_config.json" --local-dir weights
huggingface-cli download microsoft/OmniParser-v2.0 "icon_caption/model.safetensors" --local-dir weights

REM Rename icon_caption to icon_caption_florence
echo Renaming icon_caption to icon_caption_florence...
ren "weights\icon_caption" "icon_caption_florence"

echo Download and setup complete!
echo Weights have been downloaded to: %CD%\weights
pause 