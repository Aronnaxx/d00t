#!/bin/bash

# Non-interactive script to run Duck Robot with gemma3:4b model
# This version runs automatically without prompting the user

# Enable logging for debugging
exec > >(tee -a "run_with_gemma_auto.log") 2>&1
echo "[$(date)] Starting run_with_gemma_auto.sh"

# Display banner
echo "===================================================="
echo "🦆 Duck Robot with Gemma 3 (Automatic Mode) 🦆"
echo "===================================================="
echo ""

# Parse command line arguments
USE_VOICE=false
for arg in "$@"
do
    case $arg in
        --voice)
        USE_VOICE=true
        shift
        ;;
    esac
done

# Check if ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Error: Ollama is not installed."
    echo "Please install Ollama from: https://ollama.com/download"
    exit 1
fi
echo "✅ Ollama is installed."

# Check for the gemma3:4b model and pull if not available
if ! ollama list | grep -q "gemma3:4b"; then
    echo "⚠️ Gemma 3 4b model is not installed. Pulling now..."
    ollama pull gemma3:4b
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to pull the model. Please check your internet connection."
        exit 1
    fi
    echo "✅ Model successfully installed."
else
    echo "✅ Gemma 3 4b model is available."
fi

# Check for voice dependencies if voice mode requested
VOICE_OPTION=""
if [ "$USE_VOICE" = true ]; then
    echo "Checking voice dependencies..."
    python -c "import sounddevice" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "✅ Voice dependencies are available. Voice mode enabled."
        VOICE_OPTION="--voice --tts"
    else
        echo "⚠️ Voice dependencies are not installed. Running without voice features."
        echo "To enable voice features, install:"
        echo "  sudo apt-get install portaudio19-dev"
        echo "  pip install sounddevice"
    fi
else
    echo "ℹ️ Voice mode not requested. Running without voice features."
    echo "Use --voice flag to enable voice features if dependencies are available."
fi

# Run the Duck Robot with gemma3:4b model
echo ""
echo "Starting Duck Robot with Gemma 3 model..."
echo "Command: python start_robot.py --ollama-model gemma3:4b --autonomous $VOICE_OPTION"
python start_robot.py --ollama-model gemma3:4b --autonomous $VOICE_OPTION

# Store the exit code
EXIT_CODE=$?

echo "[$(date)] Duck Robot has been stopped (exit code: $EXIT_CODE)"
exit $EXIT_CODE 