#!/bin/bash

# Script to run Duck Robot with gemma3:4b model

# Enable logging for debugging
exec > >(tee -a "run_with_gemma.log") 2>&1
echo "[$(date)] Starting run_with_gemma.sh"

# Display banner
echo "===================================================="
echo "💡 Duck Robot with Gemma 3 Model Setup Helper 💡"
echo "===================================================="
echo ""

# Check if ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed."
    echo "To install Ollama, visit: https://ollama.com/download"
    echo "Follow the installation instructions for your platform."
    exit 1
else
    echo "✅ Ollama is installed."
fi

# Check for the gemma3:4b model
if ollama list | grep -q "gemma3:4b"; then
    echo "✅ Gemma 3 4b model is available."
else
    echo "❌ Gemma 3 4b model is not installed."
    echo "To install the model, run:"
    echo "  ollama pull gemma3:4b"
    
    echo ""
    echo "Would you like to pull the model now? (y/n)"
    read -r PULL_MODEL
    
    if [[ "$PULL_MODEL" == "y" || "$PULL_MODEL" == "Y" ]]; then
        echo "Pulling gemma3:4b model (this may take a while)..."
        ollama pull gemma3:4b
        
        if [ $? -ne 0 ]; then
            echo "❌ Failed to pull the model. Please check your internet connection."
            exit 1
        else
            echo "✅ Model successfully installed."
        fi
    else
        echo "⚠️ Model not installed. You will need to install it before running."
    fi
fi

# Check for voice dependencies
echo ""
echo "Checking voice dependencies..."
python -c "import sounddevice" 2>/dev/null
SOUNDDEVICE_AVAILABLE=$?

if [ $SOUNDDEVICE_AVAILABLE -eq 0 ]; then
    echo "✅ Voice dependencies are available."
    VOICE_STATUS="Voice mode is available (--voice --tts)"
else
    echo "⚠️ Voice dependencies are not installed."
    echo "To enable voice features, install:"
    echo "  sudo apt-get install portaudio19-dev"
    echo "  pip install sounddevice"
    VOICE_STATUS="Voice mode is NOT available (missing dependencies)"
fi

# Display instructions
echo ""
echo "===================================================="
echo "🦆 How to Run Duck Robot with Gemma 3 🦆"
echo "===================================================="
echo ""
echo "There are several ways to run the Duck Robot with Gemma 3:"
echo ""
echo "1️⃣ Using the config file:"
echo "   Create ~/.duck_robot/config.py with:"
echo "   ----------------------------------------"
echo "   ollama_model = \"gemma3:4b\""
echo "   ----------------------------------------"
echo "   Then run: python start_robot.py"
echo ""
echo "2️⃣ Using environment variables:"
echo "   ----------------------------------------"
echo "   export DUCK_OLLAMA_MODEL=gemma3:4b"
echo "   python start_robot.py"
echo "   ----------------------------------------"
echo ""
echo "3️⃣ Using command line arguments:"
echo "   ----------------------------------------"
echo "   python start_robot.py --ollama-model gemma3:4b --autonomous"
echo "   ----------------------------------------"
echo ""
echo "Voice Status: $VOICE_STATUS"
if [ $SOUNDDEVICE_AVAILABLE -eq 0 ]; then
    echo "To enable voice: Add --voice --tts to any command"
fi
echo ""
echo "Would you like to run the Duck Robot with Gemma 3 now? (y/n)"
read -r RUN_NOW

if [[ "$RUN_NOW" == "y" || "$RUN_NOW" == "Y" ]]; then
    VOICE_OPTION=""
    if [ $SOUNDDEVICE_AVAILABLE -eq 0 ]; then
        echo "Would you like to enable voice features? (y/n)"
        read -r ENABLE_VOICE
        if [[ "$ENABLE_VOICE" == "y" || "$ENABLE_VOICE" == "Y" ]]; then
            VOICE_OPTION="--voice --tts"
        fi
    fi
    
    echo "Running: python start_robot.py --ollama-model gemma3:4b --autonomous $VOICE_OPTION"
    python start_robot.py --ollama-model gemma3:4b --autonomous $VOICE_OPTION
else
    echo "Exiting without running. Follow the instructions above to run later."
fi

echo "[$(date)] Script completed." 