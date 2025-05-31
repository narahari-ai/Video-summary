# Video Intelligence Assistant

A modular Python system for extracting insigh2. Run the full pipeline:
```bash
# Basic run
python3 scripts/run_all.py --video path/to/your/video.mp4

# Clean previous outputs for this video before processing
python3 scripts/run_all.py --video path/to/your/video.mp4 --clean video

# Clean all previous outputs before processing
python3 scripts/run_all.py --video path/to/your/video.mp4 --clean allom video content using AI. The system can:
- Extract text transcripts from videos
- Generate summaries using multiple AI models
- Create visual mind maps
- Generate study notes and FAQs

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- FFmpeg (will be installed automatically on Linux)
- Git
- wget (for downloading models)
- graphviz (for mindmap generation)
  ```bash
  # On Ubuntu/Debian:
  sudo apt-get install graphviz
  # On macOS:
  brew install graphviz
  # On Windows:
  choco install graphviz
  ```

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd video-summary
```

2. Run the setup script:
```bash
python3 setup.py
```

This will:
- Create a Python virtual environment
- Install required packages
- Install FFmpeg (if needed)
- Download required AI models
- Run a test pipeline with the sample video

To skip downloading large model files:
```bash
python3 setup.py --skip-models
```

### Virtual Environment

1. Activate the virtual environment:
```bash
# From the project root directory
source .venv/bin/activate

# Your terminal prompt should now show (.venv) at the beginning
# Example: (.venv) user@host:~/AI/video-summary$
```

2. Deactivate when you're done:
```bash
deactivate
```

**Note**: Always make sure the virtual environment is activated before running any scripts or installing packages.

### Environment Setup

1. Set up the environment (this will set PYTHONPATH and activate the virtual environment):
```bash
source setup_env.sh
```

2. You should see a confirmation message showing the PYTHONPATH is set correctly.

3. When you're done working, deactivate the environment:
```bash
deactivate
```

### 🏃‍♂️ Running the Pipeline

1. Activate the virtual environment:
```bash
# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

2. If using WSL (Windows Subsystem for Linux), you can copy videos using one of these methods:

   a. Using Windows File Explorer:
   - Navigate to: `\\wsl$\Ubuntu\home\hari\AI\video-summary\data\videos`
   - Copy your video files directly to this folder

   b. Using Windows Command Line:
   ```cmd
   copy "C:\path\to\your\video.mp4" "\\wsl$\Ubuntu\home\hari\AI\video-summary\data\videos\"
   ```

   c. Using WSL Terminal (if video is in Windows Downloads):
   ```bash
   cp /mnt/c/Users/YourWindowsUsername/Downloads/your_video.mp4 ~/AI/video-summary/data/videos/
   ```

3. Run the full pipeline:
```bash
python3 scripts/run_all.py --video path/to/your/video.mp4

python3 scripts/run_all.py --video /home/hari/AI/video-summary/data/videos/class30.mp4

```

Or run individual components:
```bash
# Just transcription
python3 scripts/run_transcription.py --video path/to/video.mp4 --model whisper_base

/home/hari/AI/video-summary/data/videos/class30.mp4

# Just summarization
python3 scripts/run_summarization.py --transcript path/to/transcript.txt --model bart

# Generate notes from summary
python3 scripts/run_notes.py --summary path/to/summary.md
```

## 📁 Project Structure

```
video_ai_assistant/
├── app/                    # Core application logic
├── data/                   # Input/output assets
│   ├── videos/            # Input videos
│   ├── audios/            # Extracted audio
│   └── outputs/           # Generated outputs
├── models/                # AI model files
├── configs/               # Configuration files
└── scripts/               # Runner scripts
```

## 📄 Generated Outputs

All outputs are saved in the `data/outputs/` directory:
- `transcripts/`: Text transcriptions
- `summaries/`: AI-generated summaries
- `mindmaps/`: Visual mind maps
- `notes/`: Study notes
- `faqs/`: Generated FAQs
- `logs/`: Processing logs

### 📊 Viewing Logs

The system includes an interactive log viewer that can be used to view and analyze logs:

```bash
python3 scripts/view_logs.py
```

Log viewer features:
- Combined view of all logs (newest first)
- Filter logs by text
- Toggle error-only view
- Color-coded log levels
- Real-time log updates

Keyboard shortcuts:
- `q`: Quit
- `c`: Toggle combined/single log view
- `e`: Toggle errors-only view
- `/`: Enter filter pattern
- `r`: Refresh logs

Each processing run creates two log files:
- Regular log: `<video_name>_<timestamp>.log`
- Error log: `<video_name>_<timestamp>_error.log`

## 🤖 Supported AI Models

### Transcription
- Whisper (tiny/base/medium/large)
- Vosk (offline)

### Summarization
- BART (facebook/bart-large-cnn)
- T5 (t5-base)
- Pegasus (google/pegasus-xsum)
- MT5 (google/mt5-small)
- GPT-2 (text generation based summarization)

Each model can be downloaded individually using:
```bash
python3 scripts/download_models.py
```

Then select the model number you want to download:
1. Whisper (base)
2. Vosk
3. BART
4. T5
5. Pegasus
6. GPT-2
7. MT5
