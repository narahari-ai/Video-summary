#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
from pathlib import Path
from setuptools import setup, find_packages

# Setup package if being installed
if "install" in sys.argv or "develop" in sys.argv:
    setup(
        name="video_summary",
        version="1.0.0",
        packages=find_packages(),
        install_requires=[
            line.strip()
            for line in open("requirements.txt")
            if line.strip() and not line.startswith("#")
        ],
    )

def create_venv():
    """Create a virtual environment if it doesn't exist"""
    print("📦 Creating virtual environment...")
    venv_path = Path(".venv")
    if not venv_path.exists():
        subprocess.run(["python3", "-m", "venv", ".venv"], check=True)
    return venv_path

def get_venv_python():
    """Get the python executable path from virtual environment"""
    if sys.platform == "win32":
        return ".venv/Scripts/python"
    return ".venv/bin/python"

def get_venv_pip():
    """Get the pip executable path from virtual environment"""
    if sys.platform == "win32":
        return ".venv/Scripts/pip"
    return ".venv/bin/pip"

def install_requirements():
    """Install Python package requirements"""
    print("📥 Installing Python packages...")
    pip = get_venv_pip()
    subprocess.run([pip, "install", "-r", "requirements.txt"], check=True)

def download_models():
    """Download required AI models"""
    print("🤖 Downloading AI models...")
    
    # Create model directories
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Download Vosk model
    vosk_dir = models_dir / "vosk"
    if not (vosk_dir / "vosk-model-en-us").exists():
        print("Downloading Vosk model...")
        vosk_dir.mkdir(exist_ok=True)
        subprocess.run([
            "wget", "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
            "-O", str(vosk_dir / "model.zip")
        ], check=True)
        subprocess.run([
            "unzip", str(vosk_dir / "model.zip"),
            "-d", str(vosk_dir)
        ], check=True)
        os.remove(str(vosk_dir / "model.zip"))
    
    # Whisper models will be downloaded automatically when first used
    print("Note: Whisper models will be downloaded automatically when first used")

def verify_ffmpeg():
    """Verify FFmpeg installation"""
    print("🎥 Checking FFmpeg installation...")
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ FFmpeg is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg is not installed")
        if sys.platform == "linux":
            print("Installing FFmpeg...")
            subprocess.run(["sudo", "apt", "update"], check=True)
            subprocess.run(["sudo", "apt", "install", "-y", "ffmpeg"], check=True)
        else:
            print("Please install FFmpeg manually:")
            print("- Windows: https://www.ffmpeg.org/download.html")
            print("- macOS: brew install ffmpeg")
            sys.exit(1)

def test_pipeline():
    """Run a test pipeline with the sample video"""
    print("🧪 Testing the pipeline...")
    python = get_venv_python()
    subprocess.run([
        python, "scripts/run_all.py",
        "--video", "data/videos/sample.mp4"
    ], check=True)

def main():
    parser = argparse.ArgumentParser(description="Setup Video Intelligence Assistant")
    parser.add_argument("--skip-models", action="store_true",
                      help="Skip downloading large model files")
    parser.add_argument("--skip-test", action="store_true",
                      help="Skip running the test pipeline")
    args = parser.parse_args()

    try:
        # Create virtual environment
        create_venv()
        
        # Install requirements
        install_requirements()
        
        # Verify FFmpeg installation
        verify_ffmpeg()
        
        # Download models unless skipped
        if not args.skip_models:
            download_models()
        
        # Run test pipeline unless skipped
        if not args.skip_test and Path("data/videos/sample.mp4").exists():
            test_pipeline()
        
        print("✨ Setup completed successfully!")
        print("\nTo activate the virtual environment:")
        if sys.platform == "win32":
            print("    .venv\\Scripts\\activate")
        else:
            print("    source .venv/bin/activate")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
