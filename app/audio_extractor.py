import os
import ffmpeg
import subprocess
from typing import Optional, Tuple
import logging
from pathlib import Path
import uuid

class AudioExtractor:
    def __init__(self):
        self.output_dir = Path("data/audios")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def validate_video(self, video_path: str) -> Tuple[bool, Optional[str]]:
        """Validate video file integrity"""
        try:
            # Use ffprobe to check video metadata
            probe = ffmpeg.probe(video_path)
            duration = float(probe['format']['duration'])
            
            if duration < 1.0:
                return False, "Video duration is too short"
                
            # Check if video has audio stream
            has_audio = any(stream['codec_type'] == 'audio' 
                          for stream in probe['streams'])
            if not has_audio:
                return False, "No audio stream found in video"
                
            return True, None
            
        except ffmpeg.Error as e:
            return False, f"FFmpeg error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def extract(self, video_path: str) -> str:
        """Extract audio from video file"""
        # Validate video first
        is_valid, error = self.validate_video(video_path)
        if not is_valid:
            raise ValueError(f"Invalid video file: {error}")

        # Create unique output path
        video_filename = os.path.basename(video_path)
        name = os.path.splitext(video_filename)[0]
        unique_id = str(uuid.uuid4())[:8]
        output_path = self.output_dir / f"{name}_{unique_id}.wav"

        try:
            # Extract audio using ffmpeg
            self.logger.info(f"Extracting audio to {output_path}")
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(stream.audio, 
                                str(output_path),
                                acodec='pcm_s16le',
                                ac=1,
                                ar='16k')
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)

            if not output_path.exists() or output_path.stat().st_size < 1024:
                raise ValueError("Audio extraction produced no or invalid output")

            return str(output_path)

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise ValueError(f"FFmpeg error during audio extraction: {error_msg}")
        except Exception as e:
            raise ValueError(f"Error during audio extraction: {str(e)}")
