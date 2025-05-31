import os
from typing import Optional
from urllib.parse import urlparse
import pytube
import requests
import ffmpeg
import shutil
from datetime import datetime
import uuid
import logging
from pathlib import Path

class VideoLoader:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = "data/videos"
        self.processed_dir = Path("data/videos/processed")
        os.makedirs(self.output_dir, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def _get_unique_path(self, base_path: str) -> str:
        """Generate a unique path for the output file"""
        dir_name = os.path.dirname(base_path)
        file_name, ext = os.path.splitext(os.path.basename(base_path))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return os.path.join(dir_name, f"{file_name}_{timestamp}_{unique_id}{ext}")

    def _validate_video(self, path: str) -> bool:
        """Validate video file using ffprobe"""
        try:
            probe = ffmpeg.probe(path)
            video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
            if not video_streams:
                return False
            return True
        except ffmpeg.Error:
            return False

    def load(self, source: str) -> str:
        """
        Load video from various sources and return the path to the downloaded video
        
        Args:
            source: URL or local path to the video
            
        Returns:
            str: Path to the downloaded/local video file
        """
        if self.is_local_file(source):
            return self._handle_local_file(source)
        elif self.is_youtube_url(source):
            return self._download_youtube(source)
        elif self.is_redentias_url(source):
            return self._download_redentias(source)
        else:
            raise ValueError(f"Unsupported video source: {source}")

    @staticmethod
    def is_local_file(path: str) -> bool:
        return os.path.exists(path)

    @staticmethod
    def is_youtube_url(url: str) -> bool:
        return "youtube.com" in url or "youtu.be" in url

    @staticmethod
    def is_redentias_url(url: str) -> bool:
        parsed = urlparse(url)
        return "redentias" in parsed.netloc

    def _handle_local_file(self, path: str) -> str:
        """Copy local file to data directory if needed"""
        # First validate the source video
        if not self._validate_video(path):
            raise ValueError(f"Invalid video file: {path}")

        filename = os.path.basename(path)
        target_path = os.path.join(self.output_dir, filename)
        
        if path != target_path:
            # Generate unique path to prevent overwrites
            target_path = self._get_unique_path(target_path)
            # Copy file instead of moving
            shutil.copy2(path, target_path)
            self.logger.info(f"Copied video to {target_path}")

        return path  # Return original path since we want to keep source file intact

    def _download_youtube(self, url: str) -> str:
        """Download YouTube video"""
        yt = pytube.YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
        
        if not stream:
            raise ValueError(f"No suitable stream found for YouTube video: {url}")
            
        filename = f"{yt.video_id}.mp4"
        target_path = os.path.join(self.output_dir, filename)
        stream.download(output_path=self.output_dir, filename=filename)
        
        return target_path

    def _download_redentias(self, url: str) -> str:
        """Download video from Redentias"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        filename = os.path.basename(urlparse(url).path) or "video.mp4"
        target_path = os.path.join(self.output_dir, filename)
        
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return target_path
