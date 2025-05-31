#!/usr/bin/env python3
import os
import sys
import glob
from datetime import datetime
import argparse
import curses
import re

class LogViewer:
    def __init__(self, logs_dir="data/outputs/logs"):
        self.logs_dir = logs_dir
        self.logs = []
        self.current_position = 0
        self.filter_pattern = None
        self.show_errors_only = False
        self.combined_view = True

    def load_logs(self):
        """Load and sort all log files"""
        log_files = glob.glob(os.path.join(self.logs_dir, "*.log"))
        
        # Parse timestamps and sort logs
        self.logs = []
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.readlines()
                    if content:
                        # Extract timestamp from filename or first log line
                        timestamp = self._extract_timestamp(log_file, content[0])
                        self.logs.append({
                            'file': log_file,
                            'content': content,
                            'timestamp': timestamp
                        })
            except Exception as e:
                print(f"Error reading {log_file}: {str(e)}")
        
        # Sort logs by timestamp, newest first
        self.logs.sort(key=lambda x: x['timestamp'], reverse=True)

    def _extract_timestamp(self, filename, first_line):
        """Extract timestamp from filename or log line"""
        # Try to extract from filename first (YYYYMMDD_HHMMSS)
        filename_match = re.search(r'(\d{8}_\d{6})', filename)
        if filename_match:
            try:
                return datetime.strptime(filename_match.group(1), '%Y%m%d_%H%M%S')
            except ValueError:
                pass

        # Try to extract from log line
        line_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', first_line)
        if line_match:
            try:
                return datetime.strptime(line_match.group(1), '%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass

        # Return epoch if no timestamp found
        return datetime.fromtimestamp(0)

    def display(self, stdscr):
        """Display logs in curses interface"""
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_RED, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_GREEN, -1)
        
        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()
            
            # Display header
            header = f" Log Viewer | {'Combined' if self.combined_view else 'Single'} | {'Errors Only' if self.show_errors_only else 'All Logs'} "
            if self.filter_pattern:
                header += f"| Filter: {self.filter_pattern} "
            header = header.ljust(width, '-')
            stdscr.addstr(0, 0, header)
            
            # Display help
            help_text = "q:Quit | c:Toggle Combined | e:Toggle Errors | /:Filter | r:Refresh"
            stdscr.addstr(height-1, 0, help_text)
            
            # Display logs
            current_line = 1
            for log in self.logs:
                if current_line >= height-1:
                    break
                
                for line in log['content']:
                    if current_line >= height-1:
                        break
                    
                    # Apply filters
                    if self.show_errors_only and 'ERROR' not in line:
                        continue
                    if self.filter_pattern and self.filter_pattern.lower() not in line.lower():
                        continue
                    
                    # Color based on log level
                    color = 0
                    if 'ERROR' in line:
                        color = curses.color_pair(1)
                    elif 'WARNING' in line:
                        color = curses.color_pair(2)
                    elif 'INFO' in line:
                        color = curses.color_pair(3)
                    
                    try:
                        stdscr.addstr(current_line, 0, line[:width-1], color)
                        current_line += 1
                    except curses.error:
                        pass
                
                if not self.combined_view:
                    break
            
            # Get user input
            key = stdscr.getch()
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.combined_view = not self.combined_view
            elif key == ord('e'):
                self.show_errors_only = not self.show_errors_only
            elif key == ord('/'):
                curses.echo()
                stdscr.addstr(height-1, 0, "Filter: ")
                self.filter_pattern = stdscr.getstr().decode('utf-8')
                curses.noecho()
            elif key == ord('r'):
                self.load_logs()

def main():
    parser = argparse.ArgumentParser(description="Interactive Log Viewer")
    parser.add_argument("--logs-dir", default="data/outputs/logs",
                      help="Directory containing log files")
    args = parser.parse_args()
    
    viewer = LogViewer(args.logs_dir)
    viewer.load_logs()
    
    try:
        curses.wrapper(viewer.display)
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    main()
