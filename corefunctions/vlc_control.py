import subprocess
import os
import time
import requests
import threading

class VLCController:
    def __init__(self, video_path, password="vlcpassword", port="8080"):
        """
        Initialize VLC controller with video file path
        
        Args:
            video_path (str): Path to the video file
            password (str): Password for VLC HTTP interface
            port (str): Port for VLC HTTP interface
        """
        self.video_path = video_path
        self.password = password
        self.port = port
        self.vlc_process = None
        self.vlc_path = None
        self.auth = ('', password)
        self.base_url = f"http://localhost:{port}/requests/status.xml"
        
        # Find VLC executable on initialization
        self._find_vlc_path()
        
    def _find_vlc_path(self):
        """Find VLC executable on Raspberry Pi"""
        possible_paths = [
            "/usr/bin/vlc",
            "/usr/local/bin/vlc",
            "/snap/bin/vlc"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.vlc_path = path
                return
        
        # Try which command
        try:
            result = subprocess.run(['which', 'vlc'], capture_output=True, text=True)
            if result.returncode == 0:
                self.vlc_path = result.stdout.strip()
                return
        except:
            pass
        
        self.vlc_path = None
        
    def launch(self, fullscreen=True, wait_for_load=5):
        """
        Launch VLC with HTTP interface
        
        Args:
            fullscreen (bool): Whether to launch in fullscreen mode
            wait_for_load (int): Seconds to wait for VLC to fully load
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.vlc_path:
            print("VLC not found. Install with: sudo apt install vlc")
            return False
            
        if not os.path.exists(self.video_path):
            print(f"Video file not found: {self.video_path}")
            return False
        
        # Build VLC arguments
        vlc_args = [
            self.vlc_path,
            "--intf", "http",
            "--http-password", self.password,
            "--http-port", self.port,
            "--extraintf", "rc",
            "--no-qt-privacy-ask",  # Skip privacy dialog
            "--no-video-title-show",  # Don't show filename
        ]
        
        if fullscreen:
            vlc_args.append("--fullscreen")
            
        vlc_args.append(self.video_path)
        
        try:
            # Set display for GUI applications
            env = os.environ.copy()
            if 'DISPLAY' not in env:
                env['DISPLAY'] = ':0'
            
            self.vlc_process = subprocess.Popen(vlc_args, env=env)
            print(f"Launching VLC with HTTP interface on port {self.port}")
            
            # Wait for VLC to fully load
            if wait_for_load > 0:
                time.sleep(wait_for_load)
                
            return True
            
        except Exception as e:
            print(f"Error launching VLC: {e}")
            return False
    
    def _send_command(self, command, timeout=5):
        """
        Send HTTP command to VLC
        
        Args:
            command (str): Command to send
            timeout (int): Request timeout in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_running():
            print("VLC is not running")
            return False
            
        try:
            url = f"{self.base_url}?command={command}"
            response = requests.get(url, auth=self.auth, timeout=timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Command failed: {e}")
            return False
    
    def pause(self):
        """
        Pause the video
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("Pausing video")
        return self._send_command("pl_pause")
    
    def play(self):
        """
        Play/resume the video
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("Playing video")
        return self._send_command("pl_play")
    
    def stop(self):
        """
        Stop the video
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("Stopping video")
        return self._send_command("pl_stop")
    
    def restart(self):
        """
        Restart video from the beginning
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("Restarting video from beginning")
        # Seek to position 0 and ensure it's playing
        seek_success = self._send_command("seek&val=0")
        play_success = self._send_command("pl_play")
        return seek_success and play_success
    
    def seek(self, position):
        """
        Seek to a specific position in seconds
        
        Args:
            position (int): Position in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"Seeking to position {position} seconds")
        return self._send_command(f"seek&val={position}")
    
    def is_running(self):
        """
        Check if VLC process is running
        
        Returns:
            bool: True if running, False otherwise
        """
        return self.vlc_process is not None and self.vlc_process.poll() is None
    
    def close(self):
        """
        Close VLC player
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.vlc_process:
            print("Closing VLC...")
            try:
                self.vlc_process.terminate()
                self.vlc_process.wait(timeout=10)
                return True
            except subprocess.TimeoutExpired:
                print("Force killing VLC...")
                self.vlc_process.kill()
                self.vlc_process.wait()
                return True
            except Exception as e:
                print(f"Error closing VLC: {e}")
                return False
        return True
    
    def wait_until_closed(self):
        """
        Block until VLC is closed by user or programmatically
        
        Returns:
            bool: True when closed
        """
        if not self.is_running():
            return True
            
        try:
            while self.is_running():
                time.sleep(1)
            return True
        except KeyboardInterrupt:
            print("\nInterrupted by user...")
            self.close()
            return True
    
    def play_with_pause_and_restart(self, pause_after=20, pause_duration=10):
        """
        Play video, pause after specified time, then restart from beginning
        
        Args:
            pause_after (int): Seconds to wait before pausing
            pause_duration (int): Seconds to stay paused
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_running():
            print("VLC is not running. Launch first.")
            return False
        
        try:
            # Wait specified time
            print(f"Video playing... waiting {pause_after} seconds")
            time.sleep(pause_after)
            
            # Pause the video
            print(f"Pausing video for {pause_duration} seconds")
            self.pause()
            
            # Wait while paused
            time.sleep(pause_duration)
            
            # Restart from beginning
            self.restart()
            
            return True
            
        except KeyboardInterrupt:
            print("\nInterrupted by user...")
            return True
        except Exception as e:
            print(f"Error during playback sequence: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    video_file = "/home/dieter/Videos/DeadlyPrey1988vhsrip_TryFile.com_.avi"
    
    print("Testing VLC Controller...")
    vlc = VLCController(video_file)
    
    try:
        # Launch VLC
        if vlc.launch():
            print("VLC launched successfully!")
            
            # Run the pause/restart sequence
            vlc.play_with_pause_and_restart(pause_after=20, pause_duration=10)
            
            # Keep running until user closes
            print("Video restarted. Press Ctrl+C to close VLC.")
            vlc.wait_until_closed()
        else:
            print("Failed to launch VLC")
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        vlc.close()