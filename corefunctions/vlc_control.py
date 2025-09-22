import subprocess
import os
import time
import requests
import threading

class VLCController:
    def __init__(self, video_files, password="vlcpassword", port="8080"):
        """
        Initialize VLC controller with video file list
        
        Args:
            video_files (list or str): List of video file paths or single path
            password (str): Password for VLC HTTP interface
            port (str): Port for VLC HTTP interface
        """
        # Handle both single file and list inputs
        if isinstance(video_files, str):
            self.video_files = [video_files]
        else:
            self.video_files = list(video_files)
            
        self.active_file_index = 0  # Index of currently active file
        self.password = password
        self.port = port
        self.vlc_process = None
        self.vlc_path = None
        self.auth = ('', password)
        self.base_url = f"http://localhost:{port}/requests/status.xml"
        self.last_launch_template = None  # Store launch template
        
        # Find VLC executable on initialization
        self._find_vlc_path()
        
    @property
    def active_file(self):
        """Get the currently active video file path"""
        if 0 <= self.active_file_index < len(self.video_files):
            return self.video_files[self.active_file_index]
        return None
        
    def add_video_file(self, file_path):
        """
        Add a video file to the list
        
        Args:
            file_path (str): Path to the video file
            
        Returns:
            int: Index of the added file
        """
        self.video_files.append(file_path)
        print(f"Added video file: {file_path}")
        return len(self.video_files) - 1
        
    def remove_video_file(self, index):
        """
        Remove a video file from the list
        
        Args:
            index (int): Index of the file to remove
            
        Returns:
            bool: True if successful, False otherwise
        """
        if 0 <= index < len(self.video_files):
            removed_file = self.video_files.pop(index)
            print(f"Removed video file: {removed_file}")
            
            # Adjust active file index if necessary
            if self.active_file_index >= len(self.video_files) and self.video_files:
                self.active_file_index = len(self.video_files) - 1
            elif not self.video_files:
                self.active_file_index = 0
            elif self.active_file_index > index:
                self.active_file_index -= 1
                
            return True
        return False
        
    def set_active_file(self, index):
        """
        Set the active video file by index
        
        Args:
            index (int): Index of the file to make active
            
        Returns:
            bool: True if successful, False otherwise
        """
        if 0 <= index < len(self.video_files):
            self.active_file_index = index
            print(f"Active file set to: {self.active_file}")
            return True
        else:
            print(f"Invalid file index: {index}. Available range: 0-{len(self.video_files)-1}")
            return False
            
    def set_active_file_by_name(self, filename):
        """
        Set the active video file by filename (searches for partial matches)
        
        Args:
            filename (str): Filename or partial filename to search for
            
        Returns:
            bool: True if successful, False otherwise
        """
        for i, file_path in enumerate(self.video_files):
            if filename.lower() in os.path.basename(file_path).lower():
                self.active_file_index = i
                print(f"Active file set to: {self.active_file}")
                return True
        
        print(f"File not found: {filename}")
        return False
        
    def list_video_files(self):
        """
        List all video files with their indices
        
        Returns:
            list: List of tuples (index, filename, is_active)
        """
        file_list = []
        for i, file_path in enumerate(self.video_files):
            is_active = (i == self.active_file_index)
            filename = os.path.basename(file_path)
            file_list.append((i, filename, is_active))
            print(f"{'*' if is_active else ' '} [{i}] {filename}")
        return file_list
        
    def switch_to_file(self, index, start_position=0):
        """
        Switch to a different video file by restarting VLC with same settings
        
        Args:
            index (int): Index of the file to switch to
            start_position (int): Position in seconds to start from (default: 0)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.set_active_file(index):
            return False
            
        if not self.last_launch_template:
            print("No launch template available. Launch VLC first.")
            return False
            
        print(f"Switching to: {os.path.basename(self.active_file)}")
        
        # Close current VLC instance if running
        if self.is_running():
            self.close()
            time.sleep(1)  # Brief pause to ensure clean shutdown
        
        # Create launch arguments with new file
        vlc_args = self.last_launch_template.copy()
        vlc_args[-1] = self.active_file  # Replace placeholder with actual file
        
        # Relaunch with new file
        success = self._launch_with_args(vlc_args)
        
        if success and start_position > 0:
            # Wait for VLC to fully load before seeking
            time.sleep(3)
            self.seek(start_position)
            
        return success

    def switch_to_file_advanced(self, index, start_position=0):
        """
        Alternative method: Try HTTP switching first, fallback to restart
        
        Args:
            index (int): Index of the file to switch to
            start_position (int): Position in seconds to start from (default: 0)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.set_active_file(index):
            return False
            
        if not self.is_running():
            print("VLC is not running. Launch first.")
            return False
        
        # Method 1: Try clearing playlist and adding new file
        print(f"Attempting HTTP switch to: {os.path.basename(self.active_file)}")
        
        # Stop current playback
        self._send_command("pl_stop")
        time.sleep(0.5)
        
        # Clear the playlist
        clear_success = self._send_command("pl_empty")
        time.sleep(0.5)
        
        # Add new file to playlist (with proper URL encoding)
        file_url = f"file://{self.active_file.replace(' ', '%20').replace('&', '%26')}"
        add_success = self._send_command(f"in_enqueue&input={file_url}")
        time.sleep(1)
        
        # Play the new file
        play_success = self._send_command("pl_play")
        time.sleep(2)
        
        # Check if it's actually playing
        status = self.get_video_status()
        
        if status and status.get('state') == 'playing':
            print("HTTP switching successful")
            if start_position > 0:
                self.seek(start_position)
            return True
        else:
            # Fallback: Restart VLC completely
            print("HTTP switching failed, restarting VLC...")
            return self.switch_to_file(index, start_position)

    def get_video_status(self):
        """
        Get current VLC status information
        
        Returns:
            dict: Status information or None if failed
        """
        if not self.is_running():
            return None
            
        try:
            response = requests.get(self.base_url, auth=self.auth, timeout=5)
            if response.status_code == 200:
                # Parse basic info from XML
                content = response.text
                status = {}
                
                # Extract basic status info
                if 'state="playing"' in content:
                    status['state'] = 'playing'
                elif 'state="paused"' in content:
                    status['state'] = 'paused'
                elif 'state="stopped"' in content:
                    status['state'] = 'stopped'
                else:
                    status['state'] = 'unknown'
                    
                return status
        except Exception as e:
            print(f"Failed to get status: {e}")
            return None

    def ensure_video_display(self):
        """
        Try to restore video display if only audio is playing
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("Attempting to restore video display...")
        
        # Try refreshing the video output
        refresh_success = self._send_command("key-toggle-fullscreen")
        time.sleep(0.5)
        refresh_success2 = self._send_command("key-toggle-fullscreen")
        
        return refresh_success and refresh_success2
        
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

    def test_vlc_installation(self):
        """
        Test if VLC is properly installed and can run
        
        Returns:
            bool: True if VLC works, False otherwise
        """
        if not self.vlc_path:
            print("VLC path not found")
            return False
            
        print(f"Testing VLC installation at: {self.vlc_path}")
        
        # Test basic VLC execution
        try:
            result = subprocess.run([self.vlc_path, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("VLC version check successful")
                # Extract version info
                version_line = result.stdout.split('\n')[0]
                print(f"VLC Version: {version_line}")
                return True
            else:
                print(f"VLC version check failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"VLC test failed: {e}")
            return False
        
    def _launch_with_args(self, vlc_args, capture_output=False):
        """
        Launch VLC with specific arguments
        
        Args:
            vlc_args (list): List of VLC arguments
            capture_output (bool): Whether to capture stderr for debugging
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Set comprehensive environment
            env = os.environ.copy()
            
            # Force X11 environment
            if 'DISPLAY' not in env:
                env['DISPLAY'] = ':0'
            env['GDK_BACKEND'] = 'x11'
            env['QT_QPA_PLATFORM'] = 'xcb'
            env['XDG_SESSION_TYPE'] = 'x11'
            
            # Disable Wayland
            if 'WAYLAND_DISPLAY' in env:
                del env['WAYLAND_DISPLAY']
                
            if capture_output:
                print(f"Full VLC command: {' '.join(vlc_args)}")
            
            if capture_output:
                # Capture output for debugging
                self.vlc_process = subprocess.Popen(vlc_args, env=env, 
                                                  stdout=subprocess.PIPE, 
                                                  stderr=subprocess.PIPE)
            else:
                # Normal launch
                self.vlc_process = subprocess.Popen(vlc_args, env=env, 
                                                  stdout=subprocess.DEVNULL, 
                                                  stderr=subprocess.DEVNULL)
            
            # Wait for VLC to start and check if it's running
            time.sleep(3)
            
            if self.is_running():
                print("VLC process started successfully")
                return True
            else:
                print("VLC process failed to start or exited immediately")
                if capture_output and self.vlc_process:
                    # Show any error output
                    try:
                        stdout, stderr = self.vlc_process.communicate(timeout=2)
                        if stderr:
                            print(f"VLC stderr: {stderr.decode()[:1000]}")
                        if stdout:
                            print(f"VLC stdout: {stdout.decode()[:1000]}")
                    except:
                        pass
                return False
            
        except Exception as e:
            print(f"Error launching VLC: {e}")
            return False
        
    def launch_minimal(self, fullscreen=True):
        """
        Launch VLC with absolute minimal arguments - no keyboard disabling
        
        Args:
            fullscreen (bool): Whether to launch in fullscreen mode
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.vlc_path:
            print("VLC not found. Install with: sudo apt install vlc")
            return False
            
        if not self.video_files:
            print("No video files available")
            return False
            
        if not os.path.exists(self.active_file):
            print(f"Active video file not found: {self.active_file}")
            return False
        
        # Absolute minimal VLC arguments
        vlc_args = [
            self.vlc_path,
            "--intf", "http",
            "--http-password", self.password,
            "--http-port", self.port,
        ]
        
        if fullscreen:
            vlc_args.append("--fullscreen")
            
        vlc_args.append(self.active_file)
        
        # Store template for file switching
        self.last_launch_template = vlc_args.copy()
        self.last_launch_template[-1] = "{ACTIVE_FILE}"
        
        print("Attempting minimal VLC launch (no keyboard disabling)...")
        success = self._launch_with_args(vlc_args, capture_output=True)
        
        if success:
            print(f"Minimal VLC launch successful")
            print(f"Active file: {os.path.basename(self.active_file)}")
        
        return success
        
    def launch_basic(self, fullscreen=True, disable_keyboard=False):
        """
        Launch VLC with basic arguments and optional keyboard disabling
        
        Args:
            fullscreen (bool): Whether to launch in fullscreen mode
            disable_keyboard (bool): Try to disable VLC keyboard shortcuts
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.vlc_path:
            print("VLC not found. Install with: sudo apt install vlc")
            return False
            
        if not self.video_files:
            print("No video files available")
            return False
            
        if not os.path.exists(self.active_file):
            print(f"Active video file not found: {self.active_file}")
            return False
        
        # Basic VLC arguments
        vlc_args = [
            self.vlc_path,
            "--intf", "http",
            "--http-password", self.password,
            "--http-port", self.port,
            "--no-qt-privacy-ask",
        ]
        
        # Add basic keyboard disabling options
        if disable_keyboard:
            vlc_args.extend([
                "--no-keyboard-events",
                "--no-mouse-events", 
                "--no-interact",
            ])
        
        if fullscreen:
            vlc_args.append("--fullscreen")
            
        vlc_args.append(self.active_file)
        
        # Store template for file switching
        self.last_launch_template = vlc_args.copy()
        self.last_launch_template[-1] = "{ACTIVE_FILE}"
        
        print(f"Attempting basic VLC launch (keyboard disabled: {disable_keyboard})...")
        success = self._launch_with_args(vlc_args, capture_output=True)
        
        if success:
            print(f"Basic VLC launch successful")
            print(f"Active file: {os.path.basename(self.active_file)}")
            if disable_keyboard:
                print("Basic keyboard disabling applied")
        
        return success
        
    def launch(self, fullscreen=True, wait_for_load=5, video_output="x11", disable_keyboard=True):
        """
        Launch VLC with full options
        
        Args:
            fullscreen (bool): Whether to launch in fullscreen mode
            wait_for_load (int): Seconds to wait for VLC to fully load
            video_output (str): Video output method ("x11", "fb", "dummy", "auto")
            disable_keyboard (bool): Disable VLC keyboard shortcuts
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.vlc_path:
            print("VLC not found. Install with: sudo apt install vlc")
            return False
            
        if not self.video_files:
            print("No video files available")
            return False
            
        if not os.path.exists(self.active_file):
            print(f"Active video file not found: {self.active_file}")
            return False
        
        # Build VLC arguments
        vlc_args = [
            self.vlc_path,
            "--intf", "http",
            "--http-password", self.password,
            "--http-port", self.port,
            "--no-qt-privacy-ask",
            "--no-video-title-show",
        ]
        
        # Add keyboard disabling - only essential ones to avoid startup failures
        if disable_keyboard:
            vlc_args.extend([
                "--no-keyboard-events",
                "--no-mouse-events",
                "--no-interact",
                # Only disable the most common shortcuts
                "--key-play-pause=",
                "--key-pause=", 
                "--key-play=",
                "--key-stop=",
                "--key-quit=",
            ])
        
        # Video output selection
        if video_output == "x11":
            vlc_args.extend([
                "--vout", "xcb_x11",
            ])
        elif video_output == "fb":
            vlc_args.extend([
                "--vout", "fb",
                "--fbdev", "/dev/fb0",
            ])
        elif video_output == "dummy":
            vlc_args.extend([
                "--vout", "dummy",
            ])
        
        if fullscreen:
            vlc_args.append("--fullscreen")
            
        vlc_args.append(self.active_file)
        
        # Store template for file switching
        self.last_launch_template = vlc_args.copy()
        self.last_launch_template[-1] = "{ACTIVE_FILE}"
        
        success = self._launch_with_args(vlc_args, capture_output=True)
        if success:
            print(f"VLC launched with {video_output} output, keyboard disabled: {disable_keyboard}")
            print(f"Active file: {os.path.basename(self.active_file)}")
        
        return success

    def launch_with_fallback(self, fullscreen=True, wait_for_load=5, disable_keyboard=True):
        """
        Launch VLC trying different methods progressively
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Try 1: Minimal launch (most compatible)
        print("Step 1: Trying minimal VLC launch (no keyboard disabling)...")
        if self.launch_minimal(fullscreen=fullscreen):
            time.sleep(2)
            if self.is_running():
                # Test HTTP interface
                for attempt in range(5):
                    if self._send_command("volume&val=100"):
                        print("Minimal launch successful - HTTP interface working")
                        print("NOTE: Keyboard shortcuts are still active")
                        return True
                    time.sleep(1)
                print("Minimal launch succeeded but HTTP interface not responding")
                self.close()
        
        # Try 2: Basic launch with keyboard disabling
        print("Step 2: Trying basic launch with keyboard disabling...")
        if self.launch_basic(fullscreen=fullscreen, disable_keyboard=disable_keyboard):
            time.sleep(2)
            if self.is_running():
                # Test HTTP interface
                for attempt in range(3):
                    if self._send_command("volume&val=100"):
                        print("Basic launch with keyboard disabling successful")
                        return True
                    time.sleep(1)
                print("Basic launch succeeded but HTTP interface not responding")
                self.close()
        
        # Try 3: Different video outputs
        output_methods = ["x11", "fb"]
        
        for method in output_methods:
            print(f"Step 3: Trying {method} video output...")
            
            if self.launch(fullscreen=fullscreen, wait_for_load=wait_for_load, 
                          video_output=method, disable_keyboard=disable_keyboard):
                time.sleep(2)
                if self.is_running():
                    # Test HTTP interface
                    for attempt in range(3):
                        if self._send_command("volume&val=100"):
                            print(f"Successfully launched with {method} output")
                            return True
                        time.sleep(1)
                    print(f"{method} output launched but HTTP interface not responding")
                    self.close()
            
            time.sleep(1)
        
        print("All launch methods failed")
        return False

    def launch_headless(self, wait_for_load=5, disable_keyboard=True):
        """
        Launch VLC in headless mode (audio only, no video display)
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.vlc_path:
            print("VLC not found. Install with: sudo apt install vlc")
            return False
            
        if not self.video_files:
            print("No video files available")
            return False
            
        if not os.path.exists(self.active_file):
            print(f"Active video file not found: {self.active_file}")
            return False
        
        # Headless VLC arguments
        vlc_args = [
            self.vlc_path,
            "--intf", "http",
            "--http-password", self.password,
            "--http-port", self.port,
            "--no-video",  # Disable video output completely
        ]
        
        if disable_keyboard:
            vlc_args.extend([
                "--no-keyboard-events",
                "--no-mouse-events",
                "--no-interact",
            ])
        
        vlc_args.append(self.active_file)
        
        self.last_launch_template = vlc_args.copy()
        self.last_launch_template[-1] = "{ACTIVE_FILE}"
        
        success = self._launch_with_args(vlc_args)
        if success:
            print(f"VLC launched in headless mode: {os.path.basename(self.active_file)}")
            if disable_keyboard:
                print("Keyboard shortcuts disabled")
        
        return success

    def check_display_environment(self):
        """
        Check and report the current display environment
        
        Returns:
            dict: Environment information
        """
        env_info = {
            'DISPLAY': os.environ.get('DISPLAY', 'Not set'),
            'WAYLAND_DISPLAY': os.environ.get('WAYLAND_DISPLAY', 'Not set'),
            'XDG_SESSION_TYPE': os.environ.get('XDG_SESSION_TYPE', 'Not set'),
            'GDK_BACKEND': os.environ.get('GDK_BACKEND', 'Not set'),
        }
        
        print("Current display environment:")
        for key, value in env_info.items():
            print(f"  {key}: {value}")
        
        # Check if X11 is available
        try:
            result = subprocess.run(['xset', 'q'], capture_output=True, text=True)
            env_info['X11_available'] = result.returncode == 0
        except:
            env_info['X11_available'] = False
        
        print(f"  X11 available: {env_info['X11_available']}")
        
        return env_info
    
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
            return False
            
        try:
            url = f"{self.base_url}?command={command}"
            response = requests.get(url, auth=self.auth, timeout=timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException:
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
        
    def seek_to_time(self, hours=0, minutes=0, seconds=0):
        """
        Seek to a specific time position
        
        Args:
            hours (int): Hours
            minutes (int): Minutes  
            seconds (int): Seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        total_seconds = hours * 3600 + minutes * 60 + seconds
        print(f"Seeking to {hours:02d}:{minutes:02d}:{seconds:02d} ({total_seconds} seconds)")
        return self.seek(total_seconds)

    def seek_percent(self, percent):
        """
        Seek to a percentage of the video
        
        Args:
            percent (float): Percentage (0-100)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if 0 <= percent <= 100:
            print(f"Seeking to {percent}% of video")
            return self._send_command(f"seek&val={percent}%")
        else:
            print("Percent must be between 0 and 100")
            return False
    
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
                self.vlc_process = None
                return True
            except subprocess.TimeoutExpired:
                print("Force killing VLC...")
                self.vlc_process.kill()
                self.vlc_process.wait()
                self.vlc_process = None
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

    def set_volume(self, volume):
        """
        Set the volume level
        
        Args:
            volume (int): Volume level (0-100 for percentage, or 0-512 for VLC's internal range)
                         VLC's internal range: 0=mute, 256=100%, 512=200%
                         Percentage range: 0=mute, 100=100%
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not isinstance(volume, (int, float)):
            print("Volume must be a number")
            return False
            
        # Convert to int if float
        volume = int(volume)
        
        # Validate range
        if volume < 0:
            print("Volume cannot be negative, setting to 0")
            volume = 0
        elif volume > 512:
            print("Volume too high, setting to 512 (200%)")
            volume = 512
            
        # Determine if using percentage (0-100) or VLC internal range (0-512)
        if volume <= 100:
            print(f"Setting volume to {volume}%")
        else:
            percentage = round((volume / 256) * 100)
            print(f"Setting volume to {volume} ({percentage}%)")
            
        return self._send_command(f"volume&val={volume}")
    
    def mute(self):
        """
        Mute the audio
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("Muting audio")
        return self.set_volume(0)
    
    def unmute(self, volume=256):
        """
        Unmute the audio and set to specified volume
        
        Args:
            volume (int): Volume to set when unmuting (default: 256 = 100%)
            
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"Unmuting audio to volume {volume}")
        return self.set_volume(volume)


# Example usage and testing
if __name__ == "__main__":
    # Example with multiple video files - UPDATE THESE PATHS
    video_files = [
        "/home/dieter/Videos/DeadlyPrey1988vhsrip_TryFile.com_.avi",
        "/home/dieter/Videos/The.Midnight.Meat.Train.R5.2008.avi",
    ]
    
    print("Testing VLC Controller with progressive launch methods...")
    vlc = VLCController(video_files)
    
    try:
        # Test VLC installation first
        if not vlc.test_vlc_installation():
            print("VLC installation test failed. Please check VLC installation.")
            exit(1)
        
        # Check display environment
        vlc.check_display_environment()
        
        # Show available files
        print("\nAvailable video files:")
        vlc.list_video_files()
        
        # Launch VLC with progressive fallback method
        if vlc.launch_with_fallback(disable_keyboard=True):
            vlc.set_volume(512)
            print("\nVLC launched successfully!")
            
            # Wait and then test file switching
            print("Playing first video for 15 seconds...")
            time.sleep(15)
            
            print("Testing file switching...")
            
            if len(video_files) > 1:
                print("Switching to second file...")
                if vlc.switch_to_file(1):  # Switch to second file
                    print("Successfully switched to second file")
                    time.sleep(10)  # Play for 10 seconds
                    
                    print("Switching back to first file...")
                    if vlc.switch_to_file(0):  # Switch back to first file
                        print("Successfully switched back to first file")
                    else:
                        print("Failed to switch back to first file")
                else:
                    print("Failed to switch to second file")
            
            # Keep running until user closes
            print("\nVLC is running. Press Ctrl+C to close VLC and exit.")
            print("Note: If keyboard shortcuts are still active, you can disable them")
            print("by manually pressing Ctrl+P in VLC to open preferences and disable hotkeys.")
            vlc.wait_until_closed()
        else:
            print("Failed to launch VLC with any method")
            print("\nTroubleshooting steps:")
            print("1. Check VLC installation:")
            print(f"   {vlc.vlc_path} --version")
            print("2. Try launching VLC manually:")
            print(f"   {vlc.vlc_path} --intf http --http-password vlcpassword --http-port 8080 {vlc.active_file}")
            print("3. Check if port 8080 is available:")
            print("   netstat -ln | grep 8080")
            print("4. Try different port:")
            print("   vlc = VLCController(video_files, port='8081')")
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        vlc.close()