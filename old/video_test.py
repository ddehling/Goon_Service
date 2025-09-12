import subprocess
import os
import time
import requests
import signal

def launch_video_vlc_with_restart(video_path):
    """Launch VLC with HTTP interface, pause after 20s, then restart from beginning"""
    vlc_process = None
    
    try:
        # VLC executable path
        vlc_path = r"C:\Program Files\VideoLAN\VLC\vlc.exe"
        
        if not os.path.exists(vlc_path):
            vlc_path = r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe"
            
        if not os.path.exists(vlc_path):
            print("VLC not found. Please check your VLC installation path.")
            return False
            
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return False
            
        # Launch VLC with HTTP interface enabled, fullscreen, and password
        password = "vlcpassword"
        port = "8080"
        
        vlc_args = [
            vlc_path,
            "--intf", "http",
            "--http-password", password,
            "--http-port", port,
            "--fullscreen",
            "--extraintf", "rc",  # Enable remote control interface too
            video_path
        ]
        
        vlc_process = subprocess.Popen(vlc_args)
        print(f"Launching VLC with HTTP interface on port {port}")
        print("Press Ctrl+C to close VLC at any time")
        
        # Wait for VLC to fully load
        time.sleep(5)
        
        # Base auth for VLC HTTP interface
        auth = ('', password)  # VLC uses empty username
        
        # Wait 20 seconds
        print("Video playing... waiting 20 seconds")
        time.sleep(20)
        
        # Pause the video
        print("Pausing video for 10 seconds")
        pause_url = f"http://localhost:{port}/requests/status.xml?command=pl_pause"
        requests.get(pause_url, auth=auth)
        
        # Wait 10 seconds while paused
        time.sleep(10)
        
        # Restart from beginning (seek to position 0)
        print("Restarting video from beginning")
        restart_url = f"http://localhost:{port}/requests/status.xml?command=seek&val=0"
        requests.get(restart_url, auth=auth)
        
        # Make sure it's playing
        play_url = f"http://localhost:{port}/requests/status.xml?command=pl_play"
        requests.get(play_url, auth=auth)
        
        # Keep the script running so VLC stays open
        print("Video restarted. Press Ctrl+C to close VLC.")
        while vlc_process.poll() is None:
            time.sleep(1)
        
        return True
        
    except KeyboardInterrupt:
        print("\nClosing VLC...")
        if vlc_process:
            vlc_process.terminate()
            vlc_process.wait()
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        if vlc_process:
            vlc_process.terminate()
        return False

# Launch your video with restart functionality
video_file = r"C:\Users\diete\Desktop\devel-local\BM.mp4"
launch_video_vlc_with_restart(video_file)