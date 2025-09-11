import subprocess
import os
import time
import requests
import signal
import threading
import sounddevice as sd
from corefunctions.soundinput import MicrophoneAnalyzer

def test_device_access(device_id):
    """Test if a device can actually be opened for recording"""
    try:
        # Try to create a test stream
        test_stream = sd.InputStream(
            device=device_id,
            channels=1,
            samplerate=48000,
            blocksize=1024
        )
        test_stream.start()
        test_stream.stop()
        test_stream.close()
        return True
    except Exception as e:
        return False

def find_audio_device():
    """Find available audio input device on Raspberry Pi"""
    devices = sd.query_devices()
    
    # Common names for Raspberry Pi audio devices
    pi_audio_patterns = [
        'usb', 'card', 'alsa', 'bcm', 'headphones', 'analog',
        'pulse', 'default', 'sysdefault', 'hw:', 'plughw:'
    ]
    
    print("\nSearching for audio input device...")
    print("Available input devices:")
    
    working_candidates = []
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            device_name_lower = device['name'].lower()
            print(f"  Device ID {i}: {device['name']}")
            
            # Test if device actually works
            if test_device_access(i):
                status = "✓ Working"
                working_candidates.append((i, device))
                print(f"    {status}")
            else:
                status = "✗ Not accessible"
                print(f"    {status}")
    
    # Use first working device
    if working_candidates:
        device_id, device_info = working_candidates[0]
        print(f"\nSelected: {device_info['name']} (ID: {device_id})")
        return device_id
    
    print("\n" + "="*60)
    print("NO WORKING AUDIO INPUT DEVICE FOUND")
    print("="*60)
    print("\nTo use audio monitoring on Raspberry Pi:")
    print("\n1. Connect USB microphone or audio interface")
    print("2. Enable audio: sudo raspi-config → Advanced Options → Audio")
    print("3. Check ALSA devices: arecord -l")
    print("4. Install audio packages: sudo apt install alsa-utils pulseaudio")
    
    return None

def find_vlc_path():
    """Find VLC executable on Raspberry Pi"""
    possible_paths = [
        "/usr/bin/vlc",
        "/usr/local/bin/vlc",
        "/snap/bin/vlc"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Try which command
    try:
        result = subprocess.run(['which', 'vlc'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    
    return None

def launch_video_vlc_with_restart(video_path):
    """Launch VLC with HTTP interface, pause after 20s, then restart from beginning"""
    vlc_process = None
    
    try:
        # Find VLC executable
        vlc_path = find_vlc_path()
        
        if not vlc_path:
            print("VLC not found. Install with: sudo apt install vlc")
            return False
            
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return False
            
        # Launch VLC with HTTP interface enabled
        password = "vlcpassword"
        port = "8080"
        
        vlc_args = [
            vlc_path,
            "--intf", "http",
            "--http-password", password,
            "--http-port", port,
            "--fullscreen",
            "--extraintf", "rc",
            "--no-qt-privacy-ask",  # Skip privacy dialog
            "--no-video-title-show",  # Don't show filename
            video_path
        ]
        
        # Set display for GUI applications
        env = os.environ.copy()
        if 'DISPLAY' not in env:
            env['DISPLAY'] = ':0'
        
        vlc_process = subprocess.Popen(vlc_args, env=env)
        print(f"Launching VLC with HTTP interface on port {port}")
        print("Press Ctrl+C to close VLC at any time")
        
        # Wait for VLC to fully load
        time.sleep(5)
        
        # Base auth for VLC HTTP interface
        auth = ('', password)
        
        # Wait 20 seconds
        print("Video playing... waiting 20 seconds")
        time.sleep(20)
        
        # Pause the video
        print("Pausing video for 10 seconds")
        pause_url = f"http://localhost:{port}/requests/status.xml?command=pl_pause"
        try:
            requests.get(pause_url, auth=auth, timeout=5)
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not pause video via HTTP: {e}")
        
        # Wait 10 seconds while paused
        time.sleep(10)
        
        # Restart from beginning (seek to position 0)
        print("Restarting video from beginning")
        restart_url = f"http://localhost:{port}/requests/status.xml?command=seek&val=0"
        try:
            requests.get(restart_url, auth=auth, timeout=5)
            
            # Make sure it's playing
            play_url = f"http://localhost:{port}/requests/status.xml?command=pl_play"
            requests.get(play_url, auth=auth, timeout=5)
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not restart video via HTTP: {e}")
        
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

def monitor_audio_levels(analyzer, stop_event):
    """Monitor and print audio levels every second"""
    print("Starting audio monitoring...")
    
    while not stop_event.is_set():
        try:
            # Get both bass frequencies and overall sound level
            bass_level = analyzer.get_sound()
            overall_level = analyzer.get_all_sound()
            
            print(f"Audio Level - Bass: {bass_level:.3f}, Overall: {overall_level:.3f}")
        except Exception as e:
            print(f"Audio monitoring error: {e}")
        
        # Wait 1 second or until stop event is set
        if stop_event.wait(1.0):
            break
    
    print("Audio monitoring stopped.")

def run_video_with_audio_monitoring(video_path):
    """Run video with audio level monitoring"""
    
    # Find working audio device
    device_id = find_audio_device()
    
    if device_id is None:
        print("\nCannot proceed without a working audio device.")
        return False
    
    # Initialize audio analyzer
    print(f"\nInitializing audio analyzer...")
    try:
        analyzer = MicrophoneAnalyzer(device=device_id)
        analyzer.start()
        print("Audio analyzer started successfully!")
    except Exception as e:
        print(f"Failed to start audio analyzer: {e}")
        return False
    
    # Create stop event for audio monitoring
    stop_event = threading.Event()
    
    # Start audio monitoring in a separate thread
    audio_thread = threading.Thread(
        target=monitor_audio_levels, 
        args=(analyzer, stop_event)
    )
    audio_thread.daemon = True
    audio_thread.start()
    
    try:
        # Launch video (this will block until video is closed or interrupted)
        success = launch_video_vlc_with_restart(video_path)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user...")
        success = True
    finally:
        # Stop audio monitoring
        print("Stopping audio monitoring...")
        stop_event.set()
        analyzer.stop()
        audio_thread.join(timeout=2)
        
    return success

if __name__ == "__main__":
    # Update path for Raspberry Pi - adjust as needed
    video_file = "/home/dieter/Videos/DeadlyPrey1988vhsrip_TryFile.com_.avi"
    
    print("Starting video with audio monitoring on Raspberry Pi...")
    print("Audio levels will be printed every second.")
    print("Press Ctrl+C to stop at any time.")
    print("-" * 50)
    
    run_video_with_audio_monitoring(video_file)