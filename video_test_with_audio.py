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

def find_system_audio_device():
    """Find the system audio output device with proper testing"""
    devices = sd.query_devices()
    
    # Common names for system audio capture devices
    system_audio_patterns = [
        'stereo mix', 'what u hear', 'wave out mix', 'sum', 
        'loopback', 'wasapi', 'speakers', 'headphones',
        'primary sound capture', 'wave', 'realtek'
    ]
    
    print("\nSearching for system audio capture device...")
    print("Available input devices:")
    
    working_candidates = []
    failed_candidates = []
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            device_name_lower = device['name'].lower()
            print(f"  Device ID {i}: {device['name']}")
            
            # Test if device actually works
            if test_device_access(i):
                status = "✓ Working"
                # Check if this device matches system audio patterns
                for pattern in system_audio_patterns:
                    if pattern in device_name_lower:
                        working_candidates.append((i, device, pattern))
                        print(f"    ^ System audio device (matches '{pattern}') - {status}")
                        break
                else:
                    print(f"    {status}")
            else:
                status = "✗ Not accessible"
                print(f"    {status}")
                # Still add to failed candidates for reference
                for pattern in system_audio_patterns:
                    if pattern in device_name_lower:
                        failed_candidates.append((i, device, pattern))
                        break
    
    # Try working candidates first
    if working_candidates:
        # Prefer Stereo Mix or WASAPI loopback if available
        for device_id, device_info, pattern in working_candidates:
            if 'stereo mix' in pattern or 'loopback' in pattern:
                print(f"\nSelected: {device_info['name']} (ID: {device_id})")
                return device_id
        
        # Otherwise, use the first working candidate
        device_id, device_info, pattern = working_candidates[0]
        print(f"\nSelected: {device_info['name']} (ID: {device_id})")
        return device_id
    
    # If no working system audio devices, show what we found but couldn't access
    if failed_candidates:
        print(f"\nFound system audio devices but they're not accessible:")
        for device_id, device_info, pattern in failed_candidates:
            print(f"  - {device_info['name']} (ID: {device_id})")
    
    print("\n" + "="*60)
    print("NO WORKING SYSTEM AUDIO CAPTURE DEVICE FOUND")
    print("="*60)
    print("\nTo monitor system audio, you need one of these solutions:")
    print("\n1. ENABLE STEREO MIX (Windows built-in):")
    print("   - Right-click sound icon → Open Sound settings")
    print("   - Click 'Sound Control Panel'")
    print("   - Recording tab → Right-click → 'Show Disabled Devices'")
    print("   - Right-click 'Stereo Mix' → Enable")
    print("   - Right-click 'Stereo Mix' → Set as Default Device")
    
    print("\n2. INSTALL VB-AUDIO VIRTUAL CABLE:")
    print("   - Download from: https://vb-audio.com/Cable/")
    print("   - Install and restart")
    print("   - Set 'CABLE Output' as your default playback device")
    print("   - This script will use 'CABLE Input' to monitor")
    
    print("\n3. USE VOICEMEETER (Advanced):")
    print("   - Download from: https://vb-audio.com/Voicemeeter/")
    print("   - Provides virtual audio mixing with loopback")
    
    print("\nWould you like to try using the default microphone instead? (y/n)")
    
    # Let user decide to continue with microphone or exit
    try:
        choice = input().lower().strip()
        if choice in ['y', 'yes']:
            default_device = sd.default.device[0]
            if test_device_access(default_device):
                print(f"Using default microphone: {devices[default_device]['name']}")
                return default_device
            else:
                print("Default microphone also not accessible.")
                return None
        else:
            return None
    except:
        return None

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
            "--extraintf", "rc",
            video_path
        ]
        
        vlc_process = subprocess.Popen(vlc_args)
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
    device_id = find_system_audio_device()
    
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
    video_file = r"C:\Users\diete\Desktop\devel-local\BM.mp4"
    
    print("Starting video with audio monitoring...")
    print("Audio levels will be printed every second.")
    print("Press Ctrl+C to stop at any time.")
    print("-" * 50)
    
    run_video_with_audio_monitoring(video_file)