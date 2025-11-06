import numpy as np
import time
from corefunctions.Events import EventScheduler
from sceneutils.GS_states import *  # noqa: F403
from PyQt5.QtWidgets import QApplication 
from corefunctions.soundinput import MicrophoneAnalyzer
from corefunctions.vlc_control import VLCController
import sys
from pynput import keyboard
import threading
import yaml
from pathlib import Path

class EnvironmentalSystem:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.transition_time = 0
        self.transition_start = 0
        self.progress = 0
        self.video_files = [
            "/home/dieter/Videos/2025.10.02 NatureSmut - Edit.mp4",
            "/home/dieter/Videos/GoonService_KittySequence.mov",
        ]
        self.VLC = VLCController(self.video_files)
        self.speed = 0.995
        self.sound_hist = np.zeros(100)
        self.smoothing_factor = 0.08
        self.last_update_time = time.time()
        
        # Load video sequences from YAML
        self.video_sequences = self.load_video_sequences()
        
        # Improved keyboard control
        self.key_pressed = False
        self.key_lock = threading.Lock()
        self.listening_for_key = False
        self.setup_global_hotkeys()

    def load_video_sequences(self, yaml_file_path="video_sequences.yaml"):
        """Load video sequences configuration from YAML file"""
        yaml_path = Path(yaml_file_path)
        if not yaml_path.exists():
            print(f"Warning: {yaml_file_path} not found, using default configuration")
            return self.get_default_sequences()
        
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded video sequences from {yaml_file_path}")
            return config.get('videos', {})
        except Exception as e:
            print(f"Error loading {yaml_file_path}: {e}")
            print("Using default configuration")
            return self.get_default_sequences()
    
    def get_default_sequences(self):
        """Fallback default sequences if YAML file is not available"""
        return {
            0: {
                "name": "Default Video 1",
                "events": [
                    {"start": 0, "end": 15, "lighting": "GS_forest"},
                    {"start": 10, "end": 15, "lighting": "GS_tingles"}
                ]
            },
            1: {
                "name": "Default Video 2", 
                "events": [
                    {"start": 0, "end": 20, "lighting": "GS_blood_flow"}
                ]
            }
        }

    def setup_global_hotkeys(self):
        """Setup global hotkeys that work even when VLC has focus"""
        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char == 'q':
                    with self.key_lock:
                        # Only register key press if we're actively listening
                        if self.listening_for_key and not self.key_pressed:
                            self.key_pressed = True
                            print("Key 'q' detected!")
            except:
                pass

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()

    def start_listening_for_key(self):
        """Start listening for key presses and clear any previous state"""
        with self.key_lock:
            self.key_pressed = False  # Clear any previous key press
            self.listening_for_key = True
            print("Now listening for 'q' key press...")

    def stop_listening_for_key(self):
        """Stop listening for key presses"""
        with self.key_lock:
            self.listening_for_key = False

    def check_key_press(self):
        """Check if 'q' was pressed while we were listening"""
        with self.key_lock:
            if self.key_pressed:
                self.key_pressed = False  # Reset for next time
                return 'q'
            return None

    def send_variables(self):
        """Send variables to scheduler with exponential smoothing"""
        self.scheduler.state["time"] = self.current_time
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now

    def update(self):
        """Update the environmental system - should be called each frame"""
        self.current_time = time.time()
        self.send_variables()
        self.scheduler.update()
        
        if QApplication.instance():
            QApplication.instance().processEvents()

    def switch_to_video(self, vidnum):
        """Switch to video and ensure fullscreen state"""
        self.VLC.switch_to_file(vidnum)
        # Give VLC a moment to load the file
        time.sleep(.05)
        
        # Ensure fullscreen - toggle twice to guarantee fullscreen state
        self.VLC._send_command("fullscreen")
        time.sleep(0.03)
        self.VLC._send_command("fullscreen")
        
        print(f"Switched to video {vidnum + 1} - Fullscreen enforced")

    def schedule_video_events(self, vidnum):
        """Schedule events based on the video number with manual sequences"""
        # Clear any existing scheduled events
        self.scheduler.cancel_all_events()
        
        if vidnum == 0:  # First video - Deadly Prey
            self.scheduler.schedule_event(0, 15, GS_forest)
            self.scheduler.schedule_event(10, 15, GS_tingles)
            print("Scheduling events for Deadly Prey:")
            print("  - 0s-15s: GS_forest")
            print("  - 10s-15s: GS_tingles")
            
        elif vidnum == 1:  # Second video - Kitty Sequence
            self.scheduler.schedule_event(0, 194, GS_blink_fade)      # opener
            self.scheduler.schedule_event(194, 346, GS_hot_tub)
            self.scheduler.schedule_event(362, 538, GS_hypnotic_spiral)
            self.scheduler.schedule_event(547, 842, GS_forest)
            print("Scheduling events for Kitty Sequence:")
            print("  - 0s-194s: GS_blink_fade")
            print("  - 194s-346s: GS_hot_tub")
            print("  - 362s-538s: GS_hypnotic_spiral")
            print("  - 547s-842s: GS_forest")
            
        else:
            print(f"No events configured for video {vidnum}")
    
    def get_lighting_function(self, lighting_name):
        """Convert lighting function name string to actual function"""
        try:
            # Get the function from the global namespace (imported from GS_states)
            return globals().get(lighting_name)
        except Exception as e:
            print(f"Error getting lighting function '{lighting_name}': {e}")
            return None
            

# Main execution
if __name__ == "__main__":
    scheduler = EventScheduler()
    env_system = EnvironmentalSystem(scheduler)
    scheduler.setup_visualizer(False) 

    lasttime = time.perf_counter()
    FRAME_TIME = 1 / 40
    first_time = time.perf_counter()
    vidlength = [15, 15 * 60 + 5]
    vidnum = 1
    numvid = len(vidlength)
    
    try:
        env_system.VLC.launch_with_fallback(disable_keyboard=True, autoplay=False)
        env_system.VLC.set_volume(512)
        
        while True:
            # Switch to video and ensure fullscreen
            env_system.switch_to_video(vidnum)

            loopstart = time.time()
            #env_system.scheduler.schedule_event(0, 15, GS_rage_lightning)
            env_system.schedule_video_events(vidnum)

            video_finished = False  # Flag to break out of inner loop

            while True:
                env_system.update()
                current_time = time.perf_counter()
                elapsed = current_time - lasttime
                sleep_time = max(0, FRAME_TIME - elapsed)
                time.sleep(sleep_time)
                lasttime = time.perf_counter()
                
                if time.time() - loopstart > vidlength[vidnum]:
                    env_system.VLC.pause()
                    
                    # Start listening for fresh key press
                    env_system.start_listening_for_key()
                    env_system.scheduler.schedule_event(0, 15000000, GS_blink_fade)
                    while True:
                        key = env_system.check_key_press()
                        env_system.update()
                        current_time = time.perf_counter()
                        elapsed = current_time - lasttime
                        sleep_time = max(0, FRAME_TIME - elapsed)
                        time.sleep(sleep_time)
                        lasttime = time.perf_counter()
                        if key == 'q':
                            env_system.stop_listening_for_key()
                            vidnum = (vidnum + 1) % numvid
                            print(f"Switching to video {vidnum + 1}")
                            video_finished = True  # Set flag to break outer loop
                            break
                       
                    
                    # Break out of the inner playback loop
                    if video_finished:
                        break

    except KeyboardInterrupt:
        print("Done!")
    finally:
        env_system.listener.stop()