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

class EnvironmentalSystem:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.transition_time = 0
        self.transition_start = 0
        self.progress = 0
        self.video_files = [
            '/home/dieter/Videos/GoonService_SnailSequence_FINAL_111325_903PM.mov',
            '/home/dieter/Videos/GoonService_KittySequence_FINAL_111325_844P.mov',
        ]
        self.VLC = VLCController(self.video_files)
        self.speed = 0.995
        self.sound_hist = np.zeros(100)
        self.smoothing_factor = 0.08
        self.last_update_time = time.time()
        
        # Improved keyboard control
        self.key_pressed = False
        self.key_lock = threading.Lock()
        self.listening_for_key = False
        self.setup_global_hotkeys()

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
        """Schedule unique events based on the video number"""
        # Clear any existing scheduled events
        self.scheduler.cancel_all_events()
        
        if vidnum == 0:  # First (snail) video events
            self.scheduler.schedule_event(0, 149, GS_blood_flow)
            self.scheduler.schedule_event(161, 187, GS_hypnotic_spiral)
            self.scheduler.schedule_event(182, 290, GS_curious_playful)
            self.scheduler.schedule_event(285, 442, GS_blink_fade)
            self.scheduler.schedule_event(437, 564, GS_shibari)
            self.scheduler.schedule_event(559, 775, GS_forest)
            self.scheduler.schedule_event(770, 785, GS_tingles)
            self.scheduler.schedule_event(780, 900, GS_curious_playful)
            print("Scheduled events for Video 1 (Snail sequence)")
            
        elif vidnum == 1:  # Second (kitty) video events
            self.scheduler.schedule_event(0, 149, GS_blood_flow)
            self.scheduler.schedule_event(161, 185, GS_hypnotic_spiral)
            self.scheduler.schedule_event(180, 361, GS_hot_tub)
            self.scheduler.schedule_event(356, 556, GS_curious_playful)
            self.scheduler.schedule_event(554, 799, GS_forest)
            self.scheduler.schedule_event(800, 815, GS_tingles)
            self.scheduler.schedule_event(810, 900, GS_curious_playful)
            print("Scheduled events for Video 2 (Kitty sequence)")
            

# Main execution
if __name__ == "__main__":
    scheduler = EventScheduler()
    env_system = EnvironmentalSystem(scheduler)
    scheduler.setup_visualizer(False) 

    lasttime = time.perf_counter()
    FRAME_TIME = 1 / 40
    first_time = time.perf_counter()
    vidlength = [13.5* 60 + 52, 14.5 * 60 + 17] # first is snail, second is kitty. Have to add 30 seconds
    vidnum = 0
    numvid = len(vidlength)
    
    try:
        env_system.VLC.launch_with_fallback(disable_keyboard=True, autoplay=False)
        env_system.VLC.set_volume(256)
        
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