import numpy as np
import time
from corefunctions.Events import EventScheduler
from sceneutils.GS_states import *  # noqa: F403
from sceneutils.OTO_emotions import *  # noqa: F403
from PyQt5.QtWidgets import QApplication 
from corefunctions.soundinput import MicrophoneAnalyzer
 # noqa: F403
#
class EnvironmentalSystem:
    def __init__(self, scheduler):
        self.scheduler = scheduler

        self.transition_time = 0
        self.transition_start = 0
        self.progress = 0
        self.analyzer = MicrophoneAnalyzer()
        self.analyzer.start()
        self.speed=0.995
        self.sound_hist=np.zeros(100)
        # Store the latest input values

        # Set smoothing factor (0-1, lower values mean more smoothing)
        self.smoothing_factor = 0.08
        
        # Track last update time for consistent smoothing
        self.last_update_time = time.time()



    def send_variables(self):
        """Send variables to scheduler with exponential smoothing"""
        self.scheduler.state["time"] = self.current_time
        
        # Get the latest control values

        
        # Calculate elapsed time since last update for time-based smoothing
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now

            

    def get_whomp(self):
        thresh = 1.0
        maxsound = 6
        # loud = self.analyzer.get_sound()
        loud = self.analyzer.get_all_sound()
        swloud = (loud > thresh) * 1
        self.whomp = swloud * (np.clip(loud, 0, maxsound) - thresh) / (maxsound - thresh)



    def update(self):
        """Update the environmental system - should be called each frame"""

        self.current_time = time.time()
        self.get_whomp()
            
        # Apply current parameters to scheduler state
        self.send_variables()
        
        
        # Update the scheduler
        self.scheduler.update()
        
        if QApplication.instance():
            QApplication.instance().processEvents()



# Main execution
if __name__ == "__main__":
    scheduler = EventScheduler()
    env_system = EnvironmentalSystem(scheduler)
    scheduler.setup_visualizer(False) 
    # Start with summer bloom weather
   
    curr = 0
    def schedule_next(seconds, func, overlap=0):
        global curr
        end = curr + seconds
        beginning = curr
        if overlap and curr != 0:
            beginning = beginning - overlap
            end = end + overlap
        #print(f"Scheduling {func.__name__} {beginning} => {end}")
        # comment this out so we can do it without the intervening function
        env_system.scheduler.schedule_event(beginning, end, func)
        print(f"env_system.scheduler.schedule_event({beginning}, {end}, {func.__name__})")
        curr = end
    def get_active_states():
        return ", ".join(event.action.__name__ for event in scheduler.active_events)
    funcs = [GS_tingles, GS_rage_lightning, GS_curious_playful, GS_forest,
                 GS_blood_flow, GS_blink_fade, GS_hypnotic_spiral, GS_rage_lightning, GS_shibari,
                 GS_sunrise]
    f2 = list(funcs)
    f3 = list(funcs)
    import random
    random.shuffle(funcs)
    random.shuffle(f2)
    random.shuffle(f3)
    #for func in funcs + f2 + f3:
    #    schedule_next(10, func, 1)
    # schedule_next(30,GS_tingles)  # noqa: F405
    # schedule_next(40, GS_hot_tub, 10)  # noqa: F405
    schedule_next(60, GS_tingles)
    schedule_next(60, GS_rage_lightning)
    schedule_next(60, GS_blink_fade)
    schedule_next(60, GS_sunrise)
    schedule_next(60, GS_tingles)
    schedule_next(60, GS_rage_lightning)
    schedule_next(60, GS_blink_fade)
    schedule_next(60, GS_sunrise)
    schedule_next(60, GS_tingles)
    schedule_next(60, GS_rage_lightning)
    schedule_next(60, GS_blink_fade)
    schedule_next(60, GS_sunrise)
    #env_system.scheduler.schedule_event(0, 10, GS_tingles)
    #env_system.scheduler.schedule_event(10, 20, GS_rage_lightning)
    #env_system.scheduler.schedule_event(20, 30, GS_blink_fade)
    #env_system.scheduler.schedule_event(30, 40, GS_sunrise)
    #env_system.scheduler.schedule_event(40, 50, GS_tingles)
    #env_system.scheduler.schedule_event(50, 60, GS_rage_lightning)
    #env_system.scheduler.schedule_event(60, 70, GS_blink_fade)
    #env_system.scheduler.schedule_event(70, 80, GS_sunrise)
    #env_system.scheduler.schedule_event(80, 90, GS_tingles)
    #env_system.scheduler.schedule_event(90, 100, GS_rage_lightning)
    #env_system.scheduler.schedule_event(100, 110, GS_blink_fade)
    #env_system.scheduler.schedule_event(110, 120, GS_sunrise)
    #env_system.scheduler.schedule_event(0, 60, GS_hypnotic_spiral)
    #env_system.scheduler.schedule_event(0, 60, GS_forest)
    #env_system.scheduler.schedule_event(0, 300, GS_tingles)
    #env_system.scheduler.schedule_event(0, 600, GS_shibari)
    lasttime = time.perf_counter()
    FRAME_TIME = 1 / 40
    first_time = time.perf_counter()
    try:
        while True:
            # Update environmental system
            env_system.update()

            current_time = time.perf_counter()
            
            elapsed = current_time - lasttime
            sleep_time = max(0, FRAME_TIME - elapsed)
            time.sleep(sleep_time)

            # Print stats if needed
            print(["%.2f" % (1/(time.perf_counter()-lasttime)), "%.2f" % len(scheduler.active_events), len(scheduler.event_queue),"%.3f" %((lasttime-first_time)/3600), get_active_states()])
            lasttime = time.perf_counter()

    except KeyboardInterrupt:
        print("Done!")

