import numpy as np
import time
from corefunctions.Events import EventScheduler
from sceneutils.OTO_dev import *  # noqa: F403
from sceneutils.OTO_inactive import *  # noqa: F403
from sceneutils.OTO_emotions import *  # noqa: F403
from corefunctions.input_panel import InputPanel 
from PyQt5.QtWidgets import QApplication 
 # noqa: F403
#
class EnvironmentalSystem:
    def __init__(self, scheduler):
        self.scheduler = scheduler

        self.transition_time = 0
        self.transition_start = 0
        self.progress = 0
        #self.input_panel = InputPanel(title="OTO Control Panel")
        #self.input_panel.values_changed.connect(self.on_input_values_changed)
        self.raw_values={"Danger":0,"Joyful":0,"Sad":0,"Angry":0,"Curious":0,"Passionate":0,"Rage":0,"Contemplative":0,"Neutral":0}
        self.utility_names={"speed":1,"sound_in":0,"pad_left":0,"pad_right":0}
        self.state={"is_thinking":0,"is_listening":0,"is_speaking":0,"has_conversation":0}
        self.speed=0.995
        self.sound_hist=np.zeros(100)
        # Store the latest input values
        #self.input_values = self.input_panel.get_values()

                # Initialize smoothed values dictionary to track and smooth values
        self.smoothed_values = self.raw_values.copy()
        
        # Set smoothing factor (0-1, lower values mean more smoothing)
        self.smoothing_factor = 0.08
        
        # Track last update time for consistent smoothing
        self.last_update_time = time.time()

    def on_input_values_changed(self, values):
        """Handle updates from the input panel"""
        self.input_values = values
        
        # You can do immediate processing of values here if needed
        # For example, apply intensity to global settings
        
        # Pass input values to scheduler state

        

    def get_control_values(self):
        """Return the current control values from the input panel"""
        return self.input_panel.get_values()


    def send_variables(self):
        """Send variables to scheduler with exponential smoothing"""
        self.scheduler.state["time"] = self.current_time
        
        # Get the latest control values
        #raw_values = self.input_panel.get_values()
        
        # Calculate elapsed time since last update for time-based smoothing
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now
        
        # Adjust smoothing factor based on time elapsed to ensure consistent behavior
        # regardless of frame rate
        frame_smoothing = min(1.0, self.smoothing_factor * dt * 60)  # Normalize to 60fps
        
        # Apply exponential smoothing and send to scheduler
        for key, raw_value in self.raw_values.items():
            
            # Handle actual selected_mode and selected_effect without smoothing
            if key in ['selected_mode', 'selected_effect']:
                self.smoothed_values[key] = raw_value
                self.scheduler.state[f"control_{key}"] = raw_value
                
                continue
                
            # For all numeric values (including mode_X and effect_X values), apply smoothing
            if key in self.smoothed_values:
                current_smooth = self.smoothed_values[key]
                smooth_value = current_smooth + frame_smoothing * (raw_value - current_smooth)
                self.smoothed_values[key] = smooth_value
            else:
                # Initialize if not exists
                smooth_value = raw_value
                self.smoothed_values[key] = smooth_value
            
            # Store smoothed value in scheduler state
            if isinstance(smooth_value, (int, float)) and not isinstance(smooth_value, bool):
                self.scheduler.state[f"control_{key}"] = round(smooth_value, 3)
            else:
                self.scheduler.state[f"control_{key}"] = smooth_value
            


    def get_control_values(self):  # noqa: F811
        """Return the current raw control values from the input panel"""
        return self.input_panel.get_values()
    
    def get_smoothed_values(self):
        """Return the current smoothed values"""
        return self.smoothed_values.copy()
    
    def set_smoothing_factor(self, factor):
        """Set the smoothing factor (0-1, lower means more smoothing)"""
        self.smoothing_factor = max(0.0, min(1.0, factor))                


    def parse_osc_messages(self):
        """Parse OSC messages and update raw_values with the OSC address as key"""
        messages = self.scheduler.get_osc_messages()
        if messages != []:
                    
            #print(messages) 
            for message in messages:
                type=message[0].split('/')[1]
                address = message[0].split('/')[2]  # The OSC address (e.g., '/control/parameter1')
                value = float(message[1][0]) # The value sent with the message
                #print(value)
                if type == "state":
                    self.state[type]=value
                if type=="emotions":
                    self.raw_values[type]=value*10.0
                if address in self.utility_names:
                    if address=="sound_in":
                        np.roll(self.sound_hist,1)
                        self.sound_hist[0]=value
                    continue
                # Update raw_values dictionary with the OSC address as key
                #self.raw_values.append({address: value})
                #self.raw_values[address]=value
                # Optional: Print the received OSC message for debugging
                #print(f"Received OSC: {address} = {value}")
        #print(self.raw_values)
        for key in self.raw_values:
            self.smoothed_values[key]=self.smoothed_values.get(key,0)*self.speed+self.raw_values[key]*(1-self.speed)
            self.scheduler.state[f"control_{key}"]=self.smoothed_values[key]
        active=0
        for key in self.state:
            active+=self.state[key]
            self.smoothed_values[key]=self.smoothed_values.get(key,0)*self.speed+self.state[key]*(1-self.speed)
            self.scheduler.state[f"control_{key}"]=self.smoothed_values[key]

        if active>0:
            self.scheduler.state["control_active"]=1
        else:
            self.scheduler.state["control_active"]=0
        return self.raw_values

    def update(self):
        """Update the environmental system - should be called each frame"""

        self.current_time = time.time()
        self.parse_osc_messages()
            
        # Apply current parameters to scheduler state
        #self.send_variables()
        
        
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
   
    env_system.scheduler.schedule_event(0, 800000000, OTO_heartbeat)# noqa: F405
    env_system.scheduler.schedule_event(00, 800000000, OTO_danger_pulse)  # noqa: F405
    env_system.scheduler.schedule_event(00, 800000000, OTO_sunrise_joy) # noqa: F405
    env_system.scheduler.schedule_event(00, 800000000, OTO_sad_theme) # noqa: F405
    env_system.scheduler.schedule_event(00, 800000000, OTO_angry_theme) # noqa: F405
    env_system.scheduler.schedule_event(00, 800000000, OTO_curious_playful) # noqa: F405
    env_system.scheduler.schedule_event(00, 800000000, OTO_passionate_floral) # noqa: F405
    env_system.scheduler.schedule_event(00, 800000000, OTO_rage_lightning) # noqa: F405
    env_system.scheduler.schedule_event(00, 800000000, OTO_contemplative_cosmic) # noqa: F405
    env_system.scheduler.schedule_event(00, 800000000, OTO_neutral_positive) # noqa: F405
    env_system.scheduler.schedule_event(00, 800000000, OTO_inactive_pattern_cycle) # noqa: F405
    env_system.scheduler.schedule_event(00, 800000000, OTO_awaken) # noqa: F405
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
            #print(["%.2f" % (1/(time.perf_counter()-lasttime)), "%.2f" % len(scheduler.active_events), len(scheduler.event_queue),"%.3f" %((lasttime-first_time)/3600)])
            lasttime = time.perf_counter()

    except KeyboardInterrupt:
        print("Done!")

