from sceneutils.imgutils import *
import numpy as np
from pathlib import Path

ParentPath = Path(__file__).parent.parent
media_path = ParentPath / 'media'
sound_path = media_path / 'sounds'

def OTO_blink(instate, outstate):
    """
    Generator that randomly selects pixels to light up and then fades all pixels.
    Each frame, 10 new pixels are activated and all pixels fade by 5%.
    Vectorized implementation for efficiency.
    """
    name = 'blink'
    buffers = outstate['buffers']

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        return

    if instate['count'] == -1:
        buffers.generator_alphas[name] = 0
        return
    remaining_time = instate['duration'] - instate['elapsed_time']
    
    # If less than 10 seconds remain, fade the generator alpha
    if remaining_time < 10.0:
        # Linear fade from 1.0 to 0.0 over 10 seconds
        generator_alpha = remaining_time / 10.0
        # Ensure we don't go below 0
        generator_alpha = max(0.0, generator_alpha)
        
        # Update the generator's alpha value
        buffers.generator_alphas[name] = generator_alpha
    else:
        # Full intensity
        buffers.generator_alphas[name] = 1.0

    # Get all strip buffers for our generator
    all_buffers = buffers.get_all_buffers(name)
    
    # First, fade all existing pixels by 5%
    fade_factor = 0.95  # 5% reduction per frame
    
    for strip_id, buffer in all_buffers.items():
        # Vectorized fade: multiply alpha channel by fade factor
        buffer[:, 3] *= fade_factor
    
    # Add 10 new random pixels
    for _ in range(10):
        # Select a random strip
        strip_id = np.random.choice(list(all_buffers.keys()))
        buffer = all_buffers[strip_id]
        
        # Select a random pixel
        pixel_idx = np.random.randint(0, len(buffer))
        
        # Generate a random color (RGB) with full opacity
        random_color = np.append(np.random.random(3), 1.0)
        
        # Set the pixel in the buffer
        buffer[pixel_idx] = random_color





# ... existing code ...

def OTO_point_traveler(instate, outstate):
    """
    Generator that illuminates a single point moving down the length of left_wall.
    The point has a random color and pixels fade slowly after the point passes.
    """
    name = 'point_traveler'
    buffers = outstate['buffers']

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        # Initialize position at the start of the strip
        instate['position'] = 0
        # Generate a random color for this run
        instate['color'] = np.append(np.random.random(3), 1.0)
        return

    if instate['count'] == -1:
        buffers.generator_alphas[name] = 0
        return

    remaining_time = instate['duration'] - instate['elapsed_time']
    
    # If less than 10 seconds remain, fade the generator alpha
    if remaining_time < 10.0:
        # Linear fade from 1.0 to 0.0 over 10 seconds
        generator_alpha = remaining_time / 10.0
        # Ensure we don't go below 0
        generator_alpha = max(0.0, generator_alpha)
        
        # Update the generator's alpha value
        buffers.generator_alphas[name] = generator_alpha
    else:
        # Full intensity
        buffers.generator_alphas[name] = 1.0

    # Get all strip buffers for our generator
    all_buffers = buffers.get_all_buffers(name)
    
    # First, fade all existing pixels by a small amount (slower decay)
    fade_factor = 0.98  # 2% reduction per frame
    
    for strip_id, buffer in all_buffers.items():
        # Vectorized fade: multiply alpha channel by fade factor
        buffer[:, 3] *= fade_factor
    
    # Focus on the left_wall strip
    if 'right_spine' in all_buffers:
        buffer = all_buffers['right_spine']
        strip_length = len(buffer)
        
        # Set the current position to the current color
        if instate['position'] < strip_length:
            buffer[instate['position']] = instate['color']
        
        # Move the position for the next frame
        instate['position'] += 1
        
        # If we've reached the end, reset position and choose a new color
        if instate['position'] >= strip_length:
            instate['position'] = 0
            instate['color'] = np.append(np.random.random(3), 1.0)

# ... existing code ...

def OTO_heartbeat(instate, outstate):
    """
    Generator that creates a heartbeat pattern on objects tagged with the 'heart' category.
    Objects tagged with 'left' and 'right' groups will pulse in alternating phases.
    For circular strips, creates a rotating wave pattern that follows the circle.
    Heart rate and strength are configurable, and color can be varied.
    """
    name = 'heartbeat'
    buffers = outstate['buffers']

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name,1)
        
        # Initialize heartbeat parameters with defaults
        instate['heart_rate'] = 60  # beats per minute
        instate['strength'] = 1.0   # intensity of the pulse (0.0 to 1.0)
        instate['base_color'] = [0.8, 0.0, 0.0, 0.1]  # default: red with low base intensity
        instate['peak_color'] = [1.0, 0.0, 0.0, 1.0]  # default: bright red at peak
        instate['phase'] = 0.0      # current phase of the heartbeat cycle
        
        # Parameters for circular patterns
        instate['wave_freq'] = 1.0  # Number of waves around the circle
        instate['rotation_speed'] = 0.8  # Rotation speed (revolutions per second)
        
        # Print debug information
        print(f"OTO_heartbeat: Initialized with heart_rate={instate['heart_rate']}, strength={instate['strength']}")
        return

    if instate['count'] == -1:
        buffers.generator_alphas[name] = 0
        return

    # Set generator alpha to full (ensure visibility)
    buffers.generator_alphas[name] = 1.0

    # Get all strip buffers for our generator
    all_buffers = buffers.get_all_buffers(name)
    
    # Directly access the strip_manager from the buffer manager
    strip_manager = buffers.strip_manager
    
    # Fallback to outstate if not available directly
    if strip_manager is None and 'strip_manager' in outstate:
        strip_manager = outstate['strip_manager']
    
    if strip_manager is None:
        print("OTO_heartbeat: No strip_manager available!")
        return
    
    # Update phase based on heart rate
    # heart_rate is in BPM, so divide by 60 to get beats per second
    beats_per_second = instate['heart_rate'] / 60.0
    phase_increment = beats_per_second * (outstate['current_time'] - outstate['last_time'])
    instate['phase'] = (instate['phase'] + phase_increment) % 1.0
    
    # Calculate rotation for circular patterns
    rotation_increment = instate['rotation_speed'] * (outstate['current_time'] - outstate['last_time'])
    if 'rotation_phase' not in instate:
        instate['rotation_phase'] = 0.0
    instate['rotation_phase'] = (instate['rotation_phase'] + rotation_increment) % 1.0
    
    # Check if we should update the base color from external conditions
    if 'heart_color' in outstate:
        # Update base and peak colors based on external input
        base_color = outstate['heart_color'].copy()
        base_color[3] = 0.1  # Low alpha for base
        instate['base_color'] = base_color
        
        peak_color = outstate['heart_color'].copy()
        peak_color[3] = 1.0  # Full alpha for peak
        instate['peak_color'] = peak_color
    
    # Check if heart rate should be updated from external conditions
    
    instate['heart_rate'] = outstate.get('heart_speed',60)
    
    # Check if strength should be updated from external conditions
    
    instate['strength'] = outstate.get('heart_intensity',100)/100.0
    
    # Process each buffer
    for strip_id, buffer in all_buffers.items():
        # Get the actual strip object
        try:
            strip = strip_manager.get_strip(strip_id)
        except KeyError:
            # Skip if strip doesn't exist
            continue
        
        # Check if strip is tagged with 'heart' category
        if 'heart' not in strip.groups:
            continue
        
        # Determine if this is a left or right heart component based on tags
        is_left = 'left' in strip.groups
        is_right = 'right' in strip.groups
        is_circle = 'circle' in strip.groups or strip.type == "circle"
        
        # If neither left nor right, default to left phase
        if not (is_left or is_right):
            is_left = True
        
        # Phase offset: left and right sides beat in alternating phases
        phase_offset = 0.0 if is_left else 0.5
        current_phase = (instate['phase'] + phase_offset) % 1.0
        
        # Create heartbeat waveform (global intensity)
        if current_phase < 0.1:
            # Sharp rise (systole)
            global_intensity = current_phase / 0.1
        elif current_phase < 0.4:
            # First decline (early diastole)
            global_intensity = 1.0 - 0.7 * ((current_phase - 0.1) / 0.3)
        else:
            # Second, more gradual decline (late diastole)
            global_intensity = 0.3 - 0.3 * ((current_phase - 0.4) / 0.6)
        
        # Apply strength modifier to global intensity
        global_intensity *= instate['strength']
        
        # Now handle different patterns based on strip type
        strip_length = len(buffer)
        
        if is_circle:
            # For circular strips, create a rotating wave pattern
            # Apply the circular pattern to individual LEDs
            for i in range(strip_length):
                # Calculate position around circle (0 to 1)
                pos = i / strip_length
                
                # Create multiple sinusoidal waves that rotate around the circle
                # Primary wave (follows the heartbeat rhythm)
                primary_wave = 0.5 + 0.5 * np.sin(2 * np.pi * (pos + instate['rotation_phase']))
                
                # Secondary wave (higher frequency for texture)
                secondary_wave = 0.5 + 0.5 * np.sin(2 * np.pi * instate['wave_freq'] * pos + 4 * np.pi * instate['rotation_phase'])
                
                # Combine waves with global heartbeat intensity
                local_intensity = global_intensity * (primary_wave * secondary_wave)
                
                # Ensure intensity is in valid range
                local_intensity = np.clip(local_intensity, 0.0, 1.0)
                
                # Interpolate between base and peak colors based on intensity
                r = instate['base_color'][0] + (instate['peak_color'][0] - instate['base_color'][0]) * local_intensity
                g = instate['base_color'][1] + (instate['peak_color'][1] - instate['base_color'][1]) * local_intensity
                b = instate['base_color'][2] + (instate['peak_color'][2] - instate['base_color'][2]) * local_intensity
                a = instate['base_color'][3] + (instate['peak_color'][3] - instate['base_color'][3]) * local_intensity
                
                # Set individual pixel
                buffer[i] = [r, g, b, a]
        else:
            # For non-circular strips, use uniform intensity
            # Interpolate between base and peak colors based on intensity
            r = instate['base_color'][0] + (instate['peak_color'][0] - instate['base_color'][0]) * global_intensity
            g = instate['base_color'][1] + (instate['peak_color'][1] - instate['base_color'][1]) * global_intensity
            b = instate['base_color'][2] + (instate['peak_color'][2] - instate['base_color'][2]) * global_intensity
            a = instate['base_color'][3] + (instate['peak_color'][3] - instate['base_color'][3]) * global_intensity
            
            # Apply the color to all pixels in the strip
            buffer[:] = [r, g, b, a]

def OTO_waiting(instate, outstate):
    """
    Generator function for a waiting state that creates moving points with fading trails
    along different strip types based on their groups.
    
    Features:
    1. Global alpha controlled by outstate['control_waiting'] value
    2. Color set by outstate['control_hue'] key (single hue value)
    3. For spine strips: 10 spots moving downward with random speed variations
    4. For base strips: 10 spots (5 up, 5 down) that fade when reaching center
    5. For ear (circular) strips: Points moving in circular patterns
    """
    name = 'waiting'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize spots for different strip types
        instate['spots'] = {}
        
        # Get all strip IDs
        strip_ids = list(buffers.get_all_buffers(name).keys())
        
        for strip_id in strip_ids:
            # Skip if strip doesn't exist in manager
            if strip_id not in strip_manager.strips:
                continue
                
            strip = strip_manager.get_strip(strip_id)
            strip_length = strip.length
            
            # Initialize spots based on strip type/group
            if strip_id in ['left_spine', 'right_spine']:
                # 10 spots moving down the spine
                instate['spots'][strip_id] = [
                    {
                        'pos': np.random.randint(0, strip_length),
                        'speed_factor': 0.8 + np.random.random() * 0.4,  # Random speed variation
                        'alpha': 1.0
                    } for _ in range(10)
                ]
            elif 'base' in strip.groups:
                # 5 spots moving upward, 5 moving downward
                spots = []
                for i in range(5):
                    # Upward moving spots
                    spots.append({
                        'pos': np.random.randint(0, strip_length // 2),
                        'speed_factor': 0.8 + np.random.random() * 0.4,
                        'direction': 1,  # Moving upward (increasing index)
                        'alpha': 1.0
                    })
                    # Downward moving spots
                    spots.append({
                        'pos': np.random.randint(strip_length // 2, strip_length),
                        'speed_factor': 0.8 + np.random.random() * 0.4,
                        'direction': -1,  # Moving downward (decreasing index)
                        'alpha': 1.0
                    })
                instate['spots'][strip_id] = spots
            elif 'ear' in strip.groups:# or strip.type == "circle":
                # Circular movement - 3 spots per ear
                instate['spots'][strip_id] = [
                    {
                        'pos': np.random.randint(0, strip_length),
                        'speed_factor': 0.8 + np.random.random() * 0.4,
                        'alpha': 1.0
                    } for _ in range(3)
                ]
        return

    if instate['count'] == -1:
        buffers.generator_alphas[name] = 0
        return

    # Get waiting level from outstate or default to 0
    waiting_level = outstate.get('control_mode_waiting', 1)
    
    # Set generator alpha based on waiting level
    if waiting_level > 0.05:
        buffers.generator_alphas[name] = waiting_level
    else:
        buffers.generator_alphas[name] = 0
        return  # Skip further processing if not visible

    # Get base speed from outstate or use default
    base_speed = outstate.get('control_speed', 5.0)  # Default 5 pixels per second
    
    # Get hue from outstate or use default (magenta)
    hue = outstate.get('control_hue', 50)/100.0  # Default hue=0.8 (magenta/purple)
    
    # Fixed saturation and value for vibrant colors
    saturation = 1.0
    value = 1.0
    
    # Time delta for movement calculation
    delta_time = outstate['current_time'] - outstate['last_time']
    
    # Get all strip buffers for our generator
    all_buffers = buffers.get_all_buffers(name)
    
    # Process each buffer
    for strip_id, buffer in all_buffers.items():
        # Skip if we don't have spots for this strip
        if strip_id not in instate['spots']:
            continue
        
        # Fade existing pixels (create trail effect)
        fade_factor = 0.85  # 15% reduction per frame
        buffer[:, 3] *= fade_factor
        
        # Skip if strip doesn't exist in manager
        if strip_id not in strip_manager.strips:
            continue
            
        strip = strip_manager.get_strip(strip_id)
        strip_length = len(buffer)
        center_point = strip_length // 2
        
        # Process each spot for this strip
        for spot in instate['spots'][strip_id]:
            # Calculate movement based on speed, time and spot's speed factor
            movement = base_speed * delta_time * spot['speed_factor']
            
            # Update position based on strip type and direction
            if strip_id in ['left_spine', 'right_spine']:
                # Spine strips - move downward
                spot['pos'] = (spot['pos'] + movement) % strip_length
                
            elif 'base' in strip.groups:
                # Base strips - spots move toward center then fade
                direction = spot['direction']
                new_pos = spot['pos'] + (movement * direction)
                
                # If moving upward and passing center, start fading
                if direction > 0 and new_pos > center_point:
                    # Calculate fade based on distance from center
                    fade_distance = strip_length - center_point
                    spot['alpha'] = max(0, 1.0 - ((new_pos - center_point) / fade_distance))
                
                # If moving downward and passing center, start fading
                elif direction < 0 and new_pos < center_point:
                    # Calculate fade based on distance from center
                    spot['alpha'] = max(0, 1.0 - ((center_point - new_pos) / center_point))
                
                # Reset if reaching end
                if new_pos >= strip_length or new_pos < 0:
                    if direction > 0:  # Was moving upward
                        new_pos = 0
                    else:  # Was moving downward
                        new_pos = strip_length - 1
                    spot['alpha'] = 1.0  # Reset alpha
                
                spot['pos'] = new_pos
                
            elif 'ear' in strip.groups or strip.type == "circle":
                # Circular movement - just wrap around
                spot['pos'] = (spot['pos'] + movement) % strip_length
            
            # Convert position to integer index
            idx = int(spot['pos']) % strip_length
            
            # Apply the spot color with spot's alpha
            # Use HSV to RGB conversion for the color using only the provided hue
            r, g, b = hsv_to_rgb(hue, saturation, value)
            
            # Set the pixel with the spot's alpha
            buffer[idx] = [r, g, b, spot['alpha']]
            
            # Add a small glow around the spot (optional)
            glow_size = 2  # Size of glow effect
            for i in range(1, glow_size + 1):
                glow_alpha = spot['alpha'] * (0.7 ** i)  # Exponential falloff
                
                # Set pixels before and after with reduced alpha
                before_idx = (idx - i) % strip_length
                after_idx = (idx + i) % strip_length
                
                buffer[before_idx] = [r, g, b, glow_alpha]
                buffer[after_idx] = [r, g, b, glow_alpha]


# ... existing code ...

def OTO_rgb_test(instate, outstate):
    """
    Test pattern that moves 3 adjacent RGB dots through all strips in sequence.
    The dots (red, green, blue) move based on the speed input, traversing one strip completely 
    before moving to the next, and repeating the cycle once all strips are traversed.
    """
    name = 'rgb_test'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize test pattern state
        instate['position'] = 0.0  # Floating point position for smooth movement
        
        # Get ordered list of all strip IDs
        strip_ids = list(buffers.get_all_buffers(name).keys())
        instate['strip_ids'] = strip_ids
        instate['current_strip_index'] = 0
        
        return

    if instate['count'] == -1:
        buffers.generator_alphas[name] = 0
        return

    # Set generator alpha to full
    buffers.generator_alphas[name] = 1.0

    # Get base speed from outstate or use default
    base_speed = outstate.get('control_speed', 5.0)  # Default 5 pixels per second
    
    # Time delta for movement calculation
    delta_time = outstate['current_time'] - outstate['last_time']
    
    # Get all strip buffers for our generator
    all_buffers = buffers.get_all_buffers(name)
    
    # Clear all buffers first
    for strip_id, buffer in all_buffers.items():
        buffer[:] = [0, 0, 0, 0]
    
    # If no strips are available, exit
    if not instate['strip_ids']:
        return
    
    # Get current strip ID
    current_strip_id = instate['strip_ids'][instate['current_strip_index']]
    
    # Get current strip buffer
    if current_strip_id in all_buffers:
        buffer = all_buffers[current_strip_id]
        strip_length = len(buffer)
        
        # Calculate movement based on speed and time
        movement = base_speed * delta_time
        instate['position'] += movement
        
        # If position exceeds strip length, move to next strip
        if int(instate['position']) >= strip_length - 2:  # -2 to ensure all 3 dots are visible
            instate['position'] = 0.0
            instate['current_strip_index'] = (instate['current_strip_index'] + 1) % len(instate['strip_ids'])
        
        # Calculate integer position for the dots
        pos = int(instate['position'])
        
        # Set the 3 adjacent RGB dots
        if pos < strip_length:
            buffer[pos] = [1.0, 0.0, 0.0, 1.0]  # Red
        
        if pos + 1 < strip_length:
            buffer[pos + 1] = [0.0, 1.0, 0.0, 1.0]  # Green
        
        if pos + 2 < strip_length:
            buffer[pos + 2] = [0.0, 0.0, 1.0, 1.0]  # Blue