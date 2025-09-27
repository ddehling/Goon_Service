from sceneutils.imgutils import *  # noqa: F403
import numpy as np
from pathlib import Path


def hsv_to_rgb_vectorized(h, s, v):
    """
    Vectorized HSV to RGB conversion.
    h, s, v can be numpy arrays or scalars.
    Returns r, g, b arrays with same shape as input.
    """
    h = np.asarray(h)
    s = np.asarray(s)
    v = np.asarray(v)
    
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    
    idx = i % 6
    
    # Initialize output arrays
    shape = np.broadcast(h, s, v).shape
    r = np.zeros(shape)
    g = np.zeros(shape)
    b = np.zeros(shape)
    
    # Assign values based on hue sector
    r = np.select([idx == 0, idx == 1, idx == 2, idx == 3, idx == 4, idx == 5],
                  [v, q, p, p, t, v])
    g = np.select([idx == 0, idx == 1, idx == 2, idx == 3, idx == 4, idx == 5],
                  [t, v, v, q, p, p])
    b = np.select([idx == 0, idx == 1, idx == 2, idx == 3, idx == 4, idx == 5],
                  [p, p, t, v, v, q])
    
    return r, g, b
  
def GS_prototype_example(instate, outstate):
    """
    Simple colored sine wave that varies with time.
    """
    name = 'prototype_example'
    buffers = outstate['buffers']

    if instate['count'] == 0:
        buffers.register_generator(name)
        return

    if instate['count'] == -1:
        buffers.generator_alphas[name] = 0
        return

   
    sound_level=outstate.get('sound_level', 1.0)
    # Calculate fade in/out over first and last 5 seconds
    elapsed_time = instate['elapsed_time']
    remaining_time = instate['duration'] - elapsed_time
    
    fade_alpha = 1.0
    if elapsed_time < 5.0:
        # Fade in over first 5 seconds
        fade_alpha = elapsed_time / 5.0
    elif remaining_time < 5.0:
        # Fade out over last 5 seconds
        fade_alpha = remaining_time / 5.0
    
    # Combine control level with fade
    final_alpha = fade_alpha
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = final_alpha
    
    # Skip rendering if alpha is too low
    if final_alpha < 0.01:
        return
    
    
    # Get time for animation
    current_time = outstate['current_time']
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Process each strip
    for strip_id, buffer in pattern_buffers.items():
        strip_length = len(buffer)
        
        # Create sine wave across strip that moves with time
        positions = np.arange(strip_length)
        wave = 0.5 + 0.5 * np.sin(positions / strip_length * 2 * np.pi + current_time)
        
        # Color cycling with time
        hue = (current_time * 0.1) % 1.0
        
        # Convert to RGB
        r_values, g_values, b_values = hsv_to_rgb_vectorized(hue, 1, wave)

        
        # Set buffer with full alpha
        alpha_values = np.ones(strip_length)*final_alpha
        rgba_values = np.stack([r_values, g_values, b_values, alpha_values], axis=1)
        buffer[:] = rgba_values
        
def GS_blink_fade(instate, outstate):
    """
    Blink and fade pattern where random pixels turn on each frame with random colors
    from a palette, and all pixels fade at 0.3 per second.
    """
    name = 'blink_fade'
    buffers = outstate['buffers']

    if instate['count'] == 0:
        buffers.register_generator(name)
        return

    if instate['count'] == -1:
        buffers.generator_alphas[name] = 0
        return

    sound_level = outstate.get('sound_level', 1.0)
    
    # Calculate fade in/out over first and last 5 seconds
    elapsed_time = instate['elapsed_time']
    remaining_time = instate['duration'] - elapsed_time
    
    fade_alpha = 1.0
    if elapsed_time < 5.0:
        fade_alpha = elapsed_time / 5.0
    elif remaining_time < 5.0:
        fade_alpha = remaining_time / 5.0
    
    buffers.generator_alphas[name] = fade_alpha
    
    if fade_alpha < 0.01:
        return
    
    current_time = outstate['current_time']
    delta_time = outstate['current_time'] - outstate['last_time']
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Color palette (6 HSV colors)
    color_palette = np.array([
        [0.086, 0.727, 0.761],    # Pure red
        [0.159, 0.932, 0.459],   # Pure yellow  
        [0.400, 0.400, 0.196],   # Pure green
        [0.574, 0.902, 0.561],    # Pure cyan
        [0.983, 0.660, 0.635],   # Pure blue
        [0.050, 0.932, 0.463],   # Pure magenta
    ])
    
    # Parameters
    pixels_per_frame = 3  # Number of pixels to turn on each frame
    fade_rate = 0.6  # Fade rate per second
    
    for strip_id, buffer in pattern_buffers.items():
        strip_length = len(buffer)
        
        # Apply exponential fade to all pixels
        fade_factor = np.exp(-fade_rate * delta_time)
        buffer[:, :3] *= fade_factor  # Fade RGB channels
        
        # Randomly select pixels to turn on this frame
        num_new_pixels = min(pixels_per_frame, strip_length)
        new_pixel_indices = np.random.choice(strip_length, num_new_pixels, replace=False)
        
        # Randomly select colors from palette for each new pixel
        color_indices = np.random.choice(len(color_palette), num_new_pixels)
        selected_colors_hsv = color_palette[color_indices]
        
        # Convert HSV colors to RGB
        h_values = selected_colors_hsv[:, 0]
        s_values = selected_colors_hsv[:, 1] 
        v_values = selected_colors_hsv[:, 2]
        
        r_values, g_values, b_values = hsv_to_rgb_vectorized(h_values, s_values, v_values)
        
        # Set the new pixels to full brightness
        buffer[new_pixel_indices, 0] = r_values
        buffer[new_pixel_indices, 1] = g_values
        buffer[new_pixel_indices, 2] = b_values
        
        # Set alpha for entire strip
        buffer[:, 3] = fade_alpha

def GS_blood_flow(instate, outstate):
    """
    Simple blood flow simulation with colored dots moving down strips.
    Each dot is a single pixel that leaves a decaying trail behind it.
    """
    name = 'blood_flow'
    buffers = outstate['buffers']

    if instate['count'] == 0:
        buffers.register_generator(name)
        # Initialize dot states for each strip
        if not hasattr(buffers, 'blood_flow_dots'):
            buffers.blood_flow_dots = {}
        return

    if instate['count'] == -1:
        buffers.generator_alphas[name] = 0
        return

    sound_level = outstate.get('sound_level', 1.0)
    
    # Calculate fade in/out over first and last 5 seconds
    elapsed_time = instate['elapsed_time']
    remaining_time = instate['duration'] - elapsed_time
    
    fade_alpha = 1.0
    if elapsed_time < 5.0:
        fade_alpha = elapsed_time / 5.0
    elif remaining_time < 5.0:
        fade_alpha = remaining_time / 5.0
    
    buffers.generator_alphas[name] = fade_alpha
    
    if fade_alpha < 0.01:
        return
    
    current_time = outstate['current_time']
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Heart beat pattern for speed variation
    beat_period = 1.2
    beat_phase = (current_time % beat_period) / beat_period
    if beat_phase < 0.15:
        beat_intensity = (beat_phase / 0.15) ** 0.5
    else:
        decay_phase = (beat_phase - 0.15) / 0.85
        beat_intensity = np.exp(-decay_phase * 4)
    
    base_speed = 10  # pixels per second
    speed_multiplier = 1.0 + beat_intensity * 2.0
    
    for strip_id, buffer in pattern_buffers.items():
        strip_length = len(buffer)
        
        # Initialize dots for this strip if needed
        if strip_id not in buffers.blood_flow_dots:
            num_dots = 10
            
            buffers.blood_flow_dots[strip_id] = {
                'positions': np.random.uniform(0, strip_length, num_dots),
                'speeds': np.random.uniform(0.8, 1.5, num_dots),  # Speed multipliers
                'colors': np.random.rand(num_dots, 3),  # RGB colors
                'last_time': current_time
            }
            
            # Set some dots to be more red/white for blood theme
            for i in range(num_dots):
                if i < 8:  # First 3 dots are red-ish
                    h = np.random.uniform(0.9, 1.05) % 1.0  # Red hues
                    s = np.random.uniform(0.7, 1.0)
                    v = np.random.uniform(0.8, 1.0)
                else:  # Last 2 dots are white-ish
                    h = np.random.uniform(0.05, 0.15)  # Warm white
                    s = np.random.uniform(0.1, 0.4)
                    v = np.random.uniform(0.9, 1.0)
                
                r, g, b = hsv_to_rgb_vectorized(h, s, v)
                buffers.blood_flow_dots[strip_id]['colors'][i] = [r, g, b]
        
        strip_data = buffers.blood_flow_dots[strip_id]
        dt = current_time - strip_data['last_time']
        strip_data['last_time'] = current_time
        
        # Decay all pixels in the buffer (create trailing effect)
        decay_rate = 0.85  # How much remains each frame (lower = faster decay)
        buffer[:, :3] *= decay_rate
        
        # Update dot positions
        effective_speeds = strip_data['speeds'] * base_speed * speed_multiplier
        strip_data['positions'] += effective_speeds * dt
        strip_data['positions'] %= strip_length  # Wrap around
        
        # Draw dots at their new positions
        dot_indices = strip_data['positions'].astype(int)
        buffer[dot_indices, :3] = strip_data['colors']  # Set RGB
        buffer[:, 3] = fade_alpha  # Set alpha for entire strip



def GS_tingles(instate, outstate):
    """
    tingles effect: Low-intensity per-frame noise combined with bright 
    single-pixel dots that fade rapidly. Uses a persistent buffer for the dots.
    """
    name = 'tingles'
    buffers = outstate['buffers']

    if instate['count'] == 0:
        buffers.register_generator(name)
        # Initialize persistent dot buffer for each strip
        if not hasattr(buffers, 'tingles_dots'):
            buffers.tingles_dots = {}
        if not hasattr(buffers, 'tingles_last_spawn'):
            buffers.tingles_last_spawn = {}
        return

    if instate['count'] == -1:
        buffers.generator_alphas[name] = 0
        return

    sound_level = outstate.get('sound_level', 1.0)
    
    # Calculate fade in/out over first and last 5 seconds
    elapsed_time = instate['elapsed_time']
    remaining_time = instate['duration'] - elapsed_time
    
    fade_alpha = 1.0
    if elapsed_time < 5.0:
        fade_alpha = elapsed_time / 5.0
    elif remaining_time < 5.0:
        fade_alpha = remaining_time / 5.0
    
    buffers.generator_alphas[name] = fade_alpha
    
    if fade_alpha < 0.01:
        return
    
    current_time = outstate['current_time']
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Dot spawn parameters
    dot_spawn_rate = 5  # Average dots per second per strip
    dot_fade_time = 0.2   # Time for dot to fade to zero
    dot_max_brightness = 1
    dot_spawn_num=int(6*sound_level)
    
    for strip_id, buffer in pattern_buffers.items():
        strip_length = len(buffer)
        
        # Initialize persistent dot buffer for this strip (RGB values)
        if strip_id not in buffers.tingles_dots:
            buffers.tingles_dots[strip_id] = np.zeros((strip_length, 4))  # R, G, B, spawn_time
            buffers.tingles_last_spawn[strip_id] = current_time
        
        dot_buffer = buffers.tingles_dots[strip_id]
        
        # Spawn new dots randomly
        time_since_spawn = current_time - buffers.tingles_last_spawn[strip_id]
        spawn_probability = dot_spawn_rate * time_since_spawn
        
        active_mask = (current_time - dot_buffer[:, 3]) < dot_fade_time
        # Find an inactive pixel (one that has faded out)
        for n in range(dot_spawn_num):
            
            
            inactive_pixels = np.where(~active_mask)[0]
            
            if len(inactive_pixels) > 0:
                # Choose random inactive pixel
                pixel_idx = np.random.choice(inactive_pixels)
                
                # Set bright white/warm color
                hue = np.random.uniform(0.6, 0.7)  # Warm white to yellow
                saturation = np.random.uniform(0.3, 0.6)
                value = dot_max_brightness
                
                r, g, b = hsv_to_rgb_vectorized(hue, saturation, value)
                
                dot_buffer[pixel_idx, 0] = r
                dot_buffer[pixel_idx, 1] = g  
                dot_buffer[pixel_idx, 2] = b
                dot_buffer[pixel_idx, 3] = current_time  # Record spawn time
                
                buffers.tingles_last_spawn[strip_id] = current_time
        
        # Generate per-frame low-intensity noise
        noise_intensity = 0.1  # Low intensity
        noise_r = np.random.uniform(0, noise_intensity, strip_length)
        noise_g = np.random.uniform(0, noise_intensity, strip_length) 
        noise_b = np.random.uniform(0, noise_intensity, strip_length)
        
        # Bias noise toward warm colors (more red/yellow, less blue)
        noise_r *= 1.3
        noise_g *= 1.1
        noise_b *= 0.7
        
        # Start with noise as base
        buffer[:, 0] = noise_r
        buffer[:, 1] = noise_g
        buffer[:, 2] = noise_b
        buffer[:, 3] = fade_alpha
        
        # Calculate fading for active dots
        dot_ages = current_time - dot_buffer[:, 3]
        active_mask = (dot_ages < dot_fade_time) & (dot_ages >= 0)
        
        if np.any(active_mask):
            # Exponential fade for active dots  
            fade_factors = np.exp(-dot_ages[active_mask] / (dot_fade_time * 0.3))
            
            # Add faded dots to the buffer (additive)
            buffer[active_mask, 0] += dot_buffer[active_mask, 0] * fade_factors
            buffer[active_mask, 1] += dot_buffer[active_mask, 1] * fade_factors
            buffer[active_mask, 2] += dot_buffer[active_mask, 2] * fade_factors
            
            # Clamp to prevent oversaturation
            buffer[:, :3] = np.clip(buffer[:, :3], 0, 1)

def GS_rage_lightning(instate, outstate):
    """
    Generator function that creates a passionate rage-themed pattern across all strips.
    
    Features:
    1. Global alpha controlled by outstate['control_rage'] value
    2. Fast blinking lightning effects in yellows, blues, and whites
    3. High-speed component causing entire strips to flash with noisy variations
    4. Rapid changes in which strips are activated to create chaotic, angry pattern
    5. Consistent low-level noise across all pixels for added intensity
    6. Intense flashing reminiscent of electrical storms and passionate rage
    """
    name = 'rage_lightning'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['flash_timer'] = 0.0       # Timer for rapid flashes
        instate['strip_change_timer'] = 0.0  # Timer for changing active strips
        instate['active_strips'] = []      # Currently active strips
        instate['flash_state'] = False     # Current flash state (on/off)
        instate['last_flash_time'] = 0.0   # Time of last flash change
        instate['spots_state'] = {}  
        
        # Color palette (HSV values)
        instate['colors'] = {
            'electric_blue': [0.6, 0.85, 1.0],    # Intense blue
            'bright_yellow': [0.15, 0.8, 1.0],    # Bright yellow
            'white_hot': [0.0, 0.0, 1.0],         # Pure white
            'light_blue': [0.55, 0.7, 1.0],       # Light blue
            'pale_yellow': [0.13, 0.5, 1.0]       # Pale yellow
        }
        
        # Timing parameters
        instate['min_flash_time'] = 0.1    # Minimum time between flashes (seconds)
        instate['max_flash_time'] = 0.4    # Maximum time between flashes (seconds)
        instate['strip_change_time'] = 0.3  # Time between changing active strips (seconds)
        instate['active_strip_percent'] = 0.3  # Percentage of strips active at once
        
        # Noise parameters
        instate['base_noise_min'] = 0.2    # Minimum noise intensity (20%)
        instate['base_noise_max'] = 0.4    # Maximum noise intensity (40%)
        
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        return

    # Get rage level from outstate (default to 0)
    rage_level = 1
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = rage_level
    
    # Skip rendering if alpha is too low
    if rage_level < 0.01:
        return
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if remaining_time < 10.0:
        fade_alpha = remaining_time / 10.0
        fade_alpha = max(0.0, fade_alpha)
        buffers.generator_alphas[name] = fade_alpha * rage_level
    
    # Get delta time for animation calculations
    delta_time = outstate['current_time'] - outstate['last_time']
    current_time = outstate['current_time']
    
    # Update flash timer - controls rapid flash component
    instate['flash_timer'] += delta_time
    
    # Determine if it's time for a new flash
    time_since_last_flash = current_time - instate['last_flash_time']
    flash_interval = instate['min_flash_time'] + np.random.random() * (instate['max_flash_time'] - instate['min_flash_time'])
    
    # Higher rage intensifies flashing (shorter intervals)
    flash_interval = flash_interval * (1.0 - rage_level * 0.5)
    
    if time_since_last_flash >= flash_interval:
        # Time for a new flash state
        instate['last_flash_time'] = current_time
        instate['flash_state'] = not instate['flash_state']
    
    # Update strip change timer - controls which strips are active
    instate['strip_change_timer'] += delta_time
    
    # Check if it's time to change active strips
    strip_change_time = instate['strip_change_time'] * (1.0 - rage_level * 0.5)  # Faster changes with higher rage
    
    if instate['strip_change_timer'] >= strip_change_time:
        instate['strip_change_timer'] = 0.0
        
        # Select new active strips
        all_strips = list(strip_manager.strips.keys())
        if all_strips:
            # Calculate how many strips to activate
            active_percent = instate['active_strip_percent'] * (1.0 + rage_level * 0.5)  # More active strips with higher rage
            active_percent = min(0.8, active_percent)  # Cap at 80%
            
            num_to_select = max(1, int(len(all_strips) * active_percent))
            instate['active_strips'] = np.random.choice(all_strips, num_to_select, replace=False)
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Process each buffer
    for strip_id, buffer in pattern_buffers.items():
        # Skip if strip doesn't exist in manager
        if strip_id not in strip_manager.strips:
            continue
            
        strip_length = len(buffer)
        
        # Determine if this is an active strip
        is_active = strip_id in instate['active_strips']
        strip = strip_manager.get_strip(strip_id)
        strip_length = len(buffer)
        # Start with a dark base - slight blue tint
        buffer[:] = [0.0, 0.0, 0.1, 0.1]  # Very dim blue base



        
        if is_active:
            # This strip is active
            
            if instate['flash_state']:
                # Flash is on - light up the entire strip with noise
                
                # Choose a base color for this strip
                color_name = np.random.choice(list(instate['colors'].keys()))
                h, s, v = instate['colors'][color_name]
                
                # Generate base RGB values
                r, g, b = hsv_to_rgb(h, s, v)
                
                # Create noise variation across the strip
                for i in range(strip_length):
                    # Add noise to color (more noise with higher rage)
                    noise_amount = 0.2 + rage_level * 0.3
                    r_noise = r * (1.0 - noise_amount + np.random.random() * noise_amount * 2)
                    g_noise = g * (1.0 - noise_amount + np.random.random() * noise_amount * 2)
                    b_noise = b * (1.0 - noise_amount + np.random.random() * noise_amount * 2)
                    
                    # Ensure values are in valid range
                    r_noise = max(0.0, min(1.0, r_noise))
                    g_noise = max(0.0, min(1.0, g_noise))
                    b_noise = max(0.0, min(1.0, b_noise))
                    
                    # Set alpha - also with some variation
                    alpha = 0.7 + np.random.random() * 0.3
                    
                    # Add to buffer
                    buffer[i] = [r_noise, g_noise, b_noise, alpha]
                
                # Add some brighter spots (more with higher rage)
                num_bright_spots = int(strip_length * (0.1 + rage_level * 0.2))
                for _ in range(num_bright_spots):
                    pos = np.random.randint(0, strip_length)
                    
                    # Brighter version of the base color
                    br, bg, bb = hsv_to_rgb(h, s * 0.7, min(1.0, v * 1.3))  # Less saturated, brighter
                    
                    buffer[pos] = [br, bg, bb, 1.0]
            else:
                # Flash is off but strip is active - add subtle ambient glow
                for i in range(strip_length):
                    # Add small ambient effect
                    flicker = 0.05 + 0.05 * np.sin(current_time * 10 + i * 0.1)
                    
                    # Use blue for ambient glow
                    h, s, v = instate['colors']['electric_blue']
                    r, g, b = hsv_to_rgb(h, s * 0.7, v * 0.4)
                    
                    # Set pixel with low intensity but higher than non-active strips
                    buffer[i] = [r, g, b, 0.2 + flicker]
        else:
            # Non-active strip - very dim ambient only
            # But occasionally (based on rage) flash briefly
            random_flash = np.random.random() < (0.01 * rage_level)
            
            if random_flash:
                # Brief random flash on non-active strip
                color_name = np.random.choice(['electric_blue', 'light_blue'])
                h, s, v = instate['colors'][color_name]
                r, g, b = hsv_to_rgb(h, s, v)
                
                # Lower intensity than active strips
                for i in range(strip_length):
                    noise = 0.7 + np.random.random() * 0.3
                    buffer[i] = [r * noise, g * noise, b * noise, 0.4]
        
        # Apply additional base noise to EVERY pixel on EVERY strip
        # This adds the consistent 20-40% intensity noise across all pixels
        curr_rgba = buffer[:]
        
        # Generate random colors for each pixel
        color_names = list(instate['colors'].keys())
        random_color_indices = np.random.choice(len(color_names), strip_length)
        
        # Pre-convert all colors to RGB
        color_rgb_array = np.array([hsv_to_rgb(*instate['colors'][color]) for color in color_names])
        
        # Select RGB values for each pixel based on random indices
        noise_rgb = color_rgb_array[random_color_indices]  # Shape: (strip_length, 3)
        
        # Generate noise intensities for all pixels at once
        noise_base = np.random.uniform(
            instate['base_noise_min'], 
            instate['base_noise_max'], 
            strip_length
        )
        
        # Apply rage level scaling and capping
        noise_intensities = noise_base * (1.0 + rage_level * 0.5)
        noise_intensities = np.minimum(noise_intensities, 0.6)
        
        # Apply noise with additive blending
        new_rgb = np.minimum(1.0, curr_rgba[:, :3] + (noise_rgb * noise_intensities[:, np.newaxis]))
        new_alpha = np.maximum(curr_rgba[:, 3], noise_intensities)
        
        # Update buffer
        buffer[:, :3] = new_rgb
        buffer[:, 3] = new_alpha
            



def GS_curious_playful(instate, outstate):
    """
    Simplified curious/playful pattern with moving colored regions.
    Creates vibrant, moving color waves across strips.
    """
    name = 'curious_playful'
    buffers = outstate['buffers']

    if instate['count'] == 0:
        buffers.register_generator(name)
        return

    if instate['count'] == -1:
        buffers.generator_alphas[name] = 0
        return

    # Calculate fade in/out
    elapsed_time = instate['elapsed_time']
    remaining_time = instate['duration'] - elapsed_time
    
    fade_alpha = 1.0
    if elapsed_time < 5.0:
        fade_alpha = elapsed_time / 5.0
    elif remaining_time < 5.0:
        fade_alpha = remaining_time / 5.0
    
    buffers.generator_alphas[name] = fade_alpha
    
    if fade_alpha < 0.01:
        return
    
    current_time = outstate['current_time']
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Color palette (HSV)
    colors = [
        [0.0, 0.8, 0.9],   # Red
        [0.15, 0.8, 0.9],  # Yellow
        [0.33, 0.8, 0.9],  # Green
        [0.5, 0.8, 0.9],   # Cyan
        [0.67, 0.8, 0.9],  # Blue
        [0.83, 0.8, 0.9],  # Magenta
    ]
    colors = np.array([
        [0.086, 0.727, 0.761],    # Pure red
        [0.159, 0.932, 0.459],   # Pure yellow  
        [0.400, 0.400, 0.196],   # Pure green
        [0.574, 0.902, 0.561],    # Pure cyan
        [0.983, 0.660, 0.635],   # Pure blue
        [0.050, 0.932, 0.463],   # Pure magenta
    ])
    for strip_id, buffer in pattern_buffers.items():
        strip_length = len(buffer)
        
        # Create 3 moving waves per strip
        positions = np.arange(strip_length, dtype=float)
        buffer[:] = 0  # Clear buffer
        
        # Convert strip_id to number for color variation
        strip_hash = hash(strip_id) % len(colors)
        
        for wave_idx in range(3):
            # Each wave moves at different speed and uses different color
            wave_speed = 20 + wave_idx * 15  # pixels per second
            wave_width = strip_length * 0.3
            color_idx = (wave_idx + strip_hash) % len(colors)
            
            # Calculate wave center position (wraps around)
            center = (current_time * wave_speed + wave_idx * strip_length / 3) % strip_length
            
            # Calculate distances with wrap-around
            dist1 = np.abs(positions - center)
            dist2 = strip_length - dist1
            distances = np.minimum(dist1, dist2)
            
            # Gaussian-like influence
            influence = np.exp(-0.5 * (distances / (wave_width / 3))**2)
            
            # Convert color to RGB
            h, s, v = colors[color_idx]
            r, g, b = hsv_to_rgb_vectorized(h, s, v * influence)
            
            # Add to buffer (additive blending)
            buffer[:, 0] += r * influence
            buffer[:, 1] += g * influence  
            buffer[:, 2] += b * influence
            buffer[:, 3] = np.maximum(buffer[:, 3], influence * 0.7)
        
        # Clamp values and apply fade
        buffer[:, :3] = np.clip(buffer[:, :3], 0, 1)
        buffer[:, 3] = np.clip(buffer[:, 3] * fade_alpha, 0, 1)
        
        # Add occasional sparkles
        sparkle_mask = np.random.random(strip_length) < 0.01
        if np.any(sparkle_mask):
            buffer[sparkle_mask, :3] = 1.0  # White sparkles
            buffer[sparkle_mask, 3] = np.maximum(buffer[sparkle_mask, 3], 0.8)

def GS_forest(instate, outstate):
    """
    Forest simulation with colored dots moving down strips.
    Each dot is a single pixel that leaves a decaying trail behind it.
    Each dot has two colors: brown for pixels <100, green for pixels >=100.
    No heartbeat speed variation - each dot moves at its constant speed.
    Includes random twinkle overlay using same color scheme.
    """
    name = 'forest'
    buffers = outstate['buffers']

    if instate['count'] == 0:
        buffers.register_generator(name)
        # Initialize dot states for each strip
        if not hasattr(buffers, 'forest_dots'):
            buffers.forest_dots = {}
        return

    if instate['count'] == -1:
        buffers.generator_alphas[name] = 0
        return

    sound_level = outstate.get('sound_level', 1.0)
    
    # Calculate fade in/out over first and last 5 seconds
    elapsed_time = instate['elapsed_time']
    remaining_time = instate['duration'] - elapsed_time
    
    fade_alpha = 1.0
    if elapsed_time < 5.0:
        fade_alpha = elapsed_time / 5.0
    elif remaining_time < 5.0:
        fade_alpha = remaining_time / 5.0
    
    buffers.generator_alphas[name] = fade_alpha
    
    if fade_alpha < 0.01:
        return
    
    current_time = outstate['current_time']
    delta_time = outstate['current_time'] - outstate['last_time']
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Brown color palette (HSV)
    brown_colors = np.array([
        [0.08, 0.8, 0.4],   # Dark brown
        [0.06, 0.7, 0.5],   # Medium brown  
        [0.1, 0.6, 0.6],    # Light brown
        [0.05, 0.9, 0.3],   # Very dark brown
        [0.12, 0.5, 0.7],   # Tan brown
    ])
    
    # Green color palette (HSV)
    green_colors = np.array([
        [0.25, 0.8, 0.5],   # Dark forest green
        [0.33, 0.7, 0.6],   # Medium green
        [0.28, 0.9, 0.4],   # Deep green
        [0.35, 0.6, 0.7],   # Light green
        [0.22, 0.85, 0.45], # Pine green
    ])
    
    base_speed = 10  # pixels per second (constant, no heartbeat variation)
    
    # Twinkle parameters
    twinkles_per_frame = 2  # Number of twinkles to spawn each frame
    twinkle_fade_rate = 1.2  # Fade rate per second
    
    for strip_id, buffer in pattern_buffers.items():
        strip_length = len(buffer)
        
        # Initialize dots for this strip if needed
        if strip_id not in buffers.forest_dots:
            num_dots = 10
            
            buffers.forest_dots[strip_id] = {
                'positions': np.random.uniform(0, strip_length, num_dots),
                'speeds': np.random.uniform(0.8, 1.5, num_dots),  # Speed multipliers
                'brown_colors': np.zeros((num_dots, 3)),  # RGB colors for brown regions
                'green_colors': np.zeros((num_dots, 3)),  # RGB colors for green regions
                'last_time': current_time
            }
            
            # Assign colors to each dot
            for i in range(num_dots):
                # Choose random brown color
                brown_hsv = brown_colors[np.random.randint(len(brown_colors))]
                h, s, v = brown_hsv
                r, g, b = hsv_to_rgb_vectorized(h, s, v)
                buffers.forest_dots[strip_id]['brown_colors'][i] = [r, g, b]
                
                # Choose random green color
                green_hsv = green_colors[np.random.randint(len(green_colors))]
                h, s, v = green_hsv
                r, g, b = hsv_to_rgb_vectorized(h, s, v)
                buffers.forest_dots[strip_id]['green_colors'][i] = [r, g, b]
        
        strip_data = buffers.forest_dots[strip_id]
        dt = current_time - strip_data['last_time']
        strip_data['last_time'] = current_time
        
        # Apply exponential fade to all pixels for twinkle effect
        fade_factor = np.exp(-twinkle_fade_rate * delta_time)
        buffer[:, :3] *= fade_factor
        
        # Apply additional decay for trailing effect
        decay_rate = 0.995  # How much remains each frame (lower = faster decay)
        buffer[:, :3] *= decay_rate
        
        # Update dot positions (constant speed, no heartbeat variation)
        effective_speeds = strip_data['speeds'] * base_speed
        strip_data['positions'] += effective_speeds * dt
        strip_data['positions'] %= strip_length  # Wrap around
        
        # Vectorized dot rendering
        dot_indices = strip_data['positions'].astype(int)
        
        # Create mask for brown vs green regions
        brown_mask = dot_indices < 100
        green_mask = ~brown_mask
        
        # Set brown colors for dots in brown region
        if np.any(brown_mask):
            brown_dots = np.where(brown_mask)[0]
            buffer[dot_indices[brown_dots], :3] = strip_data['brown_colors'][brown_dots]
        
        # Set green colors for dots in green region  
        if np.any(green_mask):
            green_dots = np.where(green_mask)[0]
            buffer[dot_indices[green_dots], :3] = strip_data['green_colors'][green_dots]
        
        # Add random twinkles using forest color scheme
        num_new_twinkles = min(twinkles_per_frame, strip_length)
        if num_new_twinkles > 0:
            twinkle_indices = np.random.choice(strip_length, num_new_twinkles, replace=False)
            
            # Determine color palette based on pixel position
            for idx in twinkle_indices:
                if idx < 100:
                    # Use brown color palette
                    color_hsv = brown_colors[np.random.randint(len(brown_colors))]
                else:
                    # Use green color palette
                    color_hsv = green_colors[np.random.randint(len(green_colors))]
                
                h, s, v = color_hsv
                # Make twinkles brighter than the base colors
                r, g, b = hsv_to_rgb_vectorized(h, s * 0.8, min(1.0, v * 1.3))
                
                # Set the twinkle pixel to full brightness
                buffer[idx, 0] = max(buffer[idx, 0], r)
                buffer[idx, 1] = max(buffer[idx, 1], g)
                buffer[idx, 2] = max(buffer[idx, 2], b)
        
        buffer[:, 3] = fade_alpha  # Set alpha for entire strip

def GS_sunrise(instate, outstate):
    """
    Enhanced sunrise pattern with night phase featuring slowly blinking stars.
    Cycle: Night -> Sunrise -> Day -> Sunset -> Night
    During night phase, shows ~60 randomly placed, slowly blinking stars per strip.
    """
    name = 'sunrise'
    buffers = outstate['buffers']

    if instate['count'] == 0:
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['cycle_duration'] = 60.0  # Time for a complete cycle (seconds)
        instate['stars'] = {}  # Star data per strip
        
        # Phase durations (in fraction of cycle)
        instate['night_duration'] = 0.25      # 30 seconds
        instate['sunrise_duration'] = 0.25    # 30 seconds  
        instate['day_duration'] = 0.25        # 30 seconds
        instate['sunset_duration'] = 0.25     # 30 seconds
        instate['star_colors'] = np.array([
            [0.086, 0.727, 0.761],    # Pure red
            [0.159, 0.932, 0.459],    # Pure yellow  
            [0.400, 0.400, 0.196],    # Pure green
            [0.574, 0.902, 0.561],    # Pure cyan
            [0.983, 0.660, 0.635],    # Pure blue
            [0.050, 0.932, 0.463],    # Pure magenta
        ])
        
        return

    if instate['count'] == -1:
        buffers.generator_alphas[name] = 0
        return

    sound_level = outstate.get('sound_level', 1.0)
    
    # Calculate fade in/out over first and last 5 seconds
    elapsed_time = instate['elapsed_time']
    remaining_time = instate['duration'] - elapsed_time
    
    fade_alpha = 1.0
    if elapsed_time < 5.0:
        fade_alpha = elapsed_time / 5.0
    elif remaining_time < 5.0:
        fade_alpha = remaining_time / 5.0
    
    buffers.generator_alphas[name] = fade_alpha
    
    if fade_alpha < 0.01:
        return
    
    # Calculate cycle position and determine current phase
    current_time = outstate['current_time']
    cycle_position = (instate['elapsed_time'] % instate['cycle_duration']) / instate['cycle_duration']
    
    # Determine current phase
    if cycle_position < instate['night_duration']:
        # Night phase (0.0 - 0.25)
        phase = 'night'
        phase_progress = cycle_position / instate['night_duration']
    elif cycle_position < instate['night_duration'] + instate['sunrise_duration']:
        # Sunrise phase (0.25 - 0.5)
        phase = 'sunrise'
        phase_progress = (cycle_position - instate['night_duration']) / instate['sunrise_duration']
    elif cycle_position < instate['night_duration'] + instate['sunrise_duration'] + instate['day_duration']:
        # Day phase (0.5 - 0.75)
        phase = 'day'
        phase_progress = (cycle_position - instate['night_duration'] - instate['sunrise_duration']) / instate['day_duration']
    else:
        # Sunset phase (0.75 - 1.0)
        phase = 'sunset'
        phase_progress = (cycle_position - instate['night_duration'] - instate['sunrise_duration'] - instate['day_duration']) / instate['sunset_duration']
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Process each strip
    for strip_id, buffer in pattern_buffers.items():
        strip_length = len(buffer)
        # Calculate continuous sun position and size
        # Calculate continuous sun position and size
        max_position = min(275, strip_length - 1)
        
        if phase == 'night':
            sun_center = 0
            sun_radius = 0
        elif phase == 'sunrise':
            # Sun grows from pixel 0 outward
            sun_radius = max(1, int(40 * phase_progress))  # Radius grows with progress
            sun_center = sun_radius - 1  # Center positioned so left edge touches pixel 0
            # As it grows, it also moves toward final position
            target_center = int(max_position * 0.85)  # Target center position
            sun_center = int((sun_radius - 1) + (target_center - (sun_radius - 1)) * phase_progress)
        elif phase == 'day':
            # Sun at full size, centered around middle of strip
            sun_center = int(max_position * 0.85)
            sun_radius = 40*(1+(np.sin(phase_progress*np.pi)))
        elif phase == 'sunset':
            # Sun shrinks back toward pixel 0 (reverse of sunrise)
            shrink_progress = 1.0 - 1.5*phase_progress  # 1.0 to 0.0 during sunset
            sun_radius = max(1, int(40 * shrink_progress)) if shrink_progress > 0 else 0
            if sun_radius > 0:
                target_center = int(max_position * 0.85)
                sun_center = int((sun_radius - 1) + (target_center - (sun_radius - 1)) * shrink_progress)
            else:
                sun_center = 0
        
        # Initialize stars for this strip if needed
        if strip_id not in instate['stars']:
            num_stars = 60
            star_positions = np.random.choice(strip_length, size=min(num_stars, strip_length), replace=False)
            
            # Assign each star a random color from the palette
            color_indices = np.random.choice(len(instate['star_colors']), len(star_positions))
            star_rgb_colors = np.zeros((len(star_positions), 3))
            
            for i, color_idx in enumerate(color_indices):
                h, s, v = instate['star_colors'][color_idx]
                r, g, b = hsv_to_rgb_vectorized(h, s, v)
                star_rgb_colors[i] = [r, g, b]
            
            # Each star has a position, blink phase offset, blink frequency, and individual color
            instate['stars'][strip_id] = {
                'positions': star_positions,
                'phases': np.random.uniform(0, 2*np.pi, len(star_positions)),  # Phase offsets
                'frequencies': np.random.uniform(0.1, 0.3, len(star_positions)),  # Blink frequencies (Hz)
                'colors': star_rgb_colors  # Individual RGB colors for each star
            }
        
        # Initialize buffer (start with black)
        stars_data = instate['stars'][strip_id]

        buffer[:] = [0.0, 0.0, 0.0, 0.0]
        
        if phase == 'night':
            # Pure night phase - only stars
            star_positions = stars_data['positions']
            star_phases = stars_data['phases']
            star_frequencies = stars_data['frequencies']
            star_colors = stars_data['colors']
            
            # Calculate star brightness using sine waves
            star_brightness = 0.3 + 0.4 * np.sin(2 * np.pi * star_frequencies * current_time + star_phases)
            star_brightness = np.clip(star_brightness, 0.1, 0.7)  # Ensure minimum visibility
            
            # Set individual star colors
            buffer[star_positions, 0] = star_colors[:, 0] * star_brightness  # Red
            buffer[star_positions, 1] = star_colors[:, 1] * star_brightness  # Green  
            buffer[star_positions, 2] = star_colors[:, 2] * star_brightness  # Blue
            buffer[star_positions, 3] = star_brightness                      # Alpha
     # Alpha
            
        elif phase == 'sunrise':
            # Gradual transition from night to day
            if phase_progress < 0.3:
                # Early sunrise - just starting to lighten, stars still prominent
                early_progress = phase_progress / 0.3
                
                # Very subtle sky lightening
                sky_alpha = 0.1 * early_progress
                sky_r = 0.05 * early_progress
                sky_g = 0.03 * early_progress  
                sky_b = 0.08 * early_progress
                
                # Set very dim sky glow
                buffer[:, 0] = sky_r
                buffer[:, 1] = sky_g
                buffer[:, 2] = sky_b
                buffer[:, 3] = sky_alpha
                
                # Stars still very prominent (fade only slightly)
                star_alpha = 1.0 - (0.2 * early_progress)
                
            elif phase_progress < 0.7:
                # Mid sunrise - sun starts to appear, more sky color
                mid_progress = (phase_progress - 0.3) / 0.4
                
                # More noticeable sky colors
                sky_r = 0.05 + (0.4 * mid_progress)
                sky_g = 0.03 + (0.2 * mid_progress)
                sky_b = 0.08 + (0.1 * mid_progress)
                sky_alpha = 0.1 + (0.3 * mid_progress)
                
                # Set base sky colors
                buffer[:, 0] = sky_r
                buffer[:, 1] = sky_g
                buffer[:, 2] = sky_b
                buffer[:, 3] = sky_alpha
                
                # Stars start to fade more noticeably
                star_alpha = 0.8 - (0.6 * mid_progress)
                
            else:
                # Late sunrise - full sun emergence, final star fade
                late_progress = (phase_progress - 0.7) / 0.3
                
                # Full sunrise sky colors - transition to day sky colors
                sky_r = 0.45 - (0.15 * late_progress)  # 0.45 -> 0.3 (day sky)
                sky_g = 0.23 + (0.37 * late_progress)  # 0.23 -> 0.6 (day sky)
                sky_b = 0.18 + (0.72 * late_progress)  # 0.18 -> 0.9 (day sky)
                
                # Set sky colors for all pixels first
                buffer[:, 0] = sky_r
                buffer[:, 1] = sky_g
                buffer[:, 2] = sky_b
                buffer[:, 3] = 0.4 + 0.2 * late_progress
                
                # Final star fade
                star_alpha = 0.2 * (1.0 - late_progress)

            
            # Always render sun if it has size (remove show_sun logic)
            if sun_radius > 0:
                positions = np.arange(strip_length)
                distances = np.abs(positions - sun_center)
                inside_sun_mask = distances <= sun_radius
                
                if np.any(inside_sun_mask):
                    normalized_distances = distances[inside_sun_mask] / sun_radius
                    intensities = 1.0 - normalized_distances ** 2
                    
                    # Sun brightness increases with phase progress
                    sun_brightness = max(min(1.0, phase_progress * 1.5),0.5)
                    
                    buffer[inside_sun_mask, 0] = 1.0 * sun_brightness
                    buffer[inside_sun_mask, 1] = (0.7 + (0.3 * intensities)) * sun_brightness
                    buffer[inside_sun_mask, 2] = (0.2 + (0.3 * intensities)) * sun_brightness
                    buffer[inside_sun_mask, 3] = (0.8 + (0.2 * intensities)) * sun_brightness
            
            # Apply stars throughout all sunrise sub-phases
            if star_alpha > 0.01:
                star_positions = stars_data['positions']
                star_phases = stars_data['phases']
                star_frequencies = stars_data['frequencies']
                star_colors = stars_data['colors']
                
                star_brightness = 0.3 + 0.4 * np.sin(2 * np.pi * star_frequencies * current_time + star_phases)
                star_brightness = np.clip(star_brightness * star_alpha, 0.0, 0.7)
                
                # Only show stars not covered by sun
                if sun_radius > 0:
                    positions = np.arange(strip_length)
                    distances = np.abs(positions - sun_center)
                    sun_coverage_mask = distances <= sun_radius
                    visible_stars = ~np.isin(star_positions, np.where(sun_coverage_mask)[0])
                else:
                    # No sun to worry about
                    visible_stars = np.ones(len(star_positions), dtype=bool)
                
                visible_star_positions = star_positions[visible_stars]
                visible_star_brightness = star_brightness[visible_stars]
                visible_star_colors = star_colors[visible_stars]
                
                if len(visible_star_positions) > 0:
                    # Blend stars with existing sky using individual colors
                    buffer[visible_star_positions, 0] = np.maximum(buffer[visible_star_positions, 0], visible_star_colors[:, 0] * visible_star_brightness)
                    buffer[visible_star_positions, 1] = np.maximum(buffer[visible_star_positions, 1], visible_star_colors[:, 1] * visible_star_brightness)
                    buffer[visible_star_positions, 2] = np.maximum(buffer[visible_star_positions, 2], visible_star_colors[:, 2] * visible_star_brightness)
                    buffer[visible_star_positions, 3] = np.maximum(buffer[visible_star_positions, 3], visible_star_brightness)


        elif phase == 'day':
            # Full day phase
            positions = np.arange(strip_length)
            distances = np.abs(positions - sun_center)
            inside_sun_mask = distances <= sun_radius
            
            # Day sky colors (light blue)
            sky_r = 0.3
            sky_g = 0.6
            sky_b = 0.9
            
            # Set sun colors
            if np.any(inside_sun_mask):
                normalized_distances = distances[inside_sun_mask] / sun_radius
                intensities = 1.0 - normalized_distances ** 2
                
                buffer[inside_sun_mask, 0] = 1.0
                buffer[inside_sun_mask, 1] = 0.7 + (0.3 * intensities)
                buffer[inside_sun_mask, 2] = 0.2 + (0.3 * intensities)
                buffer[inside_sun_mask, 3] = 0.8 + (0.2 * intensities)
            
            # Set sky colors
            outside_sun_mask = ~inside_sun_mask
            if np.any(outside_sun_mask):
                buffer[outside_sun_mask, 0] = sky_r
                buffer[outside_sun_mask, 1] = sky_g
                buffer[outside_sun_mask, 2] = sky_b
                buffer[outside_sun_mask, 3] = 0.6
                
        elif phase == 'sunset':
            # Transition from day to night (mirror of sunrise)
            positions = np.arange(strip_length)
            
            # Sunset sky colors (reverse of sunrise)
            if phase_progress < 0.3:
                # Early sunset - still mostly day colors
                early_progress = phase_progress / 0.3
                sky_r = 0.3 + (0.6 * early_progress)
                sky_g = 0.6 - (0.1 * early_progress)
                sky_b = 0.9 - (0.7 * early_progress)
                sky_alpha = 0.6
            elif phase_progress < 0.7:
                # Mid sunset - orange colors
                mid_progress = (phase_progress - 0.3) / 0.4
                sky_r = 0.9 - (0.45 * mid_progress)
                sky_g = 0.5 - (0.17 * mid_progress)
                sky_b = 0.2 - (0.12 * mid_progress)
                sky_alpha = 0.6 - (0.2 * mid_progress)
            else:
                # Late sunset - fade to black
                late_progress = (phase_progress - 0.7) / 0.3
                sky_r = 0.45 * (1.0 - late_progress)
                sky_g = 0.33 * (1.0 - late_progress)
                sky_b = 0.08 * (1.0 - late_progress)
                sky_alpha = 0.4 * (1.0 - late_progress)
            
            # Set sky colors first
            buffer[:, 0] = sky_r
            buffer[:, 1] = sky_g
            buffer[:, 2] = sky_b
            buffer[:, 3] = sky_alpha
            
            # Always render sun if it has size
            if sun_radius > 0:
                distances = np.abs(positions - sun_center)
                inside_sun_mask = distances <= sun_radius
                
                if np.any(inside_sun_mask):
                    normalized_distances = distances[inside_sun_mask] / sun_radius
                    intensities = 1.0 - normalized_distances ** 2
                    
                    # Sun brightness decreases as it shrinks
                    sun_brightness = 1.0 - phase_progress
                    
                    buffer[inside_sun_mask, 0] = 1.0 * sun_brightness
                    buffer[inside_sun_mask, 1] = (0.7 + (0.3 * intensities)) * sun_brightness
                    buffer[inside_sun_mask, 2] = (0.2 + (0.3 * intensities)) * sun_brightness
                    buffer[inside_sun_mask, 3] = (0.8 + (0.2 * intensities)) * sun_brightness
            
            # Add fading-in stars (start appearing in mid-sunset)
            star_alpha = 0.0
            if phase_progress > 0.3:
                star_alpha = (phase_progress - 0.3) / 0.7
            
            if star_alpha > 0.01:
                star_positions = stars_data['positions']
                star_phases = stars_data['phases']
                star_frequencies = stars_data['frequencies']
                star_colors = stars_data['colors']
                
                star_brightness = 0.3 + 0.4 * np.sin(2 * np.pi * star_frequencies * current_time + star_phases)
                star_brightness = np.clip(star_brightness * star_alpha, 0.0, 0.7)
                
                # Only show stars not covered by sun
                if sun_radius > 0:
                    distances = np.abs(positions - sun_center)
                    sun_coverage_mask = distances <= sun_radius
                    visible_stars = ~np.isin(star_positions, np.where(sun_coverage_mask)[0])
                else:
                    visible_stars = np.ones(len(star_positions), dtype=bool)
                
                visible_star_positions = star_positions[visible_stars]
                visible_star_brightness = star_brightness[visible_stars]
                visible_star_colors = star_colors[visible_stars]
                
                if len(visible_star_positions) > 0:
                    # Blend stars with existing sky using individual colors
                    buffer[visible_star_positions, 0] = np.maximum(buffer[visible_star_positions, 0], visible_star_colors[:, 0] * visible_star_brightness)
                    buffer[visible_star_positions, 1] = np.maximum(buffer[visible_star_positions, 1], visible_star_colors[:, 1] * visible_star_brightness)
                    buffer[visible_star_positions, 2] = np.maximum(buffer[visible_star_positions, 2], visible_star_colors[:, 2] * visible_star_brightness)
                    buffer[visible_star_positions, 3] = np.maximum(buffer[visible_star_positions, 3], visible_star_brightness)

        
        # Add gentle ambient variation for more natural look
        if phase != 'night':
            positions = np.arange(strip_length)
            variations = 0.05 * np.sin(positions * 0.1 + current_time * 0.5)
            buffer[:, :3] += variations[:, np.newaxis]
        
        # Ensure color values are in valid range
        buffer[:, :3] = np.clip(buffer[:, :3], 0.0, 1.0)
        buffer[:, 3] = np.clip(buffer[:, 3], 0.0, 1.0)
        
        # Apply final fade alpha
        buffer[:, 3] *= fade_alpha