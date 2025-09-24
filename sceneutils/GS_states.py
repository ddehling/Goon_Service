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
        [0.0, 1.0, 1.0],    # Pure red
        [0.17, 1.0, 1.0],   # Pure yellow  
        [0.33, 1.0, 1.0],   # Pure green
        [0.5, 1.0, 1.0],    # Pure cyan
        [0.67, 1.0, 1.0],   # Pure blue
        [0.83, 1.0, 1.0],   # Pure magenta
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
                if i < 6:  # First 3 dots are red-ish
                    h = np.random.uniform(0.9, 1.1) % 1.0  # Red hues
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
    Generator function that creates a curious and playful-themed pattern across all strips.
    
    Features:
    1. Global alpha controlled by outstate['control_curious'] value
    2. Vibrant, saturated color palette with blues, greens, reds, oranges, purples, and whites
    3. Moving color regions that create dynamic patterns
    4. Fast movement with playful characteristics
    
    Uses HSV colorspace for color generation and blending.
    """
    name = 'curious_playful'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['color_regions'] = {}    # Track color regions per strip
        instate['color_shift'] = 0.0     # Global color shift for variation
        instate['spots_state'] = {}   
        
        # Color palette (HSV values) - more saturated colors
        instate['colors'] = {
            'bright_blue': [0.6, 0.7, 0.95],      # Bright blue
            'vibrant_green': [0.3, 0.8, 0.9],     # Vibrant green
            'bright_red': [0.0, 0.8, 0.95],       # Bright red
            'vibrant_orange': [0.08, 0.9, 0.95],  # Vibrant orange
            'rich_purple': [0.8, 0.8, 0.9],       # Rich purple
            'hot_pink': [0.9, 0.75, 0.95],        # Hot pink
            'turquoise': [0.45, 0.8, 0.95],       # Turquoise
            'bright_yellow': [0.15, 0.8, 0.95],   # Bright yellow
            'pure_white': [0.0, 0.0, 1.0]         # Pure white
        }
        
        # Motion parameters
        instate['region_speed_multiplier'] = 1.0  # Global speed control
        
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        return

    # Get curious level from outstate (default to 0)
    curious_level = 1
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = curious_level
    
    # Skip rendering if alpha is too low
    if curious_level < 0.01:
        return
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if instate['elapsed_time']<5:
        fade_alpha = instate['elapsed_time'] / 5.0
        fade_alpha = max(0.0, fade_alpha)
        buffers.generator_alphas[name] = fade_alpha * curious_level
    if remaining_time < 5.0:
        fade_alpha = remaining_time / 5.0
        fade_alpha = max(0.0, fade_alpha)
        buffers.generator_alphas[name] = fade_alpha * curious_level
    
    # Get delta time for animation calculations
    delta_time = outstate['current_time'] - outstate['last_time']
    
    # Update global color shift for variation - slow cycle through hues
    instate['color_shift'] = (instate['color_shift'] + 0.05 * delta_time) % 1.0
    
    # Adjust global region speed based on curious level - more curious = faster
    instate['region_speed_multiplier'] = 1.0 + curious_level  # 1.0-2.0x speed range
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Process each buffer based on strip type
    for strip_id, buffer in pattern_buffers.items():
        # Skip if strip doesn't exist in manager
        if strip_id not in strip_manager.strips:
            continue
            
        strip = strip_manager.get_strip(strip_id)
        strip_length = len(buffer)
            
    
            
        # Initialize color regions for this strip if not already done
        if strip_id not in instate['color_regions']:
            instate['color_regions'][strip_id] = []
            
            # Create initial color regions - divide strip into segments
            num_regions = max(2, min(6, strip_length // 30))  # 2-6 regions depending on strip length
            region_size = strip_length / num_regions
            
            for i in range(num_regions):
                # Create a color region with position centered in each segment
                center = (i + 0.5) * region_size
                
                # Get random color from palette
                color_name = np.random.choice(list(instate['colors'].keys()))
                h, s, v = instate['colors'][color_name]
                
                # Random direction for movement
                direction = 1 if np.random.random() < 0.5 else -1
                
                # Create region
                region = {
                    'center': center,
                    'size': region_size * 0.7,  # Slightly smaller than segment for initial gaps
                    'h': h,
                    's': s,
                    'v': v,
                    'speed': 5 + np.random.random() * 15,  # 5-20 pixels per second
                    'direction': direction,
                    'wobble_freq': 0.5 + np.random.random() * 1.5,  # 0.5-2.0 Hz
                    'wobble_amount': 0.2 + np.random.random() * 0.4,  # 0.2-0.6 size wobble
                    'wobble_offset': np.random.random() * 6.28,  # Random phase
                    'lifetime': 0,  # Time tracking for color changes
                    'color_change_time': 5 + np.random.random() * 10  # 5-15 seconds between color changes
                }
                
                instate['color_regions'][strip_id].append(region)
        
        # Update regions and check for collisions
        for i, region in enumerate(instate['color_regions'][strip_id]):
            # Update region position based on speed and direction
            effective_speed = region['speed'] * instate['region_speed_multiplier']
            region['center'] += effective_speed * region['direction'] * delta_time
            
            # Add wobble to size for a playful effect
            time_factor = outstate['current_time'] * region['wobble_freq']
            size_wobble = 1.0 + region['wobble_amount'] * np.sin(time_factor + region['wobble_offset'])
            region['effective_size'] = region['size'] * size_wobble  # Store for rendering
            
            # Handle wrapping around strip boundaries
            if region['center'] >= strip_length:
                region['center'] -= strip_length
            elif region['center'] < 0:
                region['center'] += strip_length
            
            # Handle collision with other regions - check if regions are too close
            for j, other_region in enumerate(instate['color_regions'][strip_id]):
                if i != j:  # Don't compare to self
                    # Calculate distance considering strip wrapping
                    direct_dist = abs(region['center'] - other_region['center'])
                    wrapped_dist = strip_length - direct_dist
                    distance = min(direct_dist, wrapped_dist)
                    
                    # Minimum allowed distance is sum of half sizes
                    min_distance = (region['effective_size'] + other_region.get('effective_size', other_region['size'])) * 0.5
                    
                    # If too close, reverse direction of both
                    if distance < min_distance * 0.8:  # 80% of minimum to create some bounce space
                        # Only reverse if moving toward each other
                        if ((region['center'] < other_region['center'] and region['direction'] > 0 and 
                             other_region['direction'] < 0) or
                            (region['center'] > other_region['center'] and region['direction'] < 0 and 
                             other_region['direction'] > 0)):
                            region['direction'] *= -1
                            other_region['direction'] *= -1
                            
                            # Add slight random speed variation on bounce
                            region['speed'] *= 0.9 + 0.2 * np.random.random()
                            other_region['speed'] *= 0.9 + 0.2 * np.random.random()
                            
                            # Keep speeds in reasonable range
                            region['speed'] = max(5, min(20, region['speed']))
                            other_region['speed'] = max(5, min(20, other_region['speed']))
            
            # Update lifetime and check for color change
            region['lifetime'] += delta_time
            if region['lifetime'] > region['color_change_time']:
                # Reset lifetime
                region['lifetime'] = 0
                
                # Choose a new color - avoid similar hue to neighbors
                available_colors = list(instate['colors'].keys())
                
                # Try to get neighboring regions (accounting for potential out-of-bounds)
                if len(instate['color_regions'][strip_id]) > 1:
                    # Find regions that are close by distance
                    neighbor_indices = []
                    for j, other in enumerate(instate['color_regions'][strip_id]):
                        if i != j:
                            direct_dist = abs(region['center'] - other['center'])
                            wrapped_dist = strip_length - direct_dist
                            distance = min(direct_dist, wrapped_dist)
                            
                            if distance < (region['size'] + other['size']) * 1.5:  # If close enough to be a neighbor
                                neighbor_indices.append(j)
                    
                    # If we have neighbors, try to avoid their colors
                    if neighbor_indices:
                        neighbor_hues = [instate['color_regions'][strip_id][j]['h'] for j in neighbor_indices]
                        
                        # Filter out colors with similar hue
                        filtered_colors = []
                        for color_name in available_colors:
                            h, s, v = instate['colors'][color_name]
                            is_similar = False
                            for n_hue in neighbor_hues:
                                # Check if hues are similar (considering wrap-around at 1.0)
                                hue_dist = min(abs(h - n_hue), 1.0 - abs(h - n_hue))
                                if hue_dist < 0.15:  # Consider similar if within 15% of hue space
                                    is_similar = True
                                    break
                            if not is_similar:
                                filtered_colors.append(color_name)
                        
                        # If we have filtered colors, use them, otherwise use all colors
                        if filtered_colors:
                            available_colors = filtered_colors
                
                # Choose a new color from available options
                new_color_name = np.random.choice(available_colors)
                h, s, v = instate['colors'][new_color_name]
                
                # Update region color
                region['h'] = h
                region['s'] = s
                region['v'] = v
                
                # Also randomize wobble parameters for variety
                region['wobble_freq'] = 0.5 + np.random.random() * 1.5
                region['wobble_amount'] = 0.2 + np.random.random() * 0.4
                region['wobble_offset'] = np.random.random() * 6.28
                
                # Set a new color change time
                region['color_change_time'] = 5 + np.random.random() * 10
        
        # -------- SIMPLIFIED RENDERING APPROACH --------
        # Initialize buffer with zeros
        buffer_hsv = np.zeros((strip_length, 4))  # [h, s, v, influence]
        
        # Render each region as a Gaussian-like distribution of influence
        pixels = np.arange(strip_length)
        
        for region in instate['color_regions'][strip_id]:
            # Calculate distance to center with wrapping
            direct_dist = np.abs(pixels - region['center'])
            wrapped_dist = strip_length - direct_dist
            distances = np.minimum(direct_dist, wrapped_dist)
            
            # Calculate influence using a Gaussian-like falloff
            sigma = region['effective_size'] / 2  # Standard deviation (half the size)
            influence = np.exp(-0.5 * (distances / sigma)**2)  # Gaussian-like falloff
            
            # Only apply where influence is significant
            mask = influence > 0.01
            
            # Add this region's contribution to the buffer
            # Additive blending for HSV values weighted by influence
            buffer_hsv[mask, 0] += region['h'] * influence[mask]  # Hue
            buffer_hsv[mask, 1] += region['s'] * influence[mask]  # Saturation
            buffer_hsv[mask, 2] += region['v'] * influence[mask]  # Value
            buffer_hsv[mask, 3] += influence[mask]  # Total influence for normalization
        
        # Normalize the HSV values by total influence
        has_influence = buffer_hsv[:, 3] > 0
        if np.any(has_influence):
            # Normalize hue, saturation, value by total influence
            buffer_hsv[has_influence, 0] /= buffer_hsv[has_influence, 3]
            buffer_hsv[has_influence, 1] /= buffer_hsv[has_influence, 3]
            buffer_hsv[has_influence, 2] /= buffer_hsv[has_influence, 3]
            
            # Wrap hue to 0-1 range
            buffer_hsv[:, 0] = buffer_hsv[:, 0] % 1.0
            
            # Clamp saturation and value to 0-1 range
            buffer_hsv[:, 1] = np.clip(buffer_hsv[:, 1], 0, 1)
            buffer_hsv[:, 2] = np.clip(buffer_hsv[:, 2], 0, 1)
            
            # Convert HSV to RGB
            rgb = np.zeros((strip_length, 3))
            r, g, b = hsv_to_rgb_vectorized(
                buffer_hsv[has_influence, 0], 
                buffer_hsv[has_influence, 1], 
                buffer_hsv[has_influence, 2]
            )
            
            # Set final RGB values
            rgb_buffer = np.zeros((strip_length, 4))  # [r, g, b, a]
            rgb_buffer[has_influence, 0] = r
            rgb_buffer[has_influence, 1] = g
            rgb_buffer[has_influence, 2] = b
            
            # Alpha based on influence (scale to reasonable range)
            rgb_buffer[has_influence, 3] = np.clip(buffer_hsv[has_influence, 3] * 0.5, 0, 1)
            
            # Add sparkles
            sparkle_chance = 0.02 * curious_level  # More sparkles when more curious
            sparkle_mask = np.random.random(strip_length) < sparkle_chance
            
            if np.any(sparkle_mask):
                # Create sparkles
                num_sparkles = np.sum(sparkle_mask)
                sparkle_h = np.random.random(num_sparkles)
                sparkle_s = np.full_like(sparkle_h, 0.2)  # Low saturation (white-ish)
                sparkle_v = np.ones_like(sparkle_h)  # Full brightness
                
                # Convert sparkles to RGB
                sr, sg, sb = hsv_to_rgb_vectorized(sparkle_h, sparkle_s, sparkle_v)
                
                # Add sparkles to the buffer
                sparkle_indices = np.where(sparkle_mask)[0]
                rgb_buffer[sparkle_indices, 0] = np.minimum(1.0, rgb_buffer[sparkle_indices, 0] + sr * 0.7)
                rgb_buffer[sparkle_indices, 1] = np.minimum(1.0, rgb_buffer[sparkle_indices, 1] + sg * 0.7)
                rgb_buffer[sparkle_indices, 2] = np.minimum(1.0, rgb_buffer[sparkle_indices, 2] + sb * 0.7)
                rgb_buffer[sparkle_indices, 3] = np.minimum(1.0, rgb_buffer[sparkle_indices, 3] + 0.3)
            
            # Copy from numpy array back to buffer
            mask = rgb_buffer[:, 3] > 0
            buffer[:] = 0  # Clear the original buffer
            buffer[mask] = rgb_buffer[mask]
