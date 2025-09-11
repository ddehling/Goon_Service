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
        




def GS_blood_flow(instate, outstate):
    """
    Blood flow simulation with white and red dots moving down strips.
    Red dots move faster than white dots, with a beating pattern for pulse effect.
    Dots leave fading trails behind them. Uses HSV color space.
    """
    name = 'blood_flow'
    buffers = outstate['buffers']

    if instate['count'] == 0:
        buffers.register_generator(name)
        # Initialize dot states and trail buffers for each strip
        if not hasattr(buffers, 'blood_flow_dots'):
            buffers.blood_flow_dots = {}
        if not hasattr(buffers, 'blood_flow_trails'):
            buffers.blood_flow_trails = {}
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
    
    # Heart beat pattern - using exponential decay for realistic pulse
    beat_period = 1.2  # seconds between beats
    beat_phase = (current_time % beat_period) / beat_period
    
    # Create a sharp pulse that decays quickly
    if beat_phase < 0.15:  # Sharp rise
        beat_intensity = (beat_phase / 0.15) ** 0.5
    else:  # Exponential decay
        decay_phase = (beat_phase - 0.15) / 0.85
        beat_intensity = np.exp(-decay_phase * 4)
    
    # Base speed multiplier + beat boost
    base_speed = 0.3
    beat_boost = beat_intensity * 3.5
    speed_multiplier = base_speed + beat_boost
    
    for strip_id, buffer in pattern_buffers.items():
        strip_length = len(buffer)
        
        # Initialize trail buffer for this strip (HSV + alpha)
        if strip_id not in buffers.blood_flow_trails:
            buffers.blood_flow_trails[strip_id] = np.zeros((strip_length, 4))  # H, S, V, A
        
        trail_buffer = buffers.blood_flow_trails[strip_id]
        
        # Initialize dots for this strip if needed
        if strip_id not in buffers.blood_flow_dots:
            num_white_dots = 3
            num_red_dots = 8
            
            dots = []
            # White dots (slower)
            for _ in range(num_white_dots):
                dots.append({
                    'position': np.random.uniform(0, strip_length),
                    'base_speed': np.random.uniform(8, 15),  # pixels per second
                    'hue': np.random.uniform(0.08, 0.12),  # Warm white/yellow hues
                    'saturation': np.random.uniform(0.1, 0.3),  # Low saturation for white-ish
                    'size': np.random.uniform(1.5, 3.0)
                })
            
            # Red dots (faster)  
            for _ in range(num_red_dots):
                dots.append({
                    'position': np.random.uniform(0, strip_length),
                    'base_speed': np.random.uniform(20, 35),  # pixels per second
                    'hue': np.random.uniform(0.90, 1.05) % 1.0,  # Red hues (wrap around)
                    'saturation': np.random.uniform(0.8, 1.0),  # High saturation for vibrant red
                    'size': np.random.uniform(2.0, 4.0)
                })
            
            buffers.blood_flow_dots[strip_id] = {
                'dots': dots,
                'last_time': current_time
            }
        
        strip_data = buffers.blood_flow_dots[strip_id]
        dt = current_time - strip_data['last_time']
        strip_data['last_time'] = current_time
        
        # Fade trails over time - fade value (brightness) and saturation
        trail_fade_rate = 0.25  # Per second (lower = longer trails)
        fade_factor = trail_fade_rate ** dt
        trail_buffer[:, 2] *= fade_factor  # Fade value (brightness)
        trail_buffer[:, 1] *= (fade_factor ** 0.5)  # Fade saturation more slowly
        
        # Start with dark background in HSV
        bg_hue = 0.0  # Red hue
        bg_saturation = 0.8
        bg_value = 0.05
        bg_r, bg_g, bg_b = hsv_to_rgb_vectorized(bg_hue, bg_saturation, bg_value)
        buffer[:] = [bg_r, bg_g, bg_b, fade_alpha]
        
        # Create working HSV buffer for the strip
        hsv_buffer = np.zeros((strip_length, 3))  # H, S, V
        hsv_buffer[:, 0] = bg_hue
        hsv_buffer[:, 1] = bg_saturation  
        hsv_buffer[:, 2] = bg_value
        
        # Update and render dots
        for dot in strip_data['dots']:
            # Store previous position for trail
            prev_position = dot['position']
            
            # Update position
            actual_speed = dot['base_speed'] * speed_multiplier
            dot['position'] += actual_speed * dt
            
            # Reset to beginning when reaching end
            if dot['position'] >= strip_length:
                dot['position'] = dot['position'] % strip_length
            
            # Dot HSV values with beat intensity affecting brightness
            dot_hue = dot['hue']
            dot_saturation = dot['saturation']
            base_value = 0.8
            dot_value = base_value + (beat_intensity * 0.3)  # Pulse brighter on beat
            
            # Add trail segments between previous and current position
            trail_intensity = 0.4  # How bright the trail starts
            positions_to_trail = []
            
            # Handle wrapping case
            if dot['position'] < prev_position:  # Wrapped around
                # Trail from prev_position to end
                for pos in np.arange(prev_position, strip_length, 0.5):
                    positions_to_trail.append(pos)
                # Trail from start to current position
                for pos in np.arange(0, dot['position'], 0.5):
                    positions_to_trail.append(pos)
            else:
                # Normal case - trail from prev to current
                for pos in np.arange(prev_position, dot['position'], 0.5):
                    positions_to_trail.append(pos)
            
            # Add trail points to trail buffer
            for trail_pos in positions_to_trail:
                trail_idx = int(trail_pos)
                if 0 <= trail_idx < strip_length:
                    # Additive blending in HSV space
                    trail_add_value = dot_value * trail_intensity * 0.3
                    
                    # If existing trail is dim, use new color, otherwise blend
                    if trail_buffer[trail_idx, 2] < 0.1:  # Low existing brightness
                        trail_buffer[trail_idx, 0] = dot_hue
                        trail_buffer[trail_idx, 1] = dot_saturation * 0.7  # Slightly desaturated
                        trail_buffer[trail_idx, 2] = trail_add_value
                    else:
                        # Blend hues carefully (handle wraparound)
                        existing_hue = trail_buffer[trail_idx, 0]
                        hue_diff = abs(dot_hue - existing_hue)
                        if hue_diff > 0.5:  # Wrap around case
                            if dot_hue > existing_hue:
                                existing_hue += 1.0
                            else:
                                dot_hue += 1.0
                        
                        weight = trail_add_value / (trail_buffer[trail_idx, 2] + trail_add_value + 1e-6)
                        trail_buffer[trail_idx, 0] = ((1-weight) * existing_hue + weight * dot_hue) % 1.0
                        trail_buffer[trail_idx, 1] = min(1.0, trail_buffer[trail_idx, 1] + dot_saturation * 0.1)
                        trail_buffer[trail_idx, 2] = min(1.0, trail_buffer[trail_idx, 2] + trail_add_value)
            
            # Render current dot position with gaussian falloff
            center = int(dot['position'])
            size = dot['size']
            

            for offset in range(-int(size*2), int(size*2) + 1):
                pixel_pos = center + offset
                if 0 <= pixel_pos < strip_length:
                    distance = abs(offset)
                    intensity = np.exp(-(distance**2) / (2 * (size/2)**2))
                    
                    # Add bright dot to HSV buffer
                    bright_value = dot_value * intensity
                    
                    if hsv_buffer[pixel_pos, 2] < bright_value:
                        # Replace with brighter dot
                        hsv_buffer[pixel_pos, 0] = dot_hue
                        hsv_buffer[pixel_pos, 1] = dot_saturation
                        hsv_buffer[pixel_pos, 2] = bright_value
                    elif hsv_buffer[pixel_pos, 2] > 0.1:
                        # Blend colors
                        weight = bright_value / (hsv_buffer[pixel_pos, 2] + bright_value + 1e-6)
                        # Handle hue blending with wraparound
                        existing_hue = hsv_buffer[pixel_pos, 0]
                        dot_hue_blend = dot_hue  # Initialize with original value
                        
                        hue_diff = abs(dot_hue - existing_hue)
                        if hue_diff > 0.5:
                            if dot_hue > existing_hue:
                                existing_hue += 1.0
                            else:
                                dot_hue_blend = dot_hue + 1.0
                        
                        hsv_buffer[pixel_pos, 0] = ((1-weight) * existing_hue + weight * dot_hue_blend) % 1.0
                        hsv_buffer[pixel_pos, 1] = min(1.0, hsv_buffer[pixel_pos, 1] + dot_saturation * weight)
                        hsv_buffer[pixel_pos, 2] = min(1.0, hsv_buffer[pixel_pos, 2] + bright_value * 0.5)

        
        # Combine trail buffer with main HSV buffer
        trail_brighter = trail_buffer[:, 2] > hsv_buffer[:, 2]
        trail_visible = (trail_buffer[:, 2] > 0.05) & (~trail_brighter)
        
        # Case 1: Trail is brighter - replace entirely
        hsv_buffer[trail_brighter] = trail_buffer[trail_brighter, :3]
        
        # Case 2: Trail is visible but dimmer - blend
        if np.any(trail_visible):
            # Calculate blend weights
            trail_weights = trail_buffer[trail_visible, 2] / (
                hsv_buffer[trail_visible, 2] + trail_buffer[trail_visible, 2] + 1e-6
            )
            
            # Handle hue blending with wraparound
            existing_hues = hsv_buffer[trail_visible, 0]
            trail_hues = trail_buffer[trail_visible, 0]
            
            # Calculate hue differences
            hue_diffs = np.abs(trail_hues - existing_hues)
            wraparound_mask = hue_diffs > 0.5
            
            # Adjust hues for wraparound blending
            existing_hues_adj = existing_hues.copy()
            trail_hues_adj = trail_hues.copy()
            
            # Where wraparound is needed
            trail_larger = wraparound_mask & (trail_hues > existing_hues)
            existing_larger = wraparound_mask & (trail_hues <= existing_hues)
            
            existing_hues_adj[trail_larger] += 1.0
            trail_hues_adj[existing_larger] += 1.0
            
            # Blend hues
            blended_hues = ((1 - trail_weights) * existing_hues_adj + 
                           trail_weights * trail_hues_adj) % 1.0
            
            # Update HSV buffer
            hsv_buffer[trail_visible, 0] = blended_hues
            hsv_buffer[trail_visible, 1] = np.minimum(1.0, 
                hsv_buffer[trail_visible, 1] + 
                trail_buffer[trail_visible, 1] * trail_weights
            )
            hsv_buffer[trail_visible, 2] = np.minimum(1.0,
                hsv_buffer[trail_visible, 2] + 
                trail_buffer[trail_visible, 2] * 0.7
            )
        # Convert HSV buffer to RGB and set final buffer
        r_values, g_values, b_values = hsv_to_rgb_vectorized(
            hsv_buffer[:, 0], hsv_buffer[:, 1], hsv_buffer[:, 2]
        )
        
        buffer[:, 0] = r_values
        buffer[:, 1] = g_values  
        buffer[:, 2] = b_values
        buffer[:, 3] = fade_alpha*(b_values>0.1)
        


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
        noise_intensity = 0.3  # Low intensity
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

