from sceneutils.imgutils import *  # noqa: F403
import numpy as np
from pathlib import Path
from colorsys import hsv_to_rgb

def GS_prototype_example(instate, outstate):
    """
    Prototype rendering function template.
    
    Features:
    1. Global alpha controlled by outstate['control_prototype'] value
    2. Fade in/out over 5 seconds at beginning and end
    3. Simple strip-based rendering example with color cycling
    4. Different effects based on strip groups
    """
    name = 'prototype_example'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['color_cycle_time'] = 0.0        # Track time for color cycling
        instate['wave_offset'] = 0.0             # Wave animation offset
        instate['strips_initialized'] = False    # Track if we've processed strips
        instate['strip_colors'] = {}             # Store per-strip color info
        instate['spots_state'] = {}              # Special state for spots strips
        
        # Color palette (HSV values for easy manipulation)
        instate['colors'] = {
            'primary': [0.6, 0.8, 0.9],    # Blue
            'secondary': [0.3, 0.7, 0.8],  # Green  
            'accent': [0.9, 0.9, 0.9],     # Pink
            'neutral': [0.1, 0.5, 0.7]     # Yellow
        }
        
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        
        # Clean up any allocated resources
        if 'strip_colors' in instate:
            instate['strip_colors'].clear()
        if 'spots_state' in instate:
            instate['spots_state'].clear()
            
        return

    # Get control level from outstate (0-100, convert to 0.0-1.0)
    control_level = 1
    
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
    final_alpha = control_level * max(0.0, fade_alpha)
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = final_alpha
    
    # Skip rendering if alpha is too low
    if final_alpha < 0.01:
        return
    
    # Get delta time for animation calculations
    delta_time = outstate['current_time'] - outstate['last_time']
    
    # Update animation timers
    instate['color_cycle_time'] += delta_time
    instate['wave_offset'] += delta_time * 2.0  # Wave speed multiplier
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Initialize strip-specific data on first render pass
    if not instate['strips_initialized']:
        for strip_id in pattern_buffers.keys():
            if strip_id in strip_manager.strips:
                strip = strip_manager.get_strip(strip_id)
                
                # Assign colors based on strip groups
                if 'base' in strip.groups:
                    color_key = 'primary'
                elif 'heart' in strip.groups:
                    color_key = 'accent'
                elif 'spine' in strip.groups:
                    color_key = 'secondary'
                else:
                    color_key = 'neutral'
                
                instate['strip_colors'][strip_id] = {
                    'base_color': color_key,
                    'phase_offset': np.random.random() * 2 * np.pi,  # Random phase for variety
                    'speed_multiplier': 0.5 + np.random.random() * 1.5  # Random speed 0.5-2.0x
                }
        
        instate['strips_initialized'] = True
    
    # Process each buffer based on strip type
    for strip_id, buffer in pattern_buffers.items():
        # Skip if strip doesn't exist in manager
        if strip_id not in strip_manager.strips:
            continue
            
        strip = strip_manager.get_strip(strip_id)
        strip_length = len(buffer)
        strip_groups = strip.groups
        
        # Handle spots strips with special slow transitions
        if 'spots' in strip_groups:
            # Initialize spots state if needed
            if strip_id not in instate['spots_state']:
                instate['spots_state'][strip_id] = {
                    'current_color': np.random.choice(list(instate['colors'].keys())),
                    'target_color': np.random.choice(list(instate['colors'].keys())),
                    'transition_progress': 0.0,
                    'transition_duration': 8.0,  # 8 seconds per transition
                    'glow_phase': np.random.random() * 2 * np.pi
                }
            
            spots_state = instate['spots_state'][strip_id]
            
            # Update transition
            spots_state['transition_progress'] += delta_time / spots_state['transition_duration']
            spots_state['glow_phase'] += delta_time * 0.3  # Slow glow
            
            if spots_state['transition_progress'] >= 1.0:
                # Complete transition, pick new target
                spots_state['current_color'] = spots_state['target_color']
                spots_state['target_color'] = np.random.choice(list(instate['colors'].keys()))
                spots_state['transition_progress'] = 0.0
                spots_state['transition_duration'] = 6.0 + np.random.random() * 6.0  # 6-12 seconds
            
            # Interpolate between colors
            curr_h, curr_s, curr_v = instate['colors'][spots_state['current_color']]
            targ_h, targ_s, targ_v = instate['colors'][spots_state['target_color']]
            
            # Smooth interpolation
            t = spots_state['transition_progress']
            t_smooth = t * t * (3.0 - 2.0 * t)  # Smoothstep
            
            # Handle hue wrapping
            h_diff = targ_h - curr_h
            if abs(h_diff) > 0.5:
                if curr_h > targ_h:
                    targ_h += 1.0
                else:
                    curr_h += 1.0
            
            h = (curr_h * (1 - t_smooth) + targ_h * t_smooth) % 1.0
            s = curr_s * (1 - t_smooth) + targ_s * t_smooth
            v = curr_v * (1 - t_smooth) + targ_v * t_smooth
            
            # Apply glow effect
            glow = 0.7 + 0.3 * np.sin(spots_state['glow_phase'])
            v *= glow
            
            # Convert to RGB and fill buffer
            r, g, b = hsv_to_rgb(h, s, v)
            buffer[:] = [r, g, b, 0.6]
            
            continue  # Skip regular processing for spots strips
        
        # Regular strip processing
        if strip_id in instate['strip_colors']:
            strip_info = instate['strip_colors'][strip_id]
            base_color_key = strip_info['base_color']
            phase_offset = strip_info['phase_offset']
            speed_mult = strip_info['speed_multiplier']
            
            # Get base color
            h, s, v = instate['colors'][base_color_key]
            
            # Create position array for vectorized operations
            positions = np.arange(strip_length)
            
            # Apply different effects based on strip groups
            if 'base' in strip_groups:
                # Base strips: moving wave pattern
                wave_positions = positions / strip_length * 4 * np.pi  # 2 full waves across strip
                wave_time = instate['wave_offset'] * speed_mult + phase_offset
                wave_pattern = 0.5 + 0.5 * np.sin(wave_positions + wave_time)
                
                # Modulate brightness with wave
                v_values = v * (0.6 + 0.4 * wave_pattern)
                
            elif 'heart' in strip_groups:
                # Heart strips: pulsing effect
                pulse_time = instate['color_cycle_time'] * speed_mult * 2.0 + phase_offset
                pulse_intensity = 0.5 + 0.5 * np.sin(pulse_time)
                
                # Create distance from center for heart shape
                center = strip_length // 2
                distances = np.abs(positions - center) / (strip_length / 2)
                heart_shape = 1.0 - distances * 0.5  # Brighter at center
                
                v_values = v * heart_shape * (0.4 + 0.6 * pulse_intensity)
                
            elif 'spine' in strip_groups:
                # Spine strips: traveling spark
                spark_time = (instate['wave_offset'] * speed_mult + phase_offset) % (2 * np.pi)
                spark_position = (spark_time / (2 * np.pi)) * strip_length
                
                # Create spark effect
                spark_distances = np.abs(positions - spark_position)
                spark_effect = np.maximum(0, 1.0 - spark_distances / 10.0)  # 10 pixel wide spark
                
                v_values = v * (0.3 + 0.7 * spark_effect)
                
            else:
                # Other strips: gentle breathing effect
                breath_time = instate['color_cycle_time'] * speed_mult * 0.5 + phase_offset
                breath_intensity = 0.5 + 0.5 * np.sin(breath_time)
                
                v_values = np.full(strip_length, v * (0.5 + 0.5 * breath_intensity))
            
            # Add some color cycling over time
            color_shift = 0.1 * np.sin(instate['color_cycle_time'] * 0.2 + phase_offset)
            h_shifted = (h + color_shift) % 1.0
            
            # Convert HSV to RGB for each pixel (vectorized where possible)
            r_values = np.zeros(strip_length)
            g_values = np.zeros(strip_length)
            b_values = np.zeros(strip_length)
            
            # Note: hsv_to_rgb from colorsys doesn't support arrays, so we need to loop
            for i in range(strip_length):
                r, g, b = hsv_to_rgb(h_shifted, s, v_values[i])
                r_values[i] = r
                g_values[i] = g
                b_values[i] = b
            
            # Set alpha based on control level and position
            alpha_values = np.full(strip_length, 0.7 * final_alpha)
            
            # Combine into RGBA array and set buffer
            rgba_values = np.stack([r_values, g_values, b_values, alpha_values], axis=1)
            buffer[:] = rgba_values
        
        else:
            # Fallback for strips without initialized data
            buffer[:] = [0.2, 0.2, 0.4, 0.3]  # Dim blue
  