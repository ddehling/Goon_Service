from sceneutils.imgutils import *  # noqa: F403
import numpy as np
from pathlib import Path
from colorsys import hsv_to_rgb


def OTO_sunrise_joy(instate, outstate):
    """
    A joyful sunrise-themed pattern generator.
    
    Creates a rising sun effect in the middle of each strip in the "base" groups,
    transitioning from warm orange/yellow colors to light blue sky colors.
    The sun rises from the bottom to top strips over time, with cloud effects
    in the upper sky and rippling water reflections away from the sun.
    
    Optimized with vectorized operations where possible.
    """
    name = 'sunrise_joy'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['sun_position'] = 0.0  # Position of the sun (0.0 to 1.0)
        instate['cycle_duration'] = 100.0  # Time for a complete sunrise cycle (seconds)
        instate['base_strips'] = []  # Will store IDs of strips in "base" groups
        
        # Cloud effect parameters
        instate['cloud_positions'] = {}  # Store cloud positions for each strip
        instate['cloud_speeds'] = {}     # Store cloud speeds for each strip
        
        # Ocean wave parameters
        instate['wave_offsets'] = {}     # Random offsets for wave patterns
        instate['wave_speeds'] = {}      # Wave movement speeds
        
        # Identify strips in "base" groups using strip_manager
        for strip_id, strip in strip_manager.strips.items():
            if 'base' in strip.groups:
                instate['base_strips'].append(strip_id)
                
                # Initialize clouds for this strip
                strip_length = strip.length
                instate['cloud_positions'][strip_id] = []
                instate['cloud_speeds'][strip_id] = []
                
                # Create 3-5 clouds per strip
                num_clouds = np.random.randint(3, 6)
                for _ in range(num_clouds):
                    # Random position
                    pos = np.random.uniform(0, strip_length)
                    # Random width (5-15% of strip length)
                    width = np.random.uniform(0.05, 0.15) * strip_length
                    # Random density (0.2-0.6)
                    density = np.random.uniform(0.2, 0.6)
                    # Random speed (0.5-2 pixels per second)
                    speed = np.random.uniform(0.5, 2.0)
                    
                    instate['cloud_positions'][strip_id].append({
                        'pos': pos,
                        'width': width,
                        'density': density
                    })
                    instate['cloud_speeds'][strip_id].append(speed)
                
                # Initialize wave parameters for this strip
                instate['wave_offsets'][strip_id] = np.random.uniform(0, 2 * np.pi, 3)  # 3 wave components with random phases
                instate['wave_speeds'][strip_id] = np.random.uniform(0.2, 0.5, 3)  # Different speeds for each wave component
        
        # Sort base strips from bottom to top based on their groups
        def get_strip_order(strip_id):
            strip = strip_manager.get_strip(strip_id)
            if 'bottom' in strip.groups:
                return 0
            elif 'middle' in strip.groups:
                return 1
            elif 'top' in strip.groups:
                return 2
            return 3  # For any other base strips without specific vertical position
        
        instate['base_strips'].sort(key=get_strip_order)
        
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        return

    # Get alpha level from outstate or use default 1.0
    alpha = outstate.get('control_joyful', 0.0)/100
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = alpha
    
    # Skip rendering if alpha is too low
    if alpha < 0.01:
        return
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if remaining_time < 10.0:
        # Fade out over the last 10 seconds
        fade_alpha = remaining_time / 10.0
        fade_alpha = max(0.0, fade_alpha)
        buffers.generator_alphas[name] = fade_alpha * alpha
    
    # Time delta for animation
    delta_time = outstate['current_time'] - outstate['last_time']
    
    # Update sun position based on time (smoothly looping)
    time_position = (outstate['current_time'] % instate['cycle_duration']) / instate['cycle_duration']
    instate['sun_position'] = time_position
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Number of base strips for position calculation
    num_base_strips = len(instate['base_strips'])
    
    # Update cloud positions (vectorized where possible)
    for strip_id in instate['cloud_positions']:
        if strip_id in instate['cloud_speeds']:
            for i, cloud in enumerate(instate['cloud_positions'][strip_id]):
                # Move cloud position
                speed = instate['cloud_speeds'][strip_id][i]
                cloud['pos'] += speed * delta_time
                
                # Wrap around at strip end
                if strip_id in strip_manager.strips:
                    strip_length = strip_manager.get_strip(strip_id).length
                    if cloud['pos'] > strip_length + cloud['width']:
                        cloud['pos'] = -cloud['width']
    
    # Update wave offsets (vectorized)
    for strip_id in instate['wave_offsets']:
        if strip_id in instate['wave_speeds']:
            # Vectorized update of all wave offsets for this strip
            instate['wave_offsets'][strip_id] += instate['wave_speeds'][strip_id] * delta_time
            # Keep within 0-2π range (vectorized)
            instate['wave_offsets'][strip_id] %= (2 * np.pi)
    
    # For smoother transitions, use a sine wave function for sun size
    sun_cycle = 0.5 + 0.5 * np.sin(2 * np.pi * instate['sun_position'] - np.pi/2)
    
    # For color transitions, use a curve that ensures smooth transitions
    sun_height = 0.5 + 0.5 * np.sin(2 * np.pi * instate['sun_position'] - np.pi/2)
    
    # Render pattern to each buffer
    for strip_id, buffer in pattern_buffers.items():
        # Check if this is a base strip
        if strip_id in instate['base_strips']:
            strip_length = len(buffer)
            
            # Get the vertical position of this strip (0=bottom, 1=top)
            strip_index = instate['base_strips'].index(strip_id)
            strip_vertical_position = strip_index / max(1, num_base_strips - 1)
            
            # Calculate sun center position (middle of the strip)
            sun_center = strip_length // 2
            
            # Calculate sun radius based on the smooth cycle and strip position
            base_radius = strip_length * 0.3
            strip_height_factor = 1.0 - (strip_vertical_position * 0.5)  # Larger sun at bottom
            sun_radius = max(1, int(base_radius * sun_cycle * strip_height_factor))
            
            # Create position arrays for vectorized calculations
            positions = np.arange(strip_length)
            
            # Calculate distances from sun center (vectorized)
            distances = np.abs(positions - sun_center)
            
            # Determine which pixels are inside the sun (vectorized)
            inside_sun_mask = distances <= sun_radius
            
            # Calculate normalized distances (vectorized)
            normalized_distances = np.zeros(strip_length)
            # Inside sun
            if np.any(inside_sun_mask):
                normalized_distances[inside_sun_mask] = distances[inside_sun_mask] / sun_radius
            # Outside sun
            outside_sun_mask = ~inside_sun_mask
            if np.any(outside_sun_mask):
                normalized_distances[outside_sun_mask] = np.minimum(
                    1.0, (distances[outside_sun_mask] - sun_radius) / (strip_length * 0.5)
                )
            
            # Calculate base sky colors based on sun height
            if sun_height < 0.5:
                # Sunrise/sunset phase - orange to light blue
                transition = sun_height * 2.0
                height_influence = strip_vertical_position * 0.3
                
                # Sky colors (vectorized calculation)
                sky_r = 0.9 - (0.6 * (transition + height_influence))
                sky_g = 0.5 + (0.2 * (transition + height_influence))
                sky_b = 0.2 + (0.6 * (transition + height_influence))
            else:
                # Day phase - light blue to deeper blue
                transition = (sun_height - 0.5) * 2.0
                height_influence = strip_vertical_position * 0.3
                
                # Sky colors (vectorized calculation)
                sky_r = 0.3 - (0.1 * (transition + height_influence))
                sky_g = 0.7 - (0.1 * (transition + height_influence))
                sky_b = 0.8 + (0.1 * (transition + height_influence))
            
            # Initialize color arrays
            r_values = np.zeros(strip_length)
            g_values = np.zeros(strip_length)
            b_values = np.zeros(strip_length)
            a_values = np.zeros(strip_length)
            
            # Set colors for inside sun (vectorized)
            if np.any(inside_sun_mask):
                # Calculate intensity based on normalized distance
                intensities = 1.0 - normalized_distances[inside_sun_mask] ** 2
                
                # Set colors
                r_values[inside_sun_mask] = 1.0
                g_values[inside_sun_mask] = 0.7 + (0.3 * intensities)
                b_values[inside_sun_mask] = 0.2 + (0.3 * intensities)
                a_values[inside_sun_mask] = 0.8 + (0.2 * intensities)
            
            # Set colors for outside sun (vectorized)
            if np.any(outside_sun_mask):
                # Start with sky colors
                r_values[outside_sun_mask] = sky_r
                g_values[outside_sun_mask] = sky_g
                b_values[outside_sun_mask] = sky_b
                
                # Apply sun glow effect (vectorized)
                glow_mask = (distances > sun_radius) & (distances < sun_radius * 2)
                if np.any(glow_mask):
                    glow_factors = np.maximum(0, np.minimum(0.5, 1.0 - (distances[glow_mask] - sun_radius) / sun_radius))
                    
                    # Blend with sun colors
                    r_values[glow_mask] = r_values[glow_mask] * (1 - glow_factors) + 1.0 * glow_factors
                    g_values[glow_mask] = g_values[glow_mask] * (1 - glow_factors) + 0.7 * glow_factors
                    b_values[glow_mask] = b_values[glow_mask] * (1 - glow_factors) + 0.2 * glow_factors
                
                # Set alpha
                a_values[outside_sun_mask] = 0.7 - (0.2 * normalized_distances[outside_sun_mask])
                
                # Calculate edge distances for water ripple effect (vectorized)
                left_edge_distances = positions / (strip_length * 0.25)
                right_edge_distances = (strip_length - positions - 1) / (strip_length * 0.25)
                edge_distances = np.minimum(left_edge_distances, right_edge_distances)
                
                # Create mask for edge pixels (outer 25% on each side)
                edge_mask = (edge_distances < 1.0) & outside_sun_mask
                
                if np.any(edge_mask):
                    # Ripple intensity stronger at edges and on lower strips
                    ripple_intensities = (1.0 - edge_distances[edge_mask]) * (1.0 - strip_vertical_position * 0.7)
                    
                    # Get wave offsets for this strip
                    wave_offsets = instate['wave_offsets'].get(strip_id, np.zeros(3))
                    
                    # Calculate normalized positions for ripple effect
                    norm_positions = positions[edge_mask] / strip_length
                    
                    # Calculate ripple effect (vectorized)
                    ripples = np.zeros(np.sum(edge_mask))
                    ripples += 0.5 * np.sin(norm_positions * 20 + wave_offsets[0])
                    ripples += 0.3 * np.sin(norm_positions * 35 + wave_offsets[1])
                    ripples += 0.2 * np.sin(norm_positions * 50 + wave_offsets[2])
                    
                    # Normalize to -1 to 1 range
                    ripples /= (0.5 + 0.3 + 0.2)
                    
                    # Apply ripple effect to colors
                    ripple_amount = ripples * ripple_intensities * 0.2
                    
                    # Brighten colors on wave peaks, darken in troughs
                    r_values[edge_mask] += ripple_amount
                    g_values[edge_mask] += ripple_amount
                    b_values[edge_mask] += ripple_amount * 1.2  # Slightly stronger blue modulation
                    
                    # Add slight color shift toward blue-green in wave troughs
                    trough_mask = edge_mask.copy()
                    trough_mask[edge_mask] = ripples < 0
                    if np.any(trough_mask):
                        trough_ripple_intensities = ripple_intensities[ripples < 0]
                        trough_ripple_values = ripples[ripples < 0]
                        
                        g_values[trough_mask] += np.abs(trough_ripple_values) * trough_ripple_intensities * 0.05
                        b_values[trough_mask] += np.abs(trough_ripple_values) * trough_ripple_intensities * 0.1
            
            # Apply gentle variation for a more natural look in upper sky
            if strip_vertical_position > 0.5:
                # Create a mask for upper sky (outside sun)
                upper_sky_mask = outside_sun_mask
                if np.any(upper_sky_mask):
                    # Vectorized calculation of variations
                    variations = 0.05 * np.sin(positions[upper_sky_mask] * 0.2 + outstate['current_time'])
                    
                    # Apply variations
                    r_values[upper_sky_mask] += variations
                    g_values[upper_sky_mask] += variations
                    b_values[upper_sky_mask] += variations
            
            # Ensure color values are in valid range (vectorized)
            r_values = np.clip(r_values, 0.0, 1.0)
            g_values = np.clip(g_values, 0.0, 1.0)
            b_values = np.clip(b_values, 0.0, 1.0)
            a_values = np.clip(a_values, 0.0, 1.0)
            
            # Combine into RGBA array
            rgba_values = np.stack([r_values, g_values, b_values, a_values], axis=1)
            
            # Set buffer values (vectorized)
            buffer[:] = rgba_values
            
            # Apply cloud effects (more visible on upper strips)
            if strip_vertical_position > 0.3 and strip_id in instate['cloud_positions']:
                cloud_influence = strip_vertical_position * 0.7  # More clouds visible higher up
                
                for cloud in instate['cloud_positions'][strip_id]:
                    # Calculate cloud span
                    cloud_center = int(cloud['pos'])
                    cloud_width = int(cloud['width'])
                    cloud_density = cloud['density'] * cloud_influence
                    
                    # Calculate cloud range
                    cloud_start = max(0, cloud_center - cloud_width)
                    cloud_end = min(strip_length, cloud_center + cloud_width + 1)
                    
                    if cloud_end > cloud_start:
                        # Get positions within cloud range
                        cloud_positions = np.arange(cloud_start, cloud_end)
                        
                        # Calculate distance from center (vectorized)
                        dist_from_center = np.abs(cloud_positions - cloud_center) / cloud_width
                        
                        # Cloud intensity falls off from center (vectorized)
                        cloud_intensities = cloud_density * (1.0 - dist_from_center**2)
                        
                        # Apply cloud effect only where intensity is significant
                        significant_mask = cloud_intensities > 0.01
                        if np.any(significant_mask):
                            # Filter positions and intensities
                            affect_positions = cloud_positions[significant_mask]
                            affect_intensities = cloud_intensities[significant_mask]
                            
                            # Get current pixel colors
                            curr_colors = buffer[affect_positions].copy()
                            
                            # Cloud colors
                            cloud_colors = np.array([0.9, 0.9, 0.95, 1.0])
                            
                            # Calculate alpha adjustment
                            alpha_adjust = 1.0 - 0.2 * affect_intensities
                            
                            # Reshape intensities for broadcasting
                            affect_intensities = affect_intensities[:, np.newaxis]
                            alpha_adjust = alpha_adjust[:, np.newaxis]
                            
                            # Calculate new colors (vectorized blend)
                            new_colors = curr_colors * (1 - affect_intensities) + cloud_colors[np.newaxis, :] * affect_intensities
                            
                            # Apply alpha adjustment
                            new_colors[:, 3] = curr_colors[:, 3] * alpha_adjust[:, 0]
                            
                            # Update buffer
                            buffer[affect_positions] = new_colors
        else:
            # For non-base strips, apply ambient light that matches the sunrise colors
            strip_length = len(buffer)
            
            # Get strip information if available
            strip_groups = []
            if strip_id in strip_manager.strips:
                strip = strip_manager.get_strip(strip_id)
                strip_groups = strip.groups
            
            # Determine strip type for customized effects
            is_heart = 'heart' in strip_groups
            is_spine = 'spine' in strip_groups
            is_ear = 'ear' in strip_groups
            is_head = 'head' in strip_groups
            
            # Base colors for the current sun position
            if sun_height < 0.5:
                # Sunrise/sunset colors
                transition = sun_height * 2.0
                
                # Base ambient colors
                r_base = 0.6 - (0.2 * transition)
                g_base = 0.3 + (0.3 * transition)
                b_base = 0.1 + (0.5 * transition)
                
                # Customize for different strip types
                if is_heart:
                    # Hearts get a warm glow during sunrise
                    r_base += 0.3
                    g_base -= 0.1
                elif is_spine:
                    # Spines get slight golden tint
                    r_base += 0.1
                    g_base += 0.1
                elif is_ear:
                    # Ears get slight amber tint
                    r_base += 0.2
                    g_base += 0.1
                elif is_head:
                    # Head gets a bright glow
                    r_base += 0.2
                    g_base += 0.2
                    b_base += 0.1
            else:
                # Day phase colors
                transition = (sun_height - 0.5) * 2.0
                
                # Base ambient colors
                r_base = 0.4 - (0.2 * transition)
                g_base = 0.6 + (0.1 * transition)
                b_base = 0.6 + (0.3 * transition)
                
                # Customize for different strip types
                if is_heart:
                    # Hearts get a soft glow during day
                    r_base += 0.1
                    g_base += 0.1
                    b_base += 0.2
                elif is_spine:
                    # Spines get slight green tint
                    g_base += 0.2
                elif is_ear:
                    # Ears get slight purple tint
                    r_base += 0.1
                    b_base += 0.2
                elif is_head:
                    # Head gets a bright sky blue
                    g_base += 0.1
                    b_base += 0.3
            
            # Create position array for vectorized calculations
            positions = np.arange(strip_length, dtype=float)
            normalized_positions = positions / strip_length
            
            # Apply different patterns based on strip type (vectorized)
            if is_heart:
                # Hearts get a gentle ambient glow without rhythm
                # Vectorized spatial variation calculation
                variations = 0.1 * np.cos(normalized_positions * 0.2 * strip_length)
                
                # Apply the color with subtle variation (vectorized)
                r_values = r_base + variations
                g_values = g_base + variations
                b_values = b_base + variations
                a_values = 0.7 + variations * 0.3
                
            elif is_spine:
                # Spines get a flowing wave pattern
                # Vectorized wave calculation
                wave_pos = normalized_positions - (sun_height * 2 % 1.0)
                waves = 0.5 + 0.5 * np.cos(wave_pos * 6 * np.pi)
                
                # Apply the color with wave intensity (vectorized)
                r_values = r_base * (0.7 + 0.3 * waves)
                g_values = g_base * (0.7 + 0.3 * waves)
                b_values = b_base * (0.7 + 0.3 * waves)
                a_values = 0.6 + 0.4 * waves
                
            elif is_ear or is_head:
                # Ears and head get a circular wave pattern
                # Vectorized wave calculation
                waves = 0.5 + 0.5 * np.sin(normalized_positions * 6 * np.pi + outstate['current_time'] * 2)
                
                # Apply the color with wave intensity (vectorized)
                r_values = r_base * (0.8 + 0.2 * waves)
                g_values = g_base * (0.8 + 0.2 * waves)
                b_values = b_base * (0.8 + 0.2 * waves)
                a_values = 0.7 + 0.3 * waves
                
            else:
                # Default pattern for other strips - gentle variation
                # Vectorized variation calculation
                variations = 0.1 * np.sin(normalized_positions * 0.1 * strip_length + outstate['current_time'] * 0.2)
                
                # Calculate final colors (vectorized)
                r_values = r_base + variations
                g_values = g_base + variations
                b_values = b_base + variations
                a_values = 0.6 + variations
            
            # Ensure color values are in valid range (vectorized)
            r_values = np.clip(r_values, 0.0, 1.0)
            g_values = np.clip(g_values, 0.0, 1.0)
            b_values = np.clip(b_values, 0.0, 1.0)
            a_values = np.clip(a_values, 0.0, 1.0)
            
            # Combine into RGBA array
            rgba_values = np.stack([r_values, g_values, b_values, a_values], axis=1)
            
            # Set buffer values (vectorized)
            buffer[:] = rgba_values

def OTO_sad_theme(instate, outstate):
    """
    Generator function that creates a sad-themed pattern across all strips.
    
    Features:
    1. Global alpha controlled by outstate['control_sad'] value
    2. Dark blues and grey-blues are the predominant colors 
    3. Blue raindrops on the spines with long trails that roll down and turn grey
       then wrap around to the top when they reach the bottom
    4. Similar raindrops on base strips moving in both directions with consistent respawning
    5. Asynchronous grey and blue heartbeat on heart strips with brighter, longer pulses
    6. Grey and white spots that blow away like dandelion seeds on brain and ear strips
    
    Optimized with vectorized operations for better performance.
    """
    name = 'sad_theme'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['raindrops'] = {}  # For tracking raindrops on spine strips
        instate['heart_phase_left'] = 0.0  # For left heartbeat timing
        instate['heart_phase_right'] = 0.5  # For right heartbeat timing (start offset)
        instate['dandelion_seeds'] = {}  # For tracking floating seeds
        instate['last_drop_time'] = {}  # Time tracker for raindrop creation (per strip)
        instate['spots_state'] = {}  # Track slow transitions for spots strips
        
        # Color palette (HSV values)
        instate['colors'] = {
            'deep_blue': [0.6, 0.8, 0.6],  # Deep blue
            'grey_blue': [0.6, 0.3, 0.7],  # Grey-blue
            'light_grey': [0.0, 0.0, 0.8],  # Light grey
            'dark_grey': [0.0, 0.0, 0.5],   # Dark grey
            'white': [0.0, 0.0, 0.9]        # Almost white
        }
        
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        return

    # Get sad level from outstate (default to 0)
    sad_level = outstate.get('control_sad', 0.0)/100
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = sad_level
    
    # Skip rendering if alpha is too low
    if sad_level < 0.01:
        return
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if remaining_time < 10.0:
        fade_alpha = remaining_time / 10.0
        fade_alpha = max(0.0, fade_alpha)
        buffers.generator_alphas[name] = fade_alpha * sad_level
    
    # Get delta time for animation calculations
    delta_time = outstate['current_time'] - outstate['last_time']
    
    # Update heartbeat phases - slow heartbeat around 40 BPM with slight variation between sides
    beats_per_second_left = 38 / 60.0  # Slightly slower for left
    beats_per_second_right = 42 / 60.0  # Slightly faster for right
    
    instate['heart_phase_left'] = (instate['heart_phase_left'] + beats_per_second_left * delta_time) % 1.0
    instate['heart_phase_right'] = (instate['heart_phase_right'] + beats_per_second_right * delta_time) % 1.0
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Process each buffer based on strip type
    for strip_id, buffer in pattern_buffers.items():
        # Skip if strip doesn't exist in manager
        if strip_id not in strip_manager.strips:
            continue
            
        strip = strip_manager.get_strip(strip_id)
        strip_length = len(buffer)
        
        # Start with a dim base color - slight grey-blue tint
        base_hue, base_sat, base_val = instate['colors']['grey_blue']
        base_r, base_g, base_b = hsv_to_rgb(base_hue, base_sat * 0.5, base_val * 0.3)
        buffer[:] = [base_r, base_g, base_b, 0.2]  # Very dim base

        # Special handling for spots strips - slow, gentle transitions
        if 'spots' in strip.groups:
            # Initialize spots state if not already done
            if strip_id not in instate['spots_state']:
                instate['spots_state'][strip_id] = {
                    'current_color': 'grey_blue',
                    'target_color': 'deep_blue',
                    'transition_progress': 0.0,
                    'hold_time': 0.0,
                    'pulse_phase': np.random.random() * 2 * np.pi
                }
            
            spots = instate['spots_state'][strip_id]
            
            # Update transition
            if spots['transition_progress'] < 1.0:
                # Slow transition - 3 seconds to change color
                spots['transition_progress'] += delta_time / 3.0
                spots['transition_progress'] = min(1.0, spots['transition_progress'])
            else:
                # Hold current color for a bit
                spots['hold_time'] += delta_time
                if spots['hold_time'] > 2.0:  # Hold for 2 seconds
                    # Choose new target color
                    spots['hold_time'] = 0.0
                    spots['transition_progress'] = 0.0
                    spots['current_color'] = spots['target_color']
                    # Pick a new target from sad palette
                    color_options = ['deep_blue', 'grey_blue', 'dark_grey']
                    spots['target_color'] = np.random.choice([c for c in color_options if c != spots['current_color']])
            
            # Interpolate between colors
            curr_h, curr_s, curr_v = instate['colors'][spots['current_color']]
            targ_h, targ_s, targ_v = instate['colors'][spots['target_color']]
            
            # Smooth interpolation
            t = spots['transition_progress']
            # Use smoothstep for even smoother transitions
            t = t * t * (3.0 - 2.0 * t)
            
            h = curr_h + (targ_h - curr_h) * t
            s = curr_s + (targ_s - curr_s) * t
            v = curr_v + (targ_v - curr_v) * t
            
            # Add very subtle pulse
            spots['pulse_phase'] += delta_time * 0.5  # Very slow pulse
            pulse = 0.9 + 0.1 * np.sin(spots['pulse_phase'])
            v *= pulse
            
            # Convert to RGB
            r, g, b = hsv_to_rgb(h, s, v)
            
            # Apply to entire strip uniformly
            buffer[:] = [r, g, b, 0.6]
            
            continue  # Skip other effects for spots strips    

        # Different effects based on strip type
        if 'spine' in strip.groups or 'base' in strip.groups:
            # Spine and base strips - raindrops effect
            
            # Initialize raindrops for this strip if not already done
            if strip_id not in instate['raindrops']:
                instate['raindrops'][strip_id] = []
                instate['last_drop_time'][strip_id] = 0.0
                
                # Pre-populate with some initial raindrops
                num_initial_drops = strip_length // 5  # About 1 drop every 5 pixels
                for _ in range(num_initial_drops):
                    # Random position along strip
                    pos = np.random.randint(0, strip_length)
                    
                    # For base strips, determine direction based on position
                    direction = 1  # Default downward for spine
                    if 'base' in strip.groups:
                        # For base: if in left half, move right; if in right half, move left
                        middle = strip_length // 2
                        direction = 1 if pos < middle else -1
                    
                    # Initial color phase based on position
                    color_phase = pos / strip_length if 'spine' in strip.groups else np.random.random()
                    
                    instate['raindrops'][strip_id].append({
                        'position': pos,
                        'speed': 8 + np.random.random() * 12,  # 8-20 pixels per second
                        'size': 8 + np.random.randint(0, 8),  # 8-15 pixels for longer trails
                        'color_phase': color_phase,  # 0.0 = blue, 1.0 = grey
                        'alpha': 0.7 + np.random.random() * 0.3,  # 0.7-1.0 alpha
                        'direction': direction,  # 1 for down, -1 for up (base strips)
                        'life_cycles': 0  # Track how many times the drop has cycled
                    })
            
            # Get number of active drops for this strip
            active_drops = len(instate['raindrops'][strip_id])
            
            # Determine target number of drops based on strip type and length
            if 'spine' in strip.groups:
                target_drops = max(5, strip_length // 8)  # More drops on spine
            else:  # base strips
                target_drops = max(6, strip_length // 10)  # Slightly fewer on base, but ensure at least 6
            
            # Get current time
            current_time = outstate['current_time']
            
            # Check last drop time for this specific strip
            if strip_id not in instate['last_drop_time']:
                instate['last_drop_time'][strip_id] = current_time - 1.0  # Immediate creation
            
            # Calculate time since last drop for this strip
            time_since_last_drop = current_time - instate['last_drop_time'][strip_id]
            
            # Determine if we should create a new drop
            # More aggressive spawning for base strips
            if 'base' in strip.groups:
                # For base strips: create drops more aggressively if below target
                min_time_between_drops = 0.1 if active_drops < target_drops else 0.5
                should_create_drop = time_since_last_drop > min_time_between_drops
                # Force creation if severely below target
                if active_drops < target_drops * 0.5:
                    should_create_drop = time_since_last_drop > 0.05
            else:
                # For spine strips: standard creation logic
                should_create_drop = (active_drops < target_drops and time_since_last_drop > 0.1) or time_since_last_drop > 0.5
            
            # Create new drops as needed
            if should_create_drop:
                instate['last_drop_time'][strip_id] = current_time
                
                # Higher creation chance if below target
                creation_chance = 0.9 if active_drops < target_drops * 0.8 else 0.4
                
                # For base strips, even higher chance if very low on drops
                if 'base' in strip.groups and active_drops < target_drops * 0.5:
                    creation_chance = 0.95
                
                if np.random.random() < creation_chance:
                    # For spine: always start at top
                    if 'spine' in strip.groups:
                        pos = 0
                        direction = 1  # Down
                        color_phase = 0.0  # Start blue
                    # For base: start at either end with equal probability
                    elif 'base' in strip.groups:
                        start_left = np.random.random() < 0.5  # 50% chance to start from left
                        pos = 0 if start_left else strip_length - 1
                        direction = 1 if start_left else -1  # Direction based on starting position
                        color_phase = 0.0  # Start blue
                    
                    instate['raindrops'][strip_id].append({
                        'position': pos,
                        'speed': 8 + np.random.random() * 12,  # 8-20 pixels per second
                        'size': 18 + np.random.randint(0, 8),  # 8-15 pixels for longer trails
                        'color_phase': color_phase,
                        'alpha': 0.7 + np.random.random() * 0.3,  # 0.7-1.0 alpha
                        'direction': direction,
                        'life_cycles': 0  # New drop starts with 0 cycles
                    })
            
            # Update existing raindrops
            new_raindrops = []
            
            # Process drops in batches for vectorization
            for drop in instate['raindrops'][strip_id]:
                # Update position based on direction
                drop['position'] += drop['speed'] * drop['direction'] * delta_time
                
                # Handle wrapping when drops reach the ends of the strip
                if 'spine' in strip.groups:
                    # For spine strips:
                    if drop['position'] >= strip_length:
                        # When drop reaches bottom, wrap to top with new properties
                        drop['position'] = 0
                        drop['color_phase'] = 0.0  # Reset to blue
                        drop['speed'] = 8 + np.random.random() * 12  # New random speed
                        drop['alpha'] = 0.7 + np.random.random() * 0.3  # New random alpha
                        drop['life_cycles'] += 1  # Increment cycle count
                elif 'base' in strip.groups:
                    # For base strips: handle boundary conditions
                    if drop['position'] >= strip_length:
                        # Reached right end
                        drop['position'] = 0  # Wrap to left end
                        drop['direction'] = 1  # Set direction to right
                        drop['color_phase'] = 0.0  # Reset to blue
                        drop['alpha'] = 0.9 + np.random.random() * 0.1  # New random alpha
                        drop['life_cycles'] = 0  # Reset cycle count for consistent respawning
                    elif drop['position'] < 0:
                        # Reached left end
                        drop['position'] = strip_length - 1  # Wrap to right end
                        drop['direction'] = -1  # Set direction to left
                        drop['color_phase'] = 0.0  # Reset to blue
                        drop['alpha'] = 0.9 + np.random.random() * 0.1  # New random alpha
                        drop['life_cycles'] = 0  # Reset cycle count for consistent respawning
                    
                    # Check if drop has reached the middle
                    middle = strip_length // 2
                    middle_range = strip_length * 0.1  # 10% of strip length around middle
                    
                    # If moving right (direction=1) and passed middle
                    if (drop['direction'] > 0 and 
                        middle - middle_range/2 < drop['position'] < middle + middle_range/2):
                        # Small chance to reverse direction at middle
                        if np.random.random() < 0.1:  # 10% chance per frame when in middle zone
                            drop['direction'] = -1
                            drop['alpha'] *= 0.9  # Slightly reduce alpha
                    
                    # If moving left (direction=-1) and passed middle
                    elif (drop['direction'] < 0 and 
                          middle - middle_range/2 < drop['position'] < middle + middle_range/2):
                        # Small chance to reverse direction at middle
                        if np.random.random() < 0.1:  # 10% chance per frame when in middle zone
                            drop['direction'] = 1
                            drop['alpha'] *= 0.9  # Slightly reduce alpha
                
                # Update color phase (transition from blue to grey as it moves)
                if 'spine' in strip.groups:
                    # For spine: change color based on distance from top
                    drop['color_phase'] = min(1.0, drop['position'] / strip_length)
                elif 'base' in strip.groups:
                    # For base: change color based on distance from middle
                    middle = strip_length // 2
                    distance_from_middle = abs(drop['position'] - middle) / (strip_length / 2)
                    drop['color_phase'] = distance_from_middle
                
                # Gradually reduce alpha over multiple life cycles (slower for spine, faster for base)
                if 'spine' in strip.groups:
                    max_cycles = 3  # Spine drops can cycle more times
                    cycle_alpha_reduction = 0.1  # Smaller reduction per cycle
                else:  # base strips
                    max_cycles = 3  # Base drops can now cycle more times too
                    cycle_alpha_reduction = 0.05  # Much smaller reduction for base (more persistent)
                
                # Apply alpha reduction based on cycle count
                if drop['life_cycles'] > 0:
                    drop['alpha'] *= (1.0 - (drop['life_cycles'] * cycle_alpha_reduction))
                
                # For base strips: ensure drops don't fade too much
                if 'base' in strip.groups:
                    drop['alpha'] = max(0.3, drop['alpha'])  # Maintain minimum visibility
                
                # Add drop to new list if it's still active and hasn't cycled too many times
                if drop['alpha'] > 0.1 and drop['life_cycles'] < max_cycles:
                    new_raindrops.append(drop)
                    
                    # Draw the raindrop and trail
                    pos_int = int(drop['position'])
                    
                    # Interpolate between deep blue and grey
                    blue_h, blue_s, blue_v = instate['colors']['deep_blue']
                    grey_h, grey_s, grey_v = instate['colors']['dark_grey']
                    
                    h = blue_h + (grey_h - blue_h) * drop['color_phase']
                    s = blue_s + (grey_s - blue_s) * drop['color_phase']
                    v = blue_v + (grey_v - blue_v) * drop['color_phase']
                    
                    r, g, b = hsv_to_rgb(h, s, v)
                    
                    # Draw main drop
                    if 0 <= pos_int < strip_length:
                        buffer[pos_int] = [r, g, b, drop['alpha']]
                    
                    # Prepare arrays for vectorized trail drawing
                    trail_positions = []
                    trail_colors = []
                    
                    # Calculate trail positions and colors
                    for i in range(1, drop['size']):
                        # Trail position depends on direction
                        trail_pos = pos_int - (i * drop['direction'])
                        
                        # Handle trail wrapping
                        if trail_pos >= strip_length:
                            trail_pos = trail_pos - strip_length
                        elif trail_pos < 0:
                            trail_pos = strip_length + trail_pos
                        
                        if 0 <= trail_pos < strip_length:
                            # Calculate trail alpha with gradual fade
                            trail_alpha = drop['alpha'] * (1 - (i / drop['size'])**1.5)
                            
                            # Calculate trail color (more grey as it trails)
                            trail_phase = min(1.0, drop['color_phase'] + (i / drop['size']) * 0.3)
                            trail_h = blue_h + (grey_h - blue_h) * trail_phase
                            trail_s = blue_s + (grey_s - blue_s) * trail_phase
                            trail_v = blue_v + (grey_v - blue_v) * trail_phase
                            
                            tr, tg, tb = hsv_to_rgb(trail_h, trail_s, trail_v)
                            
                            # Add to trail arrays
                            trail_positions.append(trail_pos)
                            trail_colors.append([tr, tg, tb, trail_alpha])
                    
                    # Apply all trail pixels at once if any exist
                    if trail_positions:
                        # Convert to numpy arrays for vectorized assignment
                        trail_positions = np.array(trail_positions, dtype=int)
                        trail_colors = np.array(trail_colors)
                        
                        # Set all trail pixels at once
                        for idx, pos in enumerate(trail_positions):
                            buffer[pos] = trail_colors[idx]
            
            # Replace with updated list
            instate['raindrops'][strip_id] = new_raindrops
            
        elif 'heart' in strip.groups:
            # Heart strips - asynchronous heartbeat with grey-blue color
            
            # Determine which phase to use based on left/right
            is_left = 'left' in strip.groups
            current_phase = instate['heart_phase_left'] if is_left else instate['heart_phase_right']
            
            # Create heartbeat waveform with longer linger
            if current_phase < 0.1:
                # Sharp rise (systole)
                intensity = current_phase / 0.1
            elif current_phase < 0.3:
                # Peak plateau (longer linger at peak)
                intensity = 1.0
            elif current_phase < 0.6:
                # First decline (early diastole) - more gradual
                intensity = 1.0 - 0.7 * ((current_phase - 0.3) / 0.3)
            else:
                # Second, more gradual decline (late diastole)
                intensity = 0.3 - 0.3 * ((current_phase - 0.6) / 0.4)
            
            # Stronger heartbeat but still sad
            intensity *= 0.9
            
            # Use different color based on left/right heart
            if is_left:
                # Left heart - more grey
                h, s, v = instate['colors']['dark_grey']
                # Make peak brighter
                v_peak = min(1.0, v * 2.5)
            else:
                # Right heart - more blue
                h, s, v = instate['colors']['grey_blue']
                # Make peak brighter
                v_peak = min(1.0, v * 2.0)
            
            # Adjust value based on heartbeat intensity - more dramatic range
            v_adjusted = v * 0.3 + v_peak * 0.7 * intensity
            
            # Convert to RGB
            r, g, b = hsv_to_rgb(h, s, v_adjusted)
            
            # Set uniform color for the heart strip with brighter alpha at peak - vectorized
            alpha = 0.3 + 0.7 * intensity
            buffer[:] = [r, g, b, alpha]
            
        elif 'brain' in strip.groups or 'ear' in strip.groups:
            # Brain and ear strips - dandelion seeds effect
            
            # Initialize seed particles for this strip if not already done
            if strip_id not in instate['dandelion_seeds']:
                instate['dandelion_seeds'][strip_id] = []
                
                # Create initial seeds (more for longer strips)
                num_seeds = max(3, strip_length // 10)
                for _ in range(num_seeds):
                    # Random starting position
                    pos = np.random.randint(0, strip_length)
                    
                    # Random seed properties
                    instate['dandelion_seeds'][strip_id].append({
                        'position': pos,
                        'speed': 3 + np.random.random() * 7,  # 3-10 pixels per second
                        'direction': 1 if np.random.random() < 0.5 else -1,  # Random direction
                        'color': 'white' if np.random.random() < 0.3 else 'light_grey',  # Mostly grey with some white
                        'alpha': 0.5 + np.random.random() * 0.5,  # 0.5-1.0 alpha
                        'fade_rate': 0.05 + np.random.random() * 0.15  # How quickly it fades
                    })
            
            # Chance to add new seeds
            if np.random.random() < 0.05:  # 5% chance per frame
                pos = np.random.randint(0, strip_length)
                instate['dandelion_seeds'][strip_id].append({
                    'position': pos,
                    'speed': 3 + np.random.random() * 7,
                    'direction': 1 if np.random.random() < 0.5 else -1,
                    'color': 'white' if np.random.random() < 0.3 else 'light_grey',
                    'alpha': 0.5 + np.random.random() * 0.5,
                    'fade_rate': 0.05 + np.random.random() * 0.15
                })
            
            # Update existing seeds
            new_seeds = []
            for seed in instate['dandelion_seeds'][strip_id]:
                # Update position
                seed['position'] += seed['speed'] * seed['direction'] * delta_time
                
                # Handle wrapping for positions
                if seed['position'] >= strip_length:
                    seed['position'] = 0
                elif seed['position'] < 0:
                    seed['position'] = strip_length - 1
                
                # Gradually fade
                seed['alpha'] -= seed['fade_rate'] * delta_time
                
                # Keep if still visible
                if seed['alpha'] > 0.05:
                    new_seeds.append(seed)
                    
                    # Draw the seed
                    pos_int = int(seed['position'])
                    
                    # Get color
                    h, s, v = instate['colors'][seed['color']]
                    r, g, b = hsv_to_rgb(h, s, v)
                    
                    # Set pixel
                    buffer[pos_int] = [r, g, b, seed['alpha']]
                    
                    # Prepare arrays for vectorized glow drawing
                    glow_positions = []
                    glow_colors = []
                    
                    # Add small glow
                    for i in range(1, 3):
                        glow_pos1 = (pos_int + i) % strip_length
                        glow_pos2 = (pos_int - i) % strip_length
                        glow_alpha = seed['alpha'] * (1 - (i / 3))
                        
                        glow_positions.extend([glow_pos1, glow_pos2])
                        glow_colors.extend([[r, g, b, glow_alpha], [r, g, b, glow_alpha]])
                    
                    # Apply all glow pixels at once
                    if glow_positions:
                        for idx, pos in enumerate(glow_positions):
                            buffer[pos] = glow_colors[idx]
            
            # Replace with updated list
            instate['dandelion_seeds'][strip_id] = new_seeds
            
        else:
            # Other strips - subtle blue-grey pulsing - fully vectorized
            
            # Calculate pulsing effect
            pulse = 0.3 + 0.2 * np.sin(outstate['current_time'] * 0.5)
            
            # Get color - grey-blue
            h, s, v = instate['colors']['grey_blue']
            
            # Adjust value based on pulse
            v = v * pulse
            
            # Convert to RGB
            r, g, b = hsv_to_rgb(h, s, v)
            
            # Set uniform color for the strip (vectorized)
            buffer[:] = [r, g, b, 0.3]

def OTO_angry_theme(instate, outstate):
    """
    Generator function that creates an angry-themed pattern across all strips.
    
    Features:
    1. Global alpha controlled by outstate['control_angry'] value
    2. Red, orange, and brown color palette to evoke fire and rage
    3. Simplified flame effects and explosions
    4. Spots strips use same colors but with slow transitions
    5. Optimized with vectorized operations where beneficial
    """
    import numpy as np
    from colorsys import hsv_to_rgb  # Use standard Python library for simplicity
    
    name = 'angry_theme'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['explosions'] = {}  # Track explosion effects
        instate['flames'] = {}      # Track flame particles
        instate['heart_phase'] = 0.0  # For heartbeat timing
        instate['spots_state'] = {}  # Track slow transitions for spots strips
        
        # Color palette (HSV values)
        instate['colors'] = {
            'bright_red': [0.98, 0.95, 0.95],    # Intense red
            'orange': [0.05, 0.95, 0.9],         # Bright orange
            'yellow': [0.12, 0.85, 0.9],         # Fire yellow
            'deep_red': [0.98, 0.9, 0.7],        # Deep red
        }
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        return

    # Get angry level from outstate (default to 0)
    angry_level = outstate.get('control_angry', 0.0)/100
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = angry_level
    
    # Skip rendering if alpha is too low
    if angry_level < 0.01:
        return
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if remaining_time < 10.0:
        fade_alpha = remaining_time / 10.0
        buffers.generator_alphas[name] = max(0.0, fade_alpha) * angry_level
    
    # Get delta time for animation calculations
    delta_time = outstate['current_time'] - outstate['last_time']
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Update heartbeat phase - fast heartbeat around 100-120 BPM for anger
    beats_per_second = 2.0  # 120 BPM
    instate['heart_phase'] = (instate['heart_phase'] + beats_per_second * delta_time) % 1.0
    
    # Process each buffer based on strip type
    for strip_id, buffer in pattern_buffers.items():
        # Skip if strip doesn't exist in manager
        if strip_id not in strip_manager.strips:
            continue
            
        strip = strip_manager.get_strip(strip_id)
        strip_length = len(buffer)
        
        # Start with a base color - dim red glow
        h, s, v = instate['colors']['deep_red']
        base_r, base_g, base_b = hsv_to_rgb(h, s * 0.3, v * 0.2)
        buffer[:] = [base_r, base_g, base_b, 0.2]  # Dim base

        if 'spots' in strip.groups:
            # Initialize spots state if not already done
            if strip_id not in instate['spots_state']:
                instate['spots_state'][strip_id] = {
                    'current_color': 'deep_red',
                    'target_color': 'orange',
                    'transition_progress': 0.0,
                    'glow_phase': np.random.random() * 2 * np.pi,
                    'intensity': 0.7
                }
            
            spots = instate['spots_state'][strip_id]
            
            # Update transition - slower for spots
            if spots['transition_progress'] < 1.0:
                spots['transition_progress'] += delta_time / 2.5  # 2.5 seconds transition
                spots['transition_progress'] = min(1.0, spots['transition_progress'])
            else:
                # Start new transition
                spots['transition_progress'] = 0.0
                spots['current_color'] = spots['target_color']
                # Cycle through angry colors
                color_cycle = ['deep_red', 'bright_red', 'orange', 'yellow']
                current_idx = color_cycle.index(spots['current_color'])
                spots['target_color'] = color_cycle[(current_idx + 1) % len(color_cycle)]
            
            # Interpolate between colors
            curr_h, curr_s, curr_v = instate['colors'][spots['current_color']]
            targ_h, targ_s, targ_v = instate['colors'][spots['target_color']]
            
            t = spots['transition_progress']
            # Smooth interpolation
            t = t * t * (3.0 - 2.0 * t)
            
            h = curr_h + (targ_h - curr_h) * t
            s = curr_s + (targ_s - curr_s) * t
            v = curr_v + (targ_v - curr_v) * t
            
            # Add slow "breathing" glow effect
            spots['glow_phase'] += delta_time * 1.5  # Slow glow
            glow = 0.7 + 0.3 * np.sin(spots['glow_phase'])
            v *= glow
            
            # Convert to RGB
            r, g, b = hsv_to_rgb(h, s, v)
            
            # Apply to entire strip
            buffer[:] = [r, g, b, 0.7]

        # Apply strip-specific effects based on group
        if 'heart' in strip.groups:
            # Heart strips - rapid angry heartbeat
            apply_heart_effect(buffer, strip_length, instate, outstate)
            
        elif 'brain' in strip.groups or 'ear' in strip.groups or 'head' in strip.groups:
            # Brain, ear, and head strips - explosive bursts
            apply_explosive_effect(buffer, strip_length, strip_id, instate, outstate, delta_time)
            
        elif 'spine' in strip.groups:
            # Spine strips - upward moving flames
            apply_flame_effect(buffer, strip_length, strip_id, instate, outstate, delta_time, direction=-1)
            
        elif 'base' in strip.groups:
            # Base strips - rolling flames
            apply_flame_effect(buffer, strip_length, strip_id, instate, outstate, delta_time, direction=0)
            
        else:
            # Other strips - fiery pulsing
            apply_pulse_effect(buffer, strip_length, instate, outstate)


def apply_heart_effect(buffer, strip_length, instate, outstate):
    """Apply heartbeat effect to heart strips"""
    import numpy as np
    from colorsys import hsv_to_rgb
    
    # Calculate heartbeat intensity
    phase = instate['heart_phase']
    if phase < 0.1:
        # Very sharp rise (systole)
        intensity = phase / 0.1
    elif phase < 0.2:
        # Short peak
        intensity = 1.0
    elif phase < 0.3:
        # Sharp decline
        intensity = 1.0 - ((phase - 0.2) / 0.1)
    elif phase < 0.4:
        # Brief after-beat bump
        intensity = 0.3 * ((phase - 0.3) / 0.1)
    elif phase < 0.5:
        # Small secondary peak
        intensity = 0.3
    else:
        # Rest phase
        intensity = 0.3 * (1.0 - ((phase - 0.5) / 0.5))
    
    # Make intensity stronger for anger
    intensity = intensity * 0.7 + 0.3
    
    # Use bright red for the heart
    h, s, v = instate['colors']['bright_red']
    v_adjusted = v * intensity
    r, g, b = hsv_to_rgb(h, s, v_adjusted)
    
    # Create distance array from center
    center = strip_length // 2
    for i in range(strip_length):
        dist = abs(i - center) / (strip_length / 2)
        pulse_intensity = intensity * (1.0 - dist * 0.5)
        buffer[i] = [
            r,
            g * (0.7 + 0.3 * pulse_intensity),
            b * (0.7 + 0.3 * pulse_intensity),
            (0.4 + 0.6 * intensity) * (0.8 + 0.2 * pulse_intensity)
        ]


def apply_explosive_effect(buffer, strip_length, strip_id, instate, outstate, delta_time):
    """Apply explosive burst effects to brain/ear/head strips"""
    import numpy as np
    import random
    from colorsys import hsv_to_rgb
    
    # Initialize explosions for this strip if not already done
    if strip_id not in instate['explosions']:
        instate['explosions'][strip_id] = []
    
    # Add new explosions
    if random.random() < 0.1:  # Reduced probability
        center = random.randint(0, strip_length - 1)
        instate['explosions'][strip_id].append({
            'center': center,
            'radius': 0.0,
            'max_radius': 5 + random.randint(0, 5),
            'expansion_rate': 30 + random.random() * 20,
            'intensity': 0.7 + random.random() * 0.3,
            'life': 0.0,
            'duration': 0.3 + random.random() * 0.3,
        })
    
    # Update existing explosions
    new_explosions = []
    for explosion in instate['explosions'][strip_id]:
        # Update life cycle
        explosion['life'] += delta_time / explosion['duration']
        
        # Update radius - grows quickly then stops
        if explosion['life'] < 0.6:  # Grow during first 60% of life
            explosion['radius'] += explosion['expansion_rate'] * delta_time
            explosion['radius'] = min(explosion['radius'], explosion['max_radius'])
        
        # Keep if still alive
        if explosion['life'] < 1.0:
            new_explosions.append(explosion)
            
            # Calculate intensity based on life cycle
            if explosion['life'] < 0.2:
                explosion_alpha = explosion['intensity'] * (explosion['life'] / 0.2)
            elif explosion['life'] > 0.7:
                explosion_alpha = explosion['intensity'] * (1.0 - ((explosion['life'] - 0.7) / 0.3))
            else:
                explosion_alpha = explosion['intensity']
            
            # Calculate the explosion effect
            center = explosion['center']
            radius = int(explosion['radius'])
            
            for i in range(max(0, center - radius), min(strip_length, center + radius + 1)):
                dist = abs(i - center) / explosion['radius'] if explosion['radius'] > 0 else 1.0
                if dist <= 1.0:
                    # Determine color based on distance from center
                    if dist < 0.3:
                        # Center - yellow/white hot
                        h, s, v = instate['colors']['yellow']
                        s *= 0.5  # Desaturate for white-hot look
                        v = min(1.0, v * 1.2)
                    elif dist < 0.7:
                        # Middle - orange
                        h, s, v = instate['colors']['orange']
                    else:
                        # Outer edge - red
                        h, s, v = instate['colors']['bright_red']
                    
                    r, g, b = hsv_to_rgb(h, s, v)
                    intensity = explosion_alpha * (1.0 - dist**2)
                    
                    # Add to buffer with maximum blending
                    curr_r, curr_g, curr_b, curr_a = buffer[i]
                    buffer[i] = [
                        max(curr_r, r),
                        max(curr_g, g),
                        max(curr_b, b),
                        max(curr_a, intensity)
                    ]
    
    # Replace with updated list
    instate['explosions'][strip_id] = new_explosions


def apply_flame_effect(buffer, strip_length, strip_id, instate, outstate, delta_time, direction=-1):
    """Apply flame effects (direction: -1=up, 0=outward from center, 1=down)"""
    import numpy as np
    import random
    from colorsys import hsv_to_rgb
    
    # Initialize flames for this strip if not already done
    if strip_id not in instate['flames']:
        instate['flames'][strip_id] = []
        
        # INCREASE INITIAL FLAMES: Change from strip_length // 10 to strip_length // 5
        num_initial_flames = max(5, strip_length // 5)  # More initial flames
        
        for _ in range(num_initial_flames):
            pos = random.randint(0, strip_length - 1)
            
            # Determine direction based on mode
            if direction == 0:  # Outward from center
                middle = strip_length // 2
                flame_dir = 1 if pos < middle else -1
            else:
                flame_dir = direction
            
            instate['flames'][strip_id].append({
                'position': pos,
                'speed': 25 + random.random() * 15,
                # INCREASE FLAME SIZE: from 8+7 to 12+10
                'size':82 + random.randint(0, 30),  # Larger flames
                'intensity': 0.6 + random.random() * 0.4,
                'direction': flame_dir,
                'life': 0.0,
                # INCREASE DURATION: from 0.8+0.7 to 1.2+1.0
                'duration': 3.2 + random.random() * 4.0  # Longer-lasting flames
            })
    
    # INCREASE FLAME CREATION PROBABILITY: from 0.1 to 0.2
    if random.random() < 0.2:  # 20% chance per frame for new flames
        if direction == -1:  # Upward flames (spine)
            pos = strip_length - random.randint(1, 10)  # Start near bottom
            flame_dir = direction
        elif direction == 1:  # Downward flames
            pos = random.randint(1, 10)  # Start near top
            flame_dir = direction
        else:  # Outward flames (base)
            pos = random.randint(0, strip_length - 1)
            middle = strip_length // 2
            flame_dir = 1 if pos < middle else -1
        instate['flames'][strip_id].append({
            'position': pos,
            'speed': 15 + random.random() * 15,
            # INCREASE FLAME SIZE: from 8+7 to 12+10
            'size':42 + random.randint(0, 30),  # Larger flames
            'intensity': 0.6 + random.random() * 0.4,
            'direction': flame_dir,
            'life': 0.0,
            # INCREASE DURATION: from 0.8+0.7 to 1.2+1.0
            'duration': 1.2 + random.random() * 4.0  # Longer-lasting flames
        })
    
        # instate['flames'][strip_id].append({
        #     'position': pos,
        #     'speed': 15 + random.random() * 15,
        #     # INCREASE FLAME SIZE: from 8+7 to 12+10
        #     'size': 22 + random.randint(0, 10),  # Larger flames
        #     'intensity': 0.6 + random.random() * 0.4,
        #     'direction': flame_dir,
        #     'life': 0.0,
        #     # INCREASE DURATION: from 0.8+0.7 to 1.2+1.0
        #     'duration': 1.2 + random.random() * 1.0  # Longer-lasting flames
        # })
    
    # Update existing flames
    new_flames = []
    for flame in instate['flames'][strip_id]:
        # Update life
        flame['life'] += delta_time / flame['duration']
        
        # Update position based on direction
        flame['position'] += flame['speed'] * flame['direction'] * delta_time
        
        # Keep if still in bounds and alive
        if 0 <= flame['position'] < strip_length and flame['life'] < 1.0:
            new_flames.append(flame)
            
            # Calculate flame intensity based on life cycle
            if flame['life'] < 0.2:
                flame_alpha = flame['intensity'] * (flame['life'] / 0.2)
            elif flame['life'] > 0.7:
                flame_alpha = flame['intensity'] * (1.0 - ((flame['life'] - 0.7) / 0.3))
            else:
                flame_alpha = flame['intensity']
            
            # Draw the flame
            flame_pos = int(flame['position'])
            flame_size = flame['size']
            
            for i in range(max(0, flame_pos - flame_size), min(strip_length, flame_pos + flame_size + 1)):
                # Calculate relative position in flame (0 to 1)
                rel_pos = abs(i - flame_pos) / flame_size
                if rel_pos <= 1.0:
                    # Determine color based on position in flame
                    if rel_pos < 0.3:
                        # Tip of flame - yellow
                        h, s, v = instate['colors']['yellow']
                    elif rel_pos < 0.7:
                        # Middle of flame - orange
                        h, s, v = instate['colors']['orange']
                    else:
                        # Base of flame - red
                        h, s, v = instate['colors']['deep_red']
                    
                    r, g, b = hsv_to_rgb(h, s, v)
                    intensity = flame_alpha * (1.0 - rel_pos)
                    
                    # Add to buffer with maximum blending
                    curr_r, curr_g, curr_b, curr_a = buffer[i]
                    buffer[i] = [
                        max(curr_r, r),
                        max(curr_g, g),
                        max(curr_b, b),
                        max(curr_a, intensity)
                    ]
    
    # Replace with updated list
    instate['flames'][strip_id] = new_flames


def apply_pulse_effect(buffer, strip_length, instate, outstate):
    """Apply pulsing flame effect to other strips"""
    import numpy as np
    import random
    from colorsys import hsv_to_rgb
    
    # Calculate wave parameters for fire effect
    current_time = outstate['current_time']
    
    # Create pulsing effect
    pulse = 0.7 + 0.3 * np.sin(current_time * 5.0)
    
    for i in range(strip_length):
        # Create heat wave pattern
        norm_pos = i / strip_length
        wave1 = 0.5 + 0.5 * np.sin(norm_pos * 4 * np.pi + current_time * 3.0)
        wave2 = 0.5 + 0.5 * np.sin(norm_pos * 7 * np.pi - current_time * 2.0)
        heat = (wave1 * 0.6 + wave2 * 0.4) * pulse
        
        # Determine color based on heat
        if heat > 0.8:
            # Very hot - yellow
            h, s, v = instate['colors']['yellow']
        elif heat > 0.5:
            # Hot - orange
            h, s, v = instate['colors']['orange']
        else:
            # Less hot - deep red
            h, s, v = instate['colors']['deep_red']
        
        # Adjust brightness based on heat
        v_adjusted = v * (0.6 + 0.4 * heat)
        
        r, g, b = hsv_to_rgb(h, s, v_adjusted)
        alpha = 0.3 + 0.7 * heat
        
        buffer[i] = [r, g, b, alpha]
    
    # Add ember sparks with random distribution
    for i in range(strip_length):
        if random.random() < 0.005:  # 0.5% chance per pixel
            h, s, v = instate['colors']['orange']
            r, g, b = hsv_to_rgb(h, s, v)
            spark_intensity = 0.8 + random.random() * 0.2
            
            buffer[i] = [r, g, b, spark_intensity]


def OTO_curious_playful(instate, outstate):
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
    curious_level = outstate.get('control_curious', 0.0)/100
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = curious_level
    
    # Skip rendering if alpha is too low
    if curious_level < 0.01:
        return
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if remaining_time < 10.0:
        fade_alpha = remaining_time / 10.0
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

        if 'spots' in strip.groups:
            # Initialize spots state if not already done
            if strip_id not in instate['spots_state']:
                # Pick initial colors from the playful palette
                color_list = list(instate['colors'].keys())
                instate['spots_state'][strip_id] = {
                    'colors': [np.random.choice(color_list) for _ in range(3)],  # Track 3 colors
                    'positions': [i * strip_length / 3 for i in range(3)],  # Evenly spaced
                    'velocities': [np.random.uniform(2, 5) for _ in range(3)],  # Slow movement
                    'last_color_change': [0.0, 0.0, 0.0],
                    'wave_phase': 0.0
                }
            
            spots = instate['spots_state'][strip_id]
            
            # Update wave phase for gentle animation
            spots['wave_phase'] += delta_time * 0.3  # Very slow wave
            
            # Update color positions (much slower than regular regions)
            for i in range(3):
                spots['positions'][i] += spots['velocities'][i] * delta_time
                spots['positions'][i] %= strip_length
                
                # Occasionally change colors (much less frequently)
                spots['last_color_change'][i] += delta_time
                if spots['last_color_change'][i] > 8.0:  # Change every 8 seconds
                    spots['last_color_change'][i] = 0.0
                    color_list = list(instate['colors'].keys())
                    spots['colors'][i] = np.random.choice(color_list)
                    spots['velocities'][i] = np.random.uniform(2, 5)  # New slow velocity
            
            # Render the colors with smooth blending
            for pixel in range(strip_length):
                # Calculate influence from each color
                total_influence = 0.0
                r, g, b = 0.0, 0.0, 0.0
                
                for i in range(3):
                    # Calculate distance with wrapping
                    dist = abs(pixel - spots['positions'][i])
                    dist = min(dist, strip_length - dist)
                    
                    # Smooth falloff over larger distance
                    influence = max(0, 1.0 - (dist / (strip_length * 0.4)))
                    influence = influence ** 2  # Smoother falloff
                    
                    if influence > 0:
                        h, s, v = instate['colors'][spots['colors'][i]]
                        # Add subtle wave effect
                        v *= 0.8 + 0.2 * np.sin(spots['wave_phase'] + pixel * 0.1)
                        color_r, color_g, color_b = hsv_to_rgb(h, s, v)
                        
                        r += color_r * influence
                        g += color_g * influence
                        b += color_b * influence
                        total_influence += influence
                
                # Normalize and set pixel
                if total_influence > 0:
                    r /= total_influence
                    g /= total_influence
                    b /= total_influence
                    buffer[pixel] = [r, g, b, min(0.8, total_influence)]
                else:
                    buffer[pixel] = [0.1, 0.1, 0.1, 0.2]  # Dim default
            
            continue  # Skip regular processing for spots strips
            
    
            
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
            for pixel in range(strip_length):
                if rgb_buffer[pixel, 3] > 0:
                    buffer[pixel] = [
                        rgb_buffer[pixel, 0], 
                        rgb_buffer[pixel, 1], 
                        rgb_buffer[pixel, 2], 
                        rgb_buffer[pixel, 3]
                    ]
                else:
                    buffer[pixel] = [0, 0, 0, 0]

def OTO_passionate_floral(instate, outstate):
    """
    Generator function that creates a passionate and playful floral-themed pattern.
    
    Features:
    1. Global alpha controlled by outstate['passionate_curious'] value
    2. Floral growth patterns that unfold from origination points with pink and dark blue colors
    3. Hints of green in the growth patterns to represent stems and leaves
    4. Long, moving stripes that fade away as they move
    5. Emphasis on growth and movement away from origination points
    
    Uses HSV colorspace for color generation and blending.
    """
    name = 'passionate_floral'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['bloom_points'] = {}      # Track floral bloom points per strip
        instate['moving_stripes'] = {}    # Track moving stripe patterns
        instate['growth_timer'] = 0.0     # Global timer for coordinated growth
        instate['spots_state'] = {}    
        
        # Color palette (HSV values)
        instate['colors'] = {
            'vivid_pink': [0.9, 0.85, 0.95],      # Vibrant pink
            'soft_pink': [0.95, 0.6, 0.95],       # Softer pink
            'deep_blue': [0.65, 0.9, 0.7],        # Deep blue
            'midnight_blue': [0.7, 0.85, 0.5],    # Darker blue
            'leaf_green': [0.3, 0.8, 0.6],        # Green for leaves/stems
            'light_green': [0.35, 0.6, 0.8],      # Lighter green for new growth
            'white_glow': [0.0, 0.0, 1.0] ,        # White glow for highlights
            'bright_yellow': [0.15, 0.9, 1.0],    # Bright sunny yellow
            'golden_yellow': [0.13, 0.85, 0.9]    # Warm golden yellow
        }
        
        # Growth parameters
        instate['growth_rate'] = 1.0      # Base growth rate
        instate['max_blooms'] = 8         # Maximum blooms per strip
        
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        return

    # Get passionate_curious level from outstate (default to 0)
    passionate_level = outstate.get('control_passionate', 0.0)/100
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = passionate_level
    
    # Skip rendering if alpha is too low
    if passionate_level < 0.01:
        return
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if remaining_time < 10.0:
        fade_alpha = remaining_time / 10.0
        fade_alpha = max(0.0, fade_alpha)
        buffers.generator_alphas[name] = fade_alpha * passionate_level
    
    # Get delta time for animation calculations
    delta_time = outstate['current_time'] - outstate['last_time']
    
    # Update global growth timer - controls overall growth rhythm
    instate['growth_timer'] += delta_time
    growth_phase = (np.sin(instate['growth_timer'] * 0.8) + 1.0) / 2.0  # 0.0-1.0 oscillation
    
    # Adjust growth rate based on passionate level - higher = faster growth
    effective_growth_rate = instate['growth_rate'] * (0.8 + passionate_level * 0.7)
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Process each buffer based on strip type
    for strip_id, buffer in pattern_buffers.items():
        # Skip if strip doesn't exist in manager
        if strip_id not in strip_manager.strips:
            continue
            
        strip = strip_manager.get_strip(strip_id)
        strip_length = len(buffer)
        
        # Start with a dark base
        buffer[:] = [0.0, 0.0, 0.05, 0.2]  # Very dim blue base

        if 'spots' in strip.groups:
            # Initialize spots state if not already done
            if strip_id not in instate['spots_state']:
                instate['spots_state'][strip_id] = {
                    'bloom_phase': np.random.random() * 2 * np.pi,
                    'primary_color': np.random.choice(['vivid_pink', 'soft_pink', 'deep_blue']),
                    'secondary_color': np.random.choice(['leaf_green', 'light_green']),
                    'bloom_intensity': 0.5,
                    'bloom_speed': 0.15 + np.random.random() * 0.1,  # Very slow blooming
                    'color_shift_timer': 0.0,
                    'petal_positions': [np.random.random() * strip_length for _ in range(3)]
                }
            
            spots = instate['spots_state'][strip_id]
            
            # Update bloom phase
            spots['bloom_phase'] += spots['bloom_speed'] * delta_time
            
            # Calculate bloom intensity (breathing effect)
            bloom = 0.4 + 0.6 * (0.5 + 0.5 * np.sin(spots['bloom_phase']))
            
            # Occasionally shift colors
            spots['color_shift_timer'] += delta_time
            if spots['color_shift_timer'] > 10.0:  # Change every 10 seconds
                spots['color_shift_timer'] = 0.0
                # Rotate through floral colors
                if 'pink' in spots['primary_color']:
                    spots['primary_color'] = np.random.choice(['deep_blue', 'midnight_blue'])
                else:
                    spots['primary_color'] = np.random.choice(['vivid_pink', 'soft_pink'])
            
            # Get colors
            primary_h, primary_s, primary_v = instate['colors'][spots['primary_color']]
            secondary_h, secondary_s, secondary_v = instate['colors'][spots['secondary_color']]
            
            # Create gradient across strip
            for pixel in range(strip_length):
                # Calculate position-based variation
                pos_factor = (pixel / strip_length)
                
                # Create petal-like pattern
                petal_effect = 0.0
                for petal_pos in spots['petal_positions']:
                    dist = abs(pixel - petal_pos)
                    dist = min(dist, strip_length - dist)
                    petal_influence = max(0, 1.0 - (dist / (strip_length * 0.2)))
                    petal_effect += petal_influence * 0.5
                
                # Blend colors based on position and petal effect
                if petal_effect > 0.5:
                    # Use primary color for petals
                    h, s, v = primary_h, primary_s, primary_v
                else:
                    # Blend toward secondary for stems/leaves
                    blend = petal_effect
                    h = secondary_h + (primary_h - secondary_h) * blend
                    s = secondary_s + (primary_s - secondary_s) * blend
                    v = secondary_v + (primary_v - secondary_v) * blend
                
                # Apply bloom intensity
                v *= bloom * (0.7 + 0.3 * petal_effect)
                
                # Convert to RGB
                r, g, b = hsv_to_rgb(h, s, v)
                
                buffer[pixel] = [r, g, b, 0.6 + 0.2 * bloom]
            
            # Slowly move petal positions
            for i in range(len(spots['petal_positions'])):
                spots['petal_positions'][i] += 0.5 * delta_time  # Very slow drift
                spots['petal_positions'][i] %= strip_length
            
            continue  # Skip regular processing for spots strips
        
        # Initialize bloom points for this strip if not already done
        if strip_id not in instate['bloom_points']:
            instate['bloom_points'][strip_id] = []
            instate['moving_stripes'][strip_id] = []
            
            # Create initial bloom points
            # Fix the error by ensuring the high value is always greater than the low value
            max_blooms = max(1, min(3, strip_length // 40))
            num_initial_blooms = 1 if max_blooms <= 1 else np.random.randint(1, max_blooms)
            for _ in range(num_initial_blooms):
                # Random position along strip
                pos = np.random.randint(0, strip_length)
                
                # Create bloom with random properties
                create_new_bloom(instate, strip_id, pos, strip_length)

        
        # Determine strip type for customized effects
        is_base = 'base' in strip.groups
        is_spine = 'spine' in strip.groups
        is_heart = 'heart' in strip.groups
        is_brain = 'brain' in strip.groups
        is_ear = 'ear' in strip.groups
        
        # Chance to create new bloom points (varies by strip type)
        bloom_chance = 0.02  # Base chance
        
        # Adjust bloom chance based on strip type
        if is_base:
            bloom_chance = 0.03  # More blooms on base strips
        elif is_spine:
            bloom_chance = 0.02  # Average on spine
        elif is_heart:
            bloom_chance = 0.04  # More on heart
        elif is_brain or is_ear:
            bloom_chance = 0.025  # Slightly more on brain/ear
            
        # Adjust based on number of existing blooms (less likely if many exist)
        max_blooms = instate['max_blooms']
        existing_blooms = len(instate['bloom_points'][strip_id])
        bloom_chance *= max(0.1, 1.0 - (existing_blooms / max_blooms))
        
        # Also factor in passionate level - more passion, more blooms
        bloom_chance *= (0.5 + passionate_level * 1.0)
        
        # Try to create a new bloom
        if np.random.random() < bloom_chance * delta_time * 60:  # Scale by framerate
            # Choose a random position
            pos = np.random.randint(0, strip_length)
            
            # Check if there's already a bloom nearby
            too_close = False
            for bloom in instate['bloom_points'][strip_id]:
                # Calculate distance considering wrapping
                direct_dist = abs(pos - bloom['position'])
                wrapped_dist = strip_length - direct_dist
                distance = min(direct_dist, wrapped_dist)
                
                # Skip if too close to an existing bloom
                if distance < strip_length * 0.15:  # Minimum 15% of strip length between blooms
                    too_close = True
                    break
            
            # Create new bloom if not too close to existing ones
            if not too_close and existing_blooms < max_blooms:
                create_new_bloom(instate, strip_id, pos, strip_length)
        
        # Chance to create new moving stripes
        stripe_chance = 0.01  # Base chance
        
        # Adjust stripe chance based on strip type
        if is_base:
            stripe_chance = 0.02  # More stripes on base
        elif is_spine:
            stripe_chance = 0.03  # Most on spine
        
        # Adjust based on passionate level
        stripe_chance *= (0.5 + passionate_level * 1.0)
        
        # Try to create a new stripe
        if np.random.random() < stripe_chance * delta_time * 60:  # Scale by framerate
            # Choose a random position
            pos = np.random.randint(0, strip_length)
            
            # Random direction
            direction = 1 if np.random.random() < 0.5 else -1
            
            # Random color - primarily pinks and blues
            if np.random.random() < 0.6:  # 60% pink, 40% blue
                color_name = np.random.choice(['vivid_pink', 'soft_pink'])
            else:
                color_name = np.random.choice(['deep_blue', 'midnight_blue'])
            
            h, s, v = instate['colors'][color_name]
            
            # Create a new stripe
            instate['moving_stripes'][strip_id].append({
                'position': pos,
                'direction': direction,
                'speed': 20 + np.random.random() * 40,  # 20-60 pixels per second
                'length': 10 + np.random.randint(0, int(strip_length * 0.2)),  # 10 to 20% of strip length
                'h': h,
                's': s,
                'v': v,
                'alpha': 0.7 + np.random.random() * 0.3,  # 0.7-1.0 alpha
                'age': 0.0,
                'lifetime': 2.0 + np.random.random() * 3.0  # 2-5 seconds lifetime
            })
        
        # Update and render bloom points
        new_blooms = []
        for bloom in instate['bloom_points'][strip_id]:
            # Update growth stage
            bloom['age'] += delta_time * effective_growth_rate
            
            # Calculate growth factor (0.0-1.0)
            growth_progress = min(1.0, bloom['age'] / bloom['growth_time'])
            
            # Calculate the bloom's current radius based on growth
            # Use sigmoid-like function for natural growth curve
            growth_factor = 1.0 / (1.0 + np.exp(-10 * (growth_progress - 0.5)))
            current_radius = bloom['max_radius'] * growth_factor
            
            # Calculate pulsing effect for grown blooms
            pulse_factor = 1.0
            if growth_progress > 0.9:  # Almost fully grown
                # Add gentle pulsing
                pulse_time = outstate['current_time'] * bloom['pulse_rate']
                pulse_factor = 1.0 + 0.1 * np.sin(pulse_time) * (growth_progress - 0.9) * 10
            
            # Calculate final radius with pulse
            render_radius = current_radius * pulse_factor
            
            # Keep bloom if still within lifetime
            if bloom['age'] < bloom['lifetime']:
                new_blooms.append(bloom)
                
                # Calculate alpha based on growth and lifetime
                if bloom['age'] < bloom['growth_time']:
                    # Growing phase - fade in
                    alpha_factor = growth_progress
                else:
                    # Mature phase - maintain then fade out near end of life
                    life_remaining = 1.0 - ((bloom['age'] - bloom['growth_time']) / 
                                           (bloom['lifetime'] - bloom['growth_time']))
                    alpha_factor = max(0.0, life_remaining)
                
                # Draw the bloom
                draw_bloom(buffer, strip_length, bloom, render_radius, alpha_factor, instate['colors'])
                
                # Chance to create a leaf or tendril from mature blooms
                if growth_progress > 0.5 and np.random.random() < 0.01 * delta_time * 60:
                    # Create a moving stripe that originates from this bloom
                    direction = 1 if np.random.random() < 0.5 else -1
                    
                    # Use green for leaves/tendrils
                    color_name = np.random.choice(['leaf_green', 'light_green'])
                    h, s, v = instate['colors'][color_name]
                    
                    # Create a new stripe
                    instate['moving_stripes'][strip_id].append({
                        'position': bloom['position'],
                        'direction': direction,
                        'speed': 30 + np.random.random() * 30,  # 30-60 pixels per second
                        'length': 5 + np.random.randint(0, 10),  # 5-15 pixels
                        'h': h,
                        's': s,
                        'v': v,
                        'alpha': 0.6 + np.random.random() * 0.4,  # 0.6-1.0 alpha
                        'age': 0.0,
                        'lifetime': 1.0 + np.random.random() * 2.0  # 1-3 seconds lifetime
                    })
            
        # Update bloom list
        instate['bloom_points'][strip_id] = new_blooms
        
        # Update and render moving stripes
        new_stripes = []
        for stripe in instate['moving_stripes'][strip_id]:
            # Update position
            stripe['position'] += stripe['speed'] * stripe['direction'] * delta_time
            
            # Update age
            stripe['age'] += delta_time
            
            # Check if stripe is still visible and within lifetime
            if stripe['age'] < stripe['lifetime']:
                new_stripes.append(stripe)
                
                # Calculate alpha based on age
                if stripe['age'] < stripe['lifetime'] * 0.2:
                    # Fade in
                    alpha_factor = stripe['age'] / (stripe['lifetime'] * 0.2)
                else:
                    # Fade out gradually
                    alpha_factor = 1.0 - ((stripe['age'] - stripe['lifetime'] * 0.2) / 
                                         (stripe['lifetime'] * 0.8))
                
                # Calculate final alpha
                alpha = stripe['alpha'] * alpha_factor
                
                # Draw the stripe with fading trail
                draw_stripe(buffer, strip_length, stripe, alpha)
                
        # Update stripe list
        instate['moving_stripes'][strip_id] = new_stripes

def create_new_bloom(instate, strip_id, position, strip_length):
    """Helper function to create a new bloom point with randomized properties"""
    # Randomly choose between pink and blue for primary color
    if np.random.random() < 0.7:  # 70% pink, 30% blue
        primary_color = np.random.choice(['vivid_pink', 'soft_pink'])
    else:
        primary_color = np.random.choice(['deep_blue', 'midnight_blue'])
    
    # Random secondary color for variation
    if primary_color in ['vivid_pink', 'soft_pink']:
        # Pink primary, blue secondary
        secondary_color = np.random.choice(['deep_blue', 'midnight_blue'])
    else:
        # Blue primary, pink secondary
        secondary_color = np.random.choice(['vivid_pink', 'soft_pink'])
    
    # Random growth time and lifetime
    growth_time = 2.0 + np.random.random() * 3.0  # 2-5 seconds to fully grow
    lifetime = growth_time + 3.0 + np.random.random() * 10.0  # Additional 3-13 seconds before fading
    
    # Create the bloom
    instate['bloom_points'][strip_id].append({
        'position': position,
        'max_radius': 5 + np.random.randint(5, max(6, int(strip_length * 0.15))),  # 5-15% of strip length
        'primary_color': primary_color,
        'secondary_color': secondary_color,
        'accent_color': 'leaf_green' if np.random.random() < 0.7 else 'light_green',  # Green accent
        'layers': 2 + np.random.randint(0, 3),  # 2-4 layers
        'age': 0.0,
        'growth_time': growth_time,
        'lifetime': lifetime,
        'pulse_rate': 1.0 + np.random.random() * 2.0,  # 1-3 Hz pulse when mature
        'rotation': np.random.random() * 2 * np.pi  # Random initial rotation
    })

def draw_bloom(buffer, strip_length, bloom, radius, alpha_factor, colors):
    """Draws a floral bloom with multiple layers and colors"""
    center = int(bloom['position'])
    radius_int = int(radius)
    
    # Get colors from palette
    primary_h, primary_s, primary_v = colors[bloom['primary_color']]
    secondary_h, secondary_s, secondary_v = colors[bloom['secondary_color']]
    accent_h, accent_s, accent_v = colors[bloom['accent_color']]
    
    # Calculate layer spacing
    layer_spacing = max(1, radius_int // bloom['layers'])
    
    # Draw layers from outside in
    for layer in range(bloom['layers']):
        # Calculate layer radius
        layer_radius = radius_int - layer * layer_spacing
        if layer_radius <= 0:
            continue
            
        # Calculate layer properties
        layer_norm = layer / max(1, bloom['layers'] - 1)  # 0.0 for outer layer, 1.0 for inner
        
        # Interpolate colors between primary and secondary
        h = primary_h * (1.0 - layer_norm) + secondary_h * layer_norm
        s = primary_s * (1.0 - layer_norm) + secondary_s * layer_norm
        v = primary_v * (1.0 - layer_norm) + secondary_v * layer_norm
        
        # Inner layers are brighter
        v = min(1.0, v * (1.0 + layer_norm * 0.3))
        
        # For the innermost layer, blend with white for a glowing center
        if layer == bloom['layers'] - 1:
            s *= 0.6  # Reduce saturation
            v = min(1.0, v * 1.2)  # Increase brightness
        
        # Convert to RGB
        r, g, b = hsv_to_rgb(h, s, v)
        
        # Calculate layer alpha - inner layers more transparent for a glowing effect
        layer_alpha = (0.7 + 0.3 * layer_norm) * alpha_factor
        
        # Draw the layer with petal-like pattern
        num_petals = 5 + layer  # More petals on outer layers
        rotation = bloom['rotation'] + layer * 0.2  # Rotate each layer slightly
        
        for i in range(-layer_radius, layer_radius + 1):
            pixel_pos = (center + i) % strip_length
            
            # Calculate distance from center
            dist = abs(i) / layer_radius
            
            # Calculate angle from center
            angle = np.arccos(1.0 - dist) if dist < 1.0 else 0
            angle = (angle + rotation) % (2 * np.pi)
            
            # Create petal pattern
            petal_factor = 0.5 + 0.5 * np.cos(angle * num_petals)
            
            # Calculate pixel intensity - stronger at petal centers
            intensity = (1.0 - dist**2) * petal_factor
            
            # Skip if too dim
            if intensity < 0.05:
                continue
                
            # Apply color with intensity
            pixel_alpha = layer_alpha * intensity
            
            # Occasionally add green accent color for leaf-like details
            if layer < bloom['layers'] - 1 and np.random.random() < 0.1:
                r, g, b = hsv_to_rgb(accent_h, accent_s, accent_v)
                
            # Blend with existing pixel
            curr_r, curr_g, curr_b, curr_a = buffer[pixel_pos]
            new_r = max(curr_r, r * intensity)
            new_g = max(curr_g, g * intensity)
            new_b = max(curr_b, b * intensity)
            new_a = max(curr_a, pixel_alpha)
            
            buffer[pixel_pos] = [new_r, new_g, new_b, new_a]

def draw_stripe(buffer, strip_length, stripe, alpha):
    """Draws a moving stripe with a fading trail"""
    # Get stripe properties
    pos = int(stripe['position']) % strip_length
    length = stripe['length']
    direction = stripe['direction']
    h, s, v = stripe['h'], stripe['s'], stripe['v']
    
    # Convert base color to RGB
    base_r, base_g, base_b = hsv_to_rgb(h, s, v)
    
    # Draw the stripe with fading trail
    for i in range(length):
        # Calculate position with direction
        pixel_pos = (pos - i * direction) % strip_length
        
        # Calculate intensity based on position in stripe
        # Head of stripe is brightest, fades toward tail
        intensity = 1.0 - (i / length)**1.5
        
        # Skip if too dim
        if intensity < 0.05:
            continue
            
        # Calculate color with intensity
        r = base_r * intensity
        g = base_g * intensity
        b = base_b * intensity
        a = alpha * intensity
        
        # Blend with existing pixel
        curr_r, curr_g, curr_b, curr_a = buffer[pixel_pos]
        new_r = max(curr_r, r)
        new_g = max(curr_g, g)
        new_b = max(curr_b, b)
        new_a = max(curr_a, a)
        
        buffer[pixel_pos] = [new_r, new_g, new_b, new_a]

# ... existing code ...

def OTO_rage_lightning(instate, outstate):
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
    rage_level = outstate.get('control_rage', 0.0)/100
    
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

        if 'spots' in strip.groups:
            # Initialize spots state if not already done
            if strip_id not in instate['spots_state']:
                instate['spots_state'][strip_id] = {
                    'charge_level': 0.0,
                    'charging': True,
                    'discharge_timer': 0.0,
                    'next_discharge': 2.0 + np.random.random() * 3.0,  # 2-5 seconds between discharges
                    'base_glow': 0.2,
                    'arc_positions': []  # Track small electrical arcs
                }
            
            spots = instate['spots_state'][strip_id]
            
            # Update charge/discharge cycle
            if spots['charging']:
                # Slowly build charge
                spots['charge_level'] += delta_time / 2.0  # 2 seconds to full charge
                spots['charge_level'] = min(1.0, spots['charge_level'])
                
                # Check if ready to discharge
                spots['discharge_timer'] += delta_time
                if spots['discharge_timer'] >= spots['next_discharge']:
                    spots['charging'] = False
                    spots['discharge_timer'] = 0.0
                    # Create some arc positions for the discharge
                    spots['arc_positions'] = [np.random.randint(0, strip_length) 
                                             for _ in range(np.random.randint(2, 5))]
            else:
                # Discharging - faster than charging
                spots['charge_level'] -= delta_time * 3.0  # 0.33 seconds to discharge
                
                if spots['charge_level'] <= 0:
                    spots['charge_level'] = 0.0
                    spots['charging'] = True
                    spots['next_discharge'] = 2.0 + np.random.random() * 3.0
                    spots['arc_positions'] = []
            
            # Choose color based on charge level
            if spots['charge_level'] < 0.3:
                # Low charge - deep blue
                h, s, v = instate['colors']['electric_blue']
            elif spots['charge_level'] < 0.7:
                # Medium charge - lighter blue
                h, s, v = instate['colors']['light_blue']
            else:
                # High charge - yellow/white
                if spots['charging']:
                    h, s, v = instate['colors']['pale_yellow']
                else:
                    h, s, v = instate['colors']['white_hot']
            
            # Adjust brightness based on charge
            v *= (spots['base_glow'] + spots['charge_level'] * 0.8)
            
            # Convert to RGB
            r, g, b = hsv_to_rgb(h, s, v)
            
            # Apply base color to whole strip
            buffer[:] = [r, g, b, 0.3 + spots['charge_level'] * 0.5]
            
            # Add arc effects during discharge
            if not spots['charging'] and spots['arc_positions']:
                arc_h, arc_s, arc_v = instate['colors']['white_hot']
                arc_r, arc_g, arc_b = hsv_to_rgb(arc_h, arc_s, arc_v)
                
                for arc_pos in spots['arc_positions']:
                    # Draw small electrical arc
                    arc_size = 3
                    for offset in range(-arc_size, arc_size + 1):
                        pos = (arc_pos + offset) % strip_length
                        intensity = 1.0 - abs(offset) / arc_size
                        buffer[pos] = [
                            min(1.0, r + arc_r * intensity * 0.5),
                            min(1.0, g + arc_g * intensity * 0.5),
                            min(1.0, b + arc_b * intensity * 0.5),
                            min(1.0, 0.8 * intensity)
                        ]
            
            continue  # Skip regular processing for spots strips

        
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
        for i in range(strip_length):
            # Get current pixel values
            curr_r, curr_g, curr_b, curr_a = buffer[i]
            
            # Generate random color for noise
            noise_color = np.random.choice(list(instate['colors'].keys()))
            h, s, v = instate['colors'][noise_color]
            noise_r, noise_g, noise_b = hsv_to_rgb(h, s, v)
            
            # Calculate noise intensity (20-40% range)
            noise_intensity = instate['base_noise_min'] + np.random.random() * (instate['base_noise_max'] - instate['base_noise_min'])
            
            # Increase noise with rage level
            noise_intensity *= (1.0 + rage_level * 0.5)
            noise_intensity = min(0.6, noise_intensity)  # Cap at 60% for high rage
            
            # Apply noise with additive blending
            new_r = min(1.0, curr_r + (noise_r * noise_intensity))
            new_g = min(1.0, curr_g + (noise_g * noise_intensity))
            new_b = min(1.0, curr_b + (noise_b * noise_intensity))
            new_a = max(curr_a, noise_intensity)
            
            buffer[i] = [new_r, new_g, new_b, new_a]
            


def OTO_contemplative_cosmic(instate, outstate):
    """
    Generator function that creates a contemplative cosmic-themed pattern.
    
    Features:
    1. Global alpha controlled by outstate['control_contemplative'] value
    2. Prominent, flowing nebular color fields (blues, greens, oranges) that blend and smear
    3. Slow-moving field of blinking stars (white)
    4. Green moving splotches resembling cosmic gas clouds
    5. Green/yellow blinking fireflies that move around
    
    The overall effect is a peaceful cosmic environment that evokes 
    contemplation and wonder with vivid, dynamic color flows.
    """
    name = 'contemplative_cosmic'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['stars'] = {}           # Track stars across strips
        instate['splotches'] = {}       # Track green splotches
        instate['nebulae'] = {}         # Track large nebular regions
        instate['color_flows'] = {}     # Track flowing color smears
        instate['fireflies'] = {}       # Track moving fireflies
        instate['active_strips'] = set()  # Track which strips are active
        instate['spots_state'] = {}     # Track slow nebula for spots strips
        

        # Color palette (HSV values)
        instate['colors'] = {
            'white_star': [0.0, 0.0, 1.0],          # Pure white
            'blue_deep': [0.6, 0.8, 0.6],           # Deep blue
            'blue_light': [0.55, 0.6, 0.9],         # Light blue
            'orange_nebula': [0.08, 0.7, 0.8],      # Orange nebula
            'orange_deep': [0.05, 0.85, 0.7],       # Deep orange
            'green_cosmic': [0.35, 0.8, 0.7],       # Cosmic green
            'green_deep': [0.3, 0.9, 0.6],          # Deep green
            'yellow_glow': [0.15, 0.8, 0.9],        # Yellow glow
            'green_yellow': [0.2, 0.85, 0.8],       # Green-yellow
            'purple_cosmic': [0.75, 0.7, 0.6],      # Cosmic purple
            'teal_cosmic': [0.45, 0.7, 0.7]         # Cosmic teal
        }
        
        # Global parameters for nebular flows
        instate['flow_speed_base'] = 2.0      # Base speed for color flows
        instate['flow_update_timer'] = 0.0    # Timer for updating global flow parameters
        instate['global_flow_direction'] = 1  # Global direction that can change occasionally
        
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        
        # Clean up all data structures
        if 'stars' in instate:
            instate['stars'].clear()
        if 'splotches' in instate:
            instate['splotches'].clear()
        if 'nebulae' in instate:
            instate['nebulae'].clear()
        if 'color_flows' in instate:
            instate['color_flows'].clear()
        if 'fireflies' in instate:
            instate['fireflies'].clear()
        if 'active_strips' in instate:
            instate['active_strips'].clear()
            
        return

    # Get contemplative level from outstate (default to 0)
    contemplative_level = outstate.get('control_contemplative', 0.0)/100
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = contemplative_level
    
    # Skip rendering if alpha is too low
    if contemplative_level < 0.01:
        return
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if remaining_time < 10.0:
        fade_alpha = remaining_time / 10.0
        fade_alpha = max(0.0, fade_alpha)
        buffers.generator_alphas[name] = fade_alpha * contemplative_level
    
    # Get delta time for animation calculations
    delta_time = outstate['current_time'] - outstate['last_time']
    current_time = outstate['current_time']
    
    # Update global flow parameters
    instate['flow_update_timer'] += delta_time
    if instate['flow_update_timer'] > 15.0:  # Every 15 seconds, consider changing global flow direction
        instate['flow_update_timer'] = 0.0
        if np.random.random() < 0.4:  # 40% chance to reverse global flow
            instate['global_flow_direction'] *= -1
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Clean up strips that are no longer active
    current_strip_ids = set(pattern_buffers.keys())
    if 'active_strips' in instate:
        removed_strips = instate['active_strips'] - current_strip_ids
        for strip_id in removed_strips:
            # Remove data for strips that are no longer active
            if strip_id in instate['stars']:
                del instate['stars'][strip_id]
            if strip_id in instate['splotches']:
                del instate['splotches'][strip_id]
            if strip_id in instate['nebulae']:
                del instate['nebulae'][strip_id]
            if strip_id in instate['color_flows']:
                del instate['color_flows'][strip_id]
            if strip_id in instate['fireflies']:
                del instate['fireflies'][strip_id]
        
        # Update active strips
        instate['active_strips'] = current_strip_ids
    
    # Process each buffer
    for strip_id, buffer in pattern_buffers.items():
        # Skip if strip doesn't exist in manager
        if strip_id not in strip_manager.strips:
            continue
            
        strip = strip_manager.get_strip(strip_id)
        strip_length = len(buffer)
        
        # Start with a dark base - very deep blue
        buffer[:] = [0.0, 0.0, 0.1, 0.2]  # Very dim blue base

        if 'spots' in strip.groups:
            # Initialize spots state if not already done
            if strip_id not in instate['spots_state']:
                instate['spots_state'][strip_id] = {
                    'nebula_colors': [
                        np.random.choice(['blue_deep', 'blue_light', 'green_cosmic', 'purple_cosmic']),
                        np.random.choice(['orange_nebula', 'teal_cosmic', 'green_deep'])
                    ],
                    'blend_factor': 0.5,
                    'blend_direction': 1,
                    'blend_speed': 0.05,  # Very slow blending
                    'shimmer_phase': np.random.random() * 2 * np.pi,
                    'star_positions': [np.random.randint(0, strip_length) 
                                     for _ in range(max(2, strip_length // 20))],
                    'star_phases': [np.random.random() * 2 * np.pi 
                                  for _ in range(max(2, strip_length // 20))]
                }
            
            spots = instate['spots_state'][strip_id]
            
            # Update blend factor
            spots['blend_factor'] += spots['blend_speed'] * spots['blend_direction'] * delta_time
            
            # Reverse direction at boundaries
            if spots['blend_factor'] >= 1.0:
                spots['blend_factor'] = 1.0
                spots['blend_direction'] = -1
                # Pick new second color
                color_options = ['orange_nebula', 'teal_cosmic', 'green_deep', 'yellow_glow']
                spots['nebula_colors'][1] = np.random.choice(color_options)
            elif spots['blend_factor'] <= 0.0:
                spots['blend_factor'] = 0.0
                spots['blend_direction'] = 1
                # Pick new first color
                color_options = ['blue_deep', 'blue_light', 'green_cosmic', 'purple_cosmic']
                spots['nebula_colors'][0] = np.random.choice(color_options)
            
            # Update shimmer
            spots['shimmer_phase'] += delta_time * 0.2  # Very slow shimmer
            
            # Get nebula colors
            h1, s1, v1 = instate['colors'][spots['nebula_colors'][0]]
            h2, s2, v2 = instate['colors'][spots['nebula_colors'][1]]
            
            # Smooth blend using cosine interpolation
            blend = 0.5 + 0.5 * np.cos(spots['blend_factor'] * np.pi)
            
            # Handle hue wrapping
            h_diff = h2 - h1
            if abs(h_diff) > 0.5:
                if h1 > h2:
                    h2 += 1.0
                else:
                    h1 += 1.0
            
            # Interpolate colors
            h = (h1 * blend + h2 * (1 - blend)) % 1.0
            s = s1 * blend + s2 * (1 - blend)
            v = v1 * blend + v2 * (1 - blend)
            
            # Add shimmer effect
            shimmer = 0.9 + 0.1 * np.sin(spots['shimmer_phase'])
            v *= shimmer
            
            # Convert to RGB
            r, g, b = hsv_to_rgb(h, s, v)
            
            # Apply nebula to entire strip
            buffer[:] = [r, g, b, 0.5]
            
            # Add static stars (slow twinkling)
            for i, star_pos in enumerate(spots['star_positions']):
                # Update star phase slowly
                spots['star_phases'][i] += delta_time * 0.3
                
                # Calculate star brightness
                brightness = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(spots['star_phases'][i]))
                
                # Draw star
                star_h, star_s, star_v = instate['colors']['white_star']
                star_r, star_g, star_b = hsv_to_rgb(star_h, star_s, star_v * brightness)
                
                buffer[star_pos] = [
                    max(r, star_r),
                    max(g, star_g),
                    max(b, star_b),
                    max(0.5, brightness * 0.8)
                ]
            
            continue  # Skip regular processing for spots strips
        
        # Initialize elements for this strip if not already done
        if strip_id not in instate['stars']:
            # Add to active strips
            if 'active_strips' in instate:
                instate['active_strips'].add(strip_id)
                
            # Create stars
            instate['stars'][strip_id] = []
            num_stars = max(3, strip_length // 5)  # Roughly 1 star per 10 pixels
            
            for _ in range(num_stars):
                pos = np.random.randint(0, strip_length)
                instate['stars'][strip_id].append({
                    'position': pos,
                    'brightness': 0.3 + np.random.random() * 0.7,  # 0.3-1.0 brightness
                    'blink_rate': 0.2 + np.random.random() * 1.0,  # 0.2-1.2 Hz
                    'blink_offset': np.random.random() * 2 * np.pi,  # Random phase
                    'movement_speed': 0.5 + np.random.random() * 1.5,  # 0.5-2.0 pixels per second (slow)
                    'direction': 1 if np.random.random() < 0.5 else -1  # Random direction
                })
            
            # Create green splotches
            instate['splotches'][strip_id] = []
            num_splotches = max(1, strip_length // 40)  # Fewer splotches, 1 per ~40 pixels
            
            for _ in range(num_splotches):
                pos = np.random.randint(0, strip_length)
                instate['splotches'][strip_id].append({
                    'position': pos,
                    'size': 5 + np.random.randint(0, 10),  # 5-15 pixels
                    'speed': 1.0 + np.random.random() * 2.0,  # 1-3 pixels per second (slow)
                    'direction': 1 if np.random.random() < 0.5 else -1,  # Random direction
                    'color': np.random.choice(['green_cosmic', 'green_deep']),  # Random green shade
                    'intensity': 0.5 + np.random.random() * 0.5,  # 0.5-1.0 intensity
                    'pulse_rate': 0.1 + np.random.random() * 0.3,  # 0.1-0.4 Hz (slow pulse)
                    'pulse_offset': np.random.random() * 2 * np.pi  # Random phase
                })
            
            # Create large nebulae - main colored regions
            instate['nebulae'][strip_id] = []
            num_nebulae = max(2, strip_length // 30)  # More nebulae: 1 per ~30 pixels
            
            for _ in range(num_nebulae):
                pos = np.random.randint(0, strip_length)
                color = np.random.choice(['blue_deep', 'blue_light', 'orange_nebula', 
                                         'green_cosmic', 'purple_cosmic', 'teal_cosmic'])
                
                instate['nebulae'][strip_id].append({
                    'position': pos,
                    'size': 20 + np.random.randint(0, 30),  # 20-50 pixels (much larger)
                    'speed': 0.8 + np.random.random() * 1.2,  # 0.8-2.0 pixels per second
                    'direction': 1 if np.random.random() < 0.5 else -1,  # Random direction
                    'color': color,
                    'intensity': 0.5 + np.random.random() * 0.5,  # 0.5-1.0 intensity (much brighter)
                    'pulse_rate': 0.05 + np.random.random() * 0.15,  # 0.05-0.2 Hz (very slow pulse)
                    'pulse_offset': np.random.random() * 2 * np.pi  # Random phase
                })
            
            # Create color flows - dynamic smearing nebular effects
            instate['color_flows'][strip_id] = []
            
            # Create multiple color flow layers with different characteristics
            num_flow_layers = 4 + np.random.randint(0, 3)  # 4-6 layers (more layers)
            
            for _ in range(num_flow_layers):
                # Choose complementary colors for interesting blends
                color_pairs = [
                    ('blue_deep', 'orange_nebula'),
                    ('blue_light', 'orange_deep'),
                    ('green_cosmic', 'purple_cosmic'),
                    ('green_deep', 'blue_deep'),
                    ('yellow_glow', 'blue_light'),
                    ('teal_cosmic', 'orange_deep'),
                    ('green_yellow', 'purple_cosmic')
                ]
                color_pair = color_pairs[np.random.randint(0, len(color_pairs))]
                
                # Create a flow layer
                instate['color_flows'][strip_id].append({
                    'color1': color_pair[0],
                    'color2': color_pair[1],
                    'offset': np.random.random() * strip_length,  # Random starting position
                    'scale': 15 + np.random.random() * 45,  # 15-60 pixel scale (wavelength)
                    'speed': instate['flow_speed_base'] * (0.5 + np.random.random()),  # Varied speeds
                    'direction': 1 if np.random.random() < 0.5 else -1,  # Random direction
                    'amplitude': 0.4 + np.random.random() * 0.6,  # 0.4-1.0 intensity (much stronger)
                    'phase_shift': np.random.random() * 2 * np.pi,  # Random phase
                    'drift_rate': 0.1 + np.random.random() * 0.3,  # 0.1-0.4 units per second
                    'distortion': 0.5 + np.random.random() * 1.5  # 0.5-2.0 distortion factor
                })
            
            # Create fireflies
            instate['fireflies'][strip_id] = []
            num_fireflies = max(2, strip_length // 15)  # 1 per ~15 pixels
            
            for _ in range(num_fireflies):
                pos = np.random.randint(0, strip_length)
                color = np.random.choice(['green_yellow', 'yellow_glow'])
                
                instate['fireflies'][strip_id].append({
                    'position': pos,
                    'speed': 5.0 + np.random.random() * 15.0,  # 5-20 pixels per second
                    'direction': 1 if np.random.random() < 0.5 else -1,  # Random direction
                    'color': color,
                    'blink_rate': 0.5 + np.random.random() * 1.5,  # 0.5-2.0 Hz
                    'blink_offset': np.random.random() * 2 * np.pi,  # Random phase
                    'blink_duration': 0.7 + np.random.random() * 0.8,  # 0.7-1.5 second blinks
                    'last_direction_change': current_time,
                    'direction_change_interval': 2.0 + np.random.random() * 4.0  # 2-6 seconds
                })
        
        # Set up a new array to collect all color contributions for proper blending
        # Using numpy for efficient operations
        r_values = np.zeros(strip_length)
        g_values = np.zeros(strip_length)
        b_values = np.zeros(strip_length)
        a_values = np.zeros(strip_length)
        
        # Render large nebulae (largest, slowest color regions) first
        for nebula in instate['nebulae'][strip_id]:
            # Update position (slow movement)
            nebula['position'] += nebula['speed'] * nebula['direction'] * delta_time
            
            # Handle wrapping
            nebula['position'] %= strip_length
            
            # Calculate pulse effect (very slow and subtle)
            pulse_factor = 0.8 + 0.2 * (0.5 + 0.5 * np.sin(current_time * nebula['pulse_rate'] * 2 * np.pi + nebula['pulse_offset']))
            
            # Calculate final intensity
            intensity = nebula['intensity'] * pulse_factor
            
            # Get color
            h, s, v = instate['colors'][nebula['color']]
            
            # Convert to RGB once for efficiency
            base_r, base_g, base_b = hsv_to_rgb(h, s, v)
            
            # Draw the nebula with soft gradient
            center = int(nebula['position'])
            size = nebula['size']
            
            # Create pixel positions array with wrapping
            positions = np.arange(center - size, center + size + 1) % strip_length
            
            # Calculate distances from center
            distances = np.abs(np.arange(-size, size + 1)) / size
            
            # Create softened edge profile for organic look
            # Use a power function for a more natural falloff
            edge_profile = np.power(1.0 - distances, 2) * intensity
            
            # Add some noise for texture
            noise = np.random.random(len(edge_profile)) * 0.15 - 0.075  # ±7.5% noise
            edge_profile = np.clip(edge_profile + noise, 0, 1)
            
            # Apply to color arrays (vectorized)
            mask = edge_profile > 0.01  # Skip very low intensity
            if np.any(mask):
                edge_masked = edge_profile[mask]
                pos_masked = positions[mask]
                
                r_values[pos_masked] += base_r * edge_masked * 0.7
                g_values[pos_masked] += base_g * edge_masked * 0.7
                b_values[pos_masked] += base_b * edge_masked * 0.7
                a_values[pos_masked] += edge_masked * 0.6
        
        # Render dynamic color flows (nebular smearing)
        for flow in instate['color_flows'][strip_id]:
            # Update flow position
            flow_direction = flow['direction'] * instate['global_flow_direction']  # Combine with global direction
            flow['offset'] += flow['speed'] * flow_direction * delta_time
            flow['phase_shift'] += flow['drift_rate'] * delta_time
            
            # Keep offset in range
            flow['offset'] %= strip_length
            
            # Get the two colors
            h1, s1, v1 = instate['colors'][flow['color1']]
            h2, s2, v2 = instate['colors'][flow['color2']]
            
            # Create positions array
            positions = np.arange(strip_length)
            
            # Calculate normalized positions in the flow
            pos = (positions + flow['offset']) % strip_length
            pos_norm = pos / flow['scale']
            
            # Add distortion for more organic appearance
            distortion = np.sin(pos_norm * 0.5 + flow['phase_shift']) * flow['distortion']
            
            # Calculate main flow pattern
            flow_values = 0.5 + 0.5 * np.sin(pos_norm * 2 * np.pi + flow['phase_shift'] + distortion)
            
            # Apply amplitude modulation
            intensities = flow_values * flow['amplitude']
            
            # Create mask for pixels to update (where intensity is significant)
            mask = intensities > 0.1
            
            if np.any(mask):
                # Handle hue interpolation for wrapped color space
                h_diff = h2 - h1
                if abs(h_diff) > 0.5:
                    if h1 > h2:
                        h2 += 1.0
                    else:
                        h1 += 1.0
                
                # Work with the masked values for efficiency
                flow_vals_masked = flow_values[mask]
                masked_positions = positions[mask]
                masked_intensities = intensities[mask]
                
                # Interpolate colors and convert to RGB for each pixel
                for i, pos in enumerate(masked_positions):
                    flow_val = flow_vals_masked[i]
                    intensity = masked_intensities[i]
                    
                    # Interpolate HSV
                    h_value = (h1 * (1.0 - flow_val) + h2 * flow_val) % 1.0
                    s_value = s1 * (1.0 - flow_val) + s2 * flow_val
                    v_value = v1 * (1.0 - flow_val) + v2 * flow_val
                    
                    # Convert to RGB
                    r, g, b = hsv_to_rgb(h_value, s_value, v_value)
                    
                    # Add to color arrays
                    r_values[pos] += r * intensity * 0.8
                    g_values[pos] += g * intensity * 0.8
                    b_values[pos] += b * intensity * 0.8
                    a_values[pos] += intensity * 0.7
        
        # Apply combined nebular effects to buffer (vectorized)
        mask = a_values > 0
        if np.any(mask):
            indices = np.where(mask)[0]
            
            # Ensure values are in valid range (vectorized)
            r_values[indices] = np.minimum(1.0, r_values[indices])
            g_values[indices] = np.minimum(1.0, g_values[indices])
            b_values[indices] = np.minimum(1.0, b_values[indices])
            a_values[indices] = np.minimum(0.45, a_values[indices])  # Cap alpha
            
            # Set buffer with nebular colors (needs loop due to buffer structure)
            for i in indices:
                buffer[i] = [r_values[i], g_values[i], b_values[i], a_values[i]]
        
        # Update and render stars
        for star in instate['stars'][strip_id]:
            # Update position
            star['position'] += star['movement_speed'] * star['direction'] * delta_time
            star['position'] %= strip_length
            
            # Calculate brightness with blink effect
            blink_factor = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(current_time * star['blink_rate'] * 2 * np.pi + star['blink_offset']))
            brightness = star['brightness'] * blink_factor
            
            # Draw the star
            pos = int(star['position'])
            
            # Get color and convert to RGB
            h, s, v = instate['colors']['white_star']
            r, g, b = hsv_to_rgb(h, s, v * brightness)
            alpha = 0.3 + 0.7 * brightness
            
            # Add to buffer with additive blending
            curr_r, curr_g, curr_b, curr_a = buffer[pos]
            buffer[pos] = [
                max(curr_r, r),
                max(curr_g, g),
                max(curr_b, b),
                max(curr_a, alpha)
            ]
            
            # Add small glow around bright stars
            if brightness > 0.7:
                for offset in [-1, 1]:
                    glow_pos = (pos + offset) % strip_length
                    glow_alpha = alpha * 0.5  # Half the brightness
                    
                    curr_r, curr_g, curr_b, curr_a = buffer[glow_pos]
                    buffer[glow_pos] = [
                        max(curr_r, r * 0.5),
                        max(curr_g, g * 0.5),
                        max(curr_b, b * 0.5),
                        max(curr_a, glow_alpha)
                    ]
        
        # Update and render splotches
        for splotch in instate['splotches'][strip_id]:
            # Update position
            splotch['position'] += splotch['speed'] * splotch['direction'] * delta_time
            splotch['position'] %= strip_length
            
            # Calculate pulse effect
            pulse_factor = 0.7 + 0.3 * (0.5 + 0.5 * np.sin(current_time * splotch['pulse_rate'] * 2 * np.pi + splotch['pulse_offset']))
            intensity = splotch['intensity'] * pulse_factor
            
            # Draw the splotch
            center = int(splotch['position'])
            size = splotch['size']
            
            # Create pixel positions array with wrapping
            positions = np.arange(center - size, center + size + 1) % strip_length
            
            # Calculate distances from center (normalized)
            distances = np.abs(np.arange(-size, size + 1)) / size
            
            # Skip pixels that are too far
            mask = distances <= 1.0
            if np.any(mask):
                positions = positions[mask]
                distances = distances[mask]
                
                # Calculate intensities (vectorized)
                pixel_intensities = intensity * (1.0 - distances**2)  # Quadratic falloff
                
                # Get color
                h, s, v = instate['colors'][splotch['color']]
                
                # Apply to each pixel
                for i, pos in enumerate(positions):
                    pixel_intensity = pixel_intensities[i]
                    r, g, b = hsv_to_rgb(h, s, v * pixel_intensity)
                    alpha = pixel_intensity * 0.7
                    
                    # Add to buffer with additive blending
                    curr_r, curr_g, curr_b, curr_a = buffer[pos]
                    buffer[pos] = [
                        max(curr_r, r),
                        max(curr_g, g),
                        max(curr_b, b),
                        max(curr_a, alpha)
                    ]
        
        # Update fireflies - use new list to avoid modifying while iterating
        new_fireflies = []
        
        for firefly in instate['fireflies'][strip_id]:
            # Check if it's time to change direction
            if current_time - firefly['last_direction_change'] > firefly['direction_change_interval']:
                firefly['direction'] *= -1  # Reverse direction
                firefly['last_direction_change'] = current_time
                firefly['direction_change_interval'] = 2.0 + np.random.random() * 4.0  # New interval
            
            # Update position
            firefly['position'] += firefly['speed'] * firefly['direction'] * delta_time
            firefly['position'] %= strip_length
            
            # Calculate blink effect - more distinct on/off for fireflies
            blink_phase = (current_time * firefly['blink_rate']) % 1.0
            
            # Fireflies have a brief "on" period
            is_on = blink_phase < firefly['blink_duration']
            
            # Add to new list
            new_fireflies.append(firefly)
            
            # Render visible fireflies
            if is_on:
                # Draw the firefly
                pos = int(firefly['position'])
                
                # Get color
                h, s, v = instate['colors'][firefly['color']]
                v = min(1.0, v * 1.2)  # Make brighter
                
                # Convert to RGB
                r, g, b = hsv_to_rgb(h, s, v)
                alpha = 0.9
                
                # Add to buffer with additive blending
                curr_r, curr_g, curr_b, curr_a = buffer[pos]
                buffer[pos] = [
                    max(curr_r, r),
                    max(curr_g, g),
                    max(curr_b, b),
                    max(curr_a, alpha)
                ]
                
                # Add small glow around firefly
                for offset in [-1, 1]:
                    glow_pos = (pos + offset) % strip_length
                    
                    # Glow is more green/yellow regardless of firefly color
                    glow_h, glow_s, glow_v = instate['colors']['green_yellow']
                    glow_r, glow_g, glow_b = hsv_to_rgb(glow_h, glow_s * 0.7, glow_v * 0.7)
                    
                    glow_alpha = 0.6  # Noticeable glow
                    
                    # Add to buffer with additive blending
                    curr_r, curr_g, curr_b, curr_a = buffer[glow_pos]
                    buffer[glow_pos] = [
                        max(curr_r, glow_r),
                        max(curr_g, glow_g),
                        max(curr_b, glow_b),
                        max(curr_a, glow_alpha)
                    ]
        
        # Replace firefly list - important to avoid memory leaks
        instate['fireflies'][strip_id] = new_fireflies

def OTO_neutral_positive(instate, outstate):
    """
    Generator function that creates a neutral/positive-themed pattern across all strips.
    
    Features:
    1. Global alpha controlled by outstate['control_neutral'] value
    2. Evokes sun shining through green tree foliage and sunflowers
    3. Brain strips pulse with gold and yellow colors with occasional flashes
    4. Base strips show light blue with ripples like a shallow ocean viewed from underwater
    5. Overall creates a peaceful, natural, positive atmosphere
    
    Optimized with numpy vectorization for performance.
    """
    name = 'neutral_positive'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['sun_rays'] = {}          # Track sun ray effects
        instate['foliage'] = {}           # Track foliage patterns
        instate['sunflowers'] = {}        # Track sunflower patterns
        instate['brain_pulse'] = {}       # Track brain pulse state
        instate['water_ripples'] = {}     # Track water ripple effects
        instate['flash_timer'] = 0.0      # Timer for occasional flashes
        instate['spots_state'] = {} 
        
        # Color palette (HSV values)
        instate['colors'] = {
            'sunshine_yellow': [0.13, 0.85, 0.95],   # Bright sunshine yellow
            'golden_sun': [0.11, 0.9, 0.9],          # Golden sunlight
            'leaf_green_light': [0.3, 0.7, 0.7],     # Light green foliage
            'leaf_green_dark': [0.35, 0.8, 0.5],     # Darker green foliage
            'sky_blue': [0.6, 0.3, 0.95],            # Light sky blue
            'water_blue': [0.55, 0.6, 0.8],          # Ocean water blue
            'sunflower_center': [0.09, 0.9, 0.7],    # Sunflower center brown
            'sunflower_petal': [0.13, 1.0, 1.0]      # Bright sunflower petal
        }
        
        # Initialize brain pulse parameters
        instate['brain_pulse_rate'] = 0.5           # Pulses per second
        instate['brain_flash_chance'] = 0.02        # Chance per second for a flash
        instate['global_sun_position'] = 0.0        # Global sun position (0.0-1.0)
        instate['sun_movement_speed'] = 0.05        # Speed of sun movement
        
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        return

    # Get neutral level from outstate (default to 0)
    neutral_level = outstate.get('control_neutral', 0.0)/100
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = neutral_level
    
    # Skip rendering if alpha is too low
    if neutral_level < 0.01:
        return
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if remaining_time < 10.0:
        fade_alpha = remaining_time / 10.0
        fade_alpha = max(0.0, fade_alpha)
        buffers.generator_alphas[name] = fade_alpha * neutral_level
    
    # Get delta time for animation calculations
    delta_time = outstate['current_time'] - outstate['last_time']
    current_time = outstate['current_time']
    
    # Update global sun position (0.0-1.0 cycle)
    instate['global_sun_position'] = (instate['global_sun_position'] + 
                                      instate['sun_movement_speed'] * delta_time) % 1.0
    
    # Update flash timer for brain effects
    instate['flash_timer'] += delta_time
    should_flash = False
    if instate['flash_timer'] >= 1.0:  # Check once per second
        instate['flash_timer'] = 0.0
        if np.random.random() < instate['brain_flash_chance']:
            should_flash = True
    
    # Calculate global sun visibility factor
    sun_factor = 0.5 + 0.5 * np.sin(instate['global_sun_position'] * 2 * np.pi)
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Process each buffer
    for strip_id, buffer in pattern_buffers.items():
        # Skip if strip doesn't exist in manager
        if strip_id not in strip_manager.strips:
            continue
            
        strip = strip_manager.get_strip(strip_id)
        strip_length = len(buffer)
        strip_groups = strip.groups

        if 'spots' in strip_groups:
            # Initialize spots state if not already done
            if strip_id not in instate['spots_state']:
                instate['spots_state'][strip_id] = {
                    'sun_phase': np.random.random() * 2 * np.pi,
                    'color_phase': 0.0,
                    'current_palette': 'morning',  # morning, noon, evening
                    'palette_timer': 0.0,
                    'dappled_light': [np.random.random() for _ in range(strip_length)]
                }
            
            spots = instate['spots_state'][strip_id]
            
            # Update sun phase (gentle breathing)
            spots['sun_phase'] += delta_time * 0.25  # Very slow pulse
            
            # Update palette timer
            spots['palette_timer'] += delta_time
            if spots['palette_timer'] > 12.0:  # Change every 12 seconds
                spots['palette_timer'] = 0.0
                # Cycle through times of day
                if spots['current_palette'] == 'morning':
                    spots['current_palette'] = 'noon'
                elif spots['current_palette'] == 'noon':
                    spots['current_palette'] = 'evening'
                else:
                    spots['current_palette'] = 'morning'
            
            # Get base color based on palette
            if spots['current_palette'] == 'morning':
                # Soft golden morning light
                h, s, v = instate['colors']['golden_sun']
                s *= 0.7  # Softer saturation
            elif spots['current_palette'] == 'noon':
                # Bright sunshine
                h, s, v = instate['colors']['sunshine_yellow']
                v *= 0.9  # Slightly less intense
            else:  # evening
                # Warm golden hour
                h, s, v = instate['colors']['golden_sun']
                h += 0.02  # Shift slightly toward orange
                s *= 0.9
            
            # Calculate sun intensity
            sun_intensity = 0.6 + 0.4 * (0.5 + 0.5 * np.sin(spots['sun_phase']))
            v *= sun_intensity
            
            # Apply dappled light effect (like through leaves)
            for pixel in range(strip_length):
                # Slowly vary the dappled pattern
                spots['dappled_light'][pixel] += (np.random.random() - 0.5) * 0.01
                spots['dappled_light'][pixel] = max(0.3, min(1.0, spots['dappled_light'][pixel]))
                
                # Apply dappled effect
                pixel_v = v * spots['dappled_light'][pixel]
                
                # Convert to RGB
                r, g, b = hsv_to_rgb(h, s, pixel_v)
                
                # Add slight green tint occasionally (foliage effect)
                if spots['dappled_light'][pixel] < 0.5:
                    g += 0.1
                    g = min(1.0, g)
                
                buffer[pixel] = [r, g, b, 0.6]
            
            continue  # Skip regular processing for spots strips
        
        # Initialize elements for this strip if not already done
        if strip_id not in instate['sun_rays']:
            # Create sun rays
            instate['sun_rays'][strip_id] = []
            num_rays = max(2, strip_length // 20)  # About 1 ray per 20 pixels
            
            for _ in range(num_rays):
                pos = np.random.randint(0, strip_length)
                width = 3 + np.random.randint(0, 7)  # 3-10 pixels wide
                
                instate['sun_rays'][strip_id].append({
                    'position': pos,
                    'width': width,
                    'intensity': 0.6 + np.random.random() * 0.4,  # 0.6-1.0 intensity
                    'movement_speed': 1.0 + np.random.random() * 2.0,  # 1-3 pixels/sec
                    'direction': 1 if np.random.random() < 0.5 else -1  # Random direction
                })
            
            # Create foliage elements
            instate['foliage'][strip_id] = []
            num_leaves = max(3, strip_length // 15)  # About 1 leaf per 15 pixels
            
            for _ in range(num_leaves):
                pos = np.random.randint(0, strip_length)
                size = 5 + np.random.randint(0, 10)  # 5-15 pixels
                
                instate['foliage'][strip_id].append({
                    'position': pos,
                    'size': size,
                    'color': 'leaf_green_light' if np.random.random() < 0.6 else 'leaf_green_dark',
                    'sway_amplitude': 1.0 + np.random.random() * 2.0,  # 1-3 pixels
                    'sway_frequency': 0.2 + np.random.random() * 0.4,  # 0.2-0.6 Hz
                    'sway_offset': np.random.random() * 2 * np.pi  # Random phase
                })
            
            # Create sunflowers
            instate['sunflowers'][strip_id] = []
            num_flowers = max(1, strip_length // 40)  # About 1 flower per 40 pixels
            
            for _ in range(num_flowers):
                pos = np.random.randint(0, strip_length)
                size = 8 + np.random.randint(0, 7)  # 8-15 pixels
                
                instate['sunflowers'][strip_id].append({
                    'position': pos,
                    'size': size,
                    'rotate_speed': 0.1 + np.random.random() * 0.2,  # 0.1-0.3 radians/sec
                    'rotation': np.random.random() * 2 * np.pi,  # Random initial rotation
                    'sway_amplitude': 0.5 + np.random.random() * 1.5,  # 0.5-2.0 pixels
                    'sway_frequency': 0.1 + np.random.random() * 0.3,  # 0.1-0.4 Hz
                    'sway_offset': np.random.random() * 2 * np.pi  # Random phase
                })
            
            # Create water ripples for base strips
            instate['water_ripples'][strip_id] = []
            num_ripples = max(2, strip_length // 25)  # About 1 ripple per 25 pixels
            
            for _ in range(num_ripples):
                pos = np.random.randint(0, strip_length)
                wavelength = 15 + np.random.randint(0, 20)  # 15-35 pixels
                
                instate['water_ripples'][strip_id].append({
                    'position': pos,
                    'wavelength': wavelength,
                    'amplitude': 0.2 + np.random.random() * 0.3,  # 0.2-0.5 intensity variation
                    'speed': 5.0 + np.random.random() * 10.0,  # 5-15 pixels/sec
                    'direction': 1 if np.random.random() < 0.5 else -1  # Random direction
                })
            
            # Initialize brain pulse state
            if 'brain' in strip_groups:
                instate['brain_pulse'][strip_id] = {
                    'phase': np.random.random(),  # Random initial phase
                    'flash_active': False,
                    'flash_intensity': 0.0,
                    'flash_decay': 5.0  # Flash decay rate
                }
        
        # Determine strip type for customized effects
        is_base = 'base' in strip_groups
        is_brain = 'brain' in strip_groups
        
        # Create position array for the whole strip once
        positions = np.arange(strip_length)
        
        # Start with a base color depending on strip type
        # Start with a base color depending on strip type
        if is_base:
            # Light blue base for water effect
            h, s, v = instate['colors']['water_blue']
            r, g, b = hsv_to_rgb(h, s, v * 0.7)  # Slightly dimmed
            buffer[:] = [r, g, b, 0.5]
        elif is_brain:
            # Darker base for brain to allow pulses to stand out
            buffer[:] = [0.1, 0.1, 0.05, 0.1]
        else:
            # Sky blue base for other strips - brighter, more saturated sky blue
            h, s, v = instate['colors']['sky_blue']
            # Increase saturation and brightness for more vibrant sky blue
            s = min(1.0, s * 1.2)  # More saturated
            v = 0.9  # Brighter
            r, g, b = hsv_to_rgb(h, s, v)
            buffer[:] = [r, g, b, 0.5]  # Increased alpha for more visibility

            
        # Apply different effects based on strip type
        if is_base:
            # Water ripples for base strips - vectorized implementation
            
            # Initialize arrays for ripple and caustic effects
            ripple_effect = np.zeros(strip_length)
            
            # Update and apply all ripples
            for ripple in instate['water_ripples'][strip_id]:
                # Update ripple position
                ripple['position'] += ripple['speed'] * ripple['direction'] * delta_time
                ripple['position'] %= strip_length
                
                # Create distance array (accounting for wrapping)
                direct_dist = np.abs(positions - ripple['position'])
                wrapped_dist = strip_length - direct_dist
                distance = np.minimum(direct_dist, wrapped_dist)
                
                # Calculate ripple effect based on sine wave (vectorized)
                ripple_phase = distance / ripple['wavelength'] * 2 * np.pi
                ripple_effect += ripple['amplitude'] * np.sin(ripple_phase + current_time * 2.0)
            
            # Add underwater caustics effect (light patterns) - vectorized
            caustic_effect = 0.15 * np.sin(positions * 0.2 + current_time * 1.5) * np.cos(positions * 0.1 + current_time)
            
            # Combine effects
            total_effect = ripple_effect + caustic_effect
            
            # Extract current buffer values
            curr_r = buffer[:, 0]
            curr_g = buffer[:, 1]
            curr_b = buffer[:, 2]
            curr_a = buffer[:, 3]
            
            # Create masks for bright and dark areas
            bright_mask = total_effect > 0
            dark_mask = ~bright_mask
            
            # Apply effects to bright areas (vectorized)
            if np.any(bright_mask):
                # Shift toward white-blue for bright spots
                new_r = curr_r.copy()
                new_g = curr_g.copy()
                new_b = curr_b.copy()
                
                new_r[bright_mask] += total_effect[bright_mask] * 0.4
                new_g[bright_mask] += total_effect[bright_mask] * 0.5
                new_b[bright_mask] += total_effect[bright_mask] * 0.6
            
            # Apply effects to dark areas (vectorized)
            if np.any(dark_mask):
                # Shift toward deeper blue for darker spots
                new_r[dark_mask] = np.maximum(0, curr_r[dark_mask] + total_effect[dark_mask] * 0.3)
                new_g[dark_mask] = np.maximum(0, curr_g[dark_mask] + total_effect[dark_mask] * 0.4)
                new_b[dark_mask] = np.maximum(0, curr_b[dark_mask] + total_effect[dark_mask] * 0.3)
            
            # Ensure all values are in valid range (vectorized)
            new_r = np.clip(new_r, 0.0, 1.0)
            new_g = np.clip(new_g, 0.0, 1.0)
            new_b = np.clip(new_b, 0.0, 1.0)
            
            # Update buffer (vectorized)
            buffer[:, 0] = new_r
            buffer[:, 1] = new_g
            buffer[:, 2] = new_b
            
        elif is_brain:
            # Brain pulse effect with gold and yellow - vectorized implementation
            if strip_id in instate['brain_pulse']:
                pulse_state = instate['brain_pulse'][strip_id]
                
                # Update pulse phase
                pulse_state['phase'] += instate['brain_pulse_rate'] * delta_time
                pulse_state['phase'] %= 1.0
                
                # Calculate pulse intensity (0.0-1.0)
                phase = pulse_state['phase']
                if phase < 0.3:
                    # Rising phase
                    pulse_intensity = phase / 0.3
                elif phase < 0.5:
                    # Peak phase
                    pulse_intensity = 1.0
                else:
                    # Falling phase
                    pulse_intensity = 1.0 - ((phase - 0.5) / 0.5)
                
                # Handle flash effect
                if should_flash and not pulse_state['flash_active']:
                    pulse_state['flash_active'] = True
                    pulse_state['flash_intensity'] = 1.0
                
                if pulse_state['flash_active']:
                    # Update flash intensity
                    pulse_state['flash_intensity'] -= pulse_state['flash_decay'] * delta_time
                    
                    if pulse_state['flash_intensity'] <= 0:
                        pulse_state['flash_active'] = False
                        pulse_state['flash_intensity'] = 0.0
                
                # Combine regular pulse with flash
                combined_intensity = max(pulse_intensity, pulse_state['flash_intensity'])
                
                # Create spatial variation (vectorized)
                position_factor = 0.8 + 0.2 * np.sin(positions * 0.2 + current_time)
                
                # Create alternating color mask
                alternate_mask = (positions % 2 == 0)
                
                # Create color arrays
                h_values = np.zeros(strip_length)
                s_values = np.zeros(strip_length)
                v_values = np.zeros(strip_length)
                
                # Set colors based on alternating pattern
                h_golden, s_golden, v_golden = instate['colors']['golden_sun']
                h_sunshine, s_sunshine, v_sunshine = instate['colors']['sunshine_yellow']
                
                # Apply alternating colors (vectorized)
                h_values[alternate_mask] = h_golden
                s_values[alternate_mask] = s_golden
                v_values[alternate_mask] = v_golden
                
                h_values[~alternate_mask] = h_sunshine
                s_values[~alternate_mask] = s_sunshine
                v_values[~alternate_mask] = v_sunshine
                
                # If flash is active, use golden color everywhere
                if pulse_state['flash_active']:
                    h_values[:] = h_golden
                    s_values[:] = s_golden
                    v_values[:] = v_golden
                
                # Adjust brightness based on pulse (vectorized)
                v_adjusted = v_values * (0.3 + 0.7 * combined_intensity * position_factor)
                
                # Convert to RGB
                # Unfortunately we need to convert each pixel individually
                r_values = np.zeros(strip_length)
                g_values = np.zeros(strip_length)
                b_values = np.zeros(strip_length)
                
                for i in range(strip_length):
                    r, g, b = hsv_to_rgb(h_values[i], s_values[i], v_adjusted[i])
                    r_values[i] = r
                    g_values[i] = g
                    b_values[i] = b
                
                # Set alpha based on intensity (vectorized)
                alpha_values = 0.2 + 0.8 * combined_intensity * position_factor
                
                # Update buffer (vectorized)
                buffer[:, 0] = r_values
                buffer[:, 1] = g_values
                buffer[:, 2] = b_values
                buffer[:, 3] = alpha_values
            
        else:
            # Default effect: Sunlight through foliage - vectorized implementation
            
            # Initialize buffer to capture foliage
            foliage_r = np.zeros(strip_length)
            foliage_g = np.zeros(strip_length)
            foliage_b = np.zeros(strip_length)
            foliage_a = np.zeros(strip_length)
            
            # Draw foliage first as background
            for leaf in instate['foliage'][strip_id]:
                # Calculate swaying motion
                sway = leaf['sway_amplitude'] * np.sin(current_time * leaf['sway_frequency'] * 2 * np.pi + 
                                                      leaf['sway_offset'])
                
                # Calculate actual position with sway
                actual_pos = (leaf['position'] + sway) % strip_length
                center = int(actual_pos)
                size = leaf['size']
                
                # Create range of positions for this leaf
                leaf_range = np.arange(center - size, center + size + 1) % strip_length
                
                # Calculate distances from center (normalized)
                rel_positions = np.arange(-size, size + 1)
                distances = np.abs(rel_positions) / size
                
                # Calculate leaf shape factor - more oval than circular (vectorized)
                valid_mask = distances <= 1.0
                shape_factors = np.zeros_like(distances)
                shape_factors[valid_mask] = 1.0 - distances[valid_mask]**1.5
                
                # Calculate final intensities
                intensities = shape_factors * 0.7  # Leaves are partially transparent
                
                # Add some texture/variation to the leaf (vectorized)
                texture = 0.15 * np.sin(rel_positions * 0.8 + leaf['position'] * 0.3)
                
                # Get leaf color
                h, s, v = instate['colors'][leaf['color']]
                
                # Apply texture to color
                v_adjusted = np.clip(v * (1.0 + texture), 0.0, 1.0)
                
                # Convert to RGB (need to do each pixel individually due to hsv_to_rgb limitations)
                for i, pos in enumerate(leaf_range):
                    if intensities[i] > 0.1:  # Skip very low intensity pixels
                        r, g, b = hsv_to_rgb(h, s, v_adjusted[i])
                        
                        # Blend with existing foliage buffer using maximum
                        if intensities[i] > foliage_a[pos]:
                            foliage_r[pos] = r
                            foliage_g[pos] = g
                            foliage_b[pos] = b
                            foliage_a[pos] = intensities[i]
            
            # Apply foliage to buffer where alpha is significant
            foliage_mask = foliage_a > 0.1
            if np.any(foliage_mask):
                idx = np.where(foliage_mask)[0]
                for i in idx:
                    # Blend with existing buffer
                    intensity = foliage_a[i]
                    curr_r, curr_g, curr_b, curr_a = buffer[i]
                    
                    buffer[i] = [
                        foliage_r[i] * intensity + curr_r * (1 - intensity),
                        foliage_g[i] * intensity + curr_g * (1 - intensity),
                        foliage_b[i] * intensity + curr_b * (1 - intensity),
                        max(curr_a, intensity)
                    ]
            
            # Draw sunflowers
            for flower in instate['sunflowers'][strip_id]:
                # Update rotation
                flower['rotation'] += flower['rotate_speed'] * delta_time
                
                # Calculate swaying motion
                sway = flower['sway_amplitude'] * np.sin(current_time * flower['sway_frequency'] * 2 * np.pi + 
                                                        flower['sway_offset'])
                
                # Calculate actual position with sway
                actual_pos = (flower['position'] + sway) % strip_length
                center = int(actual_pos)
                
                # Draw the sunflower petals
                num_petals = 8
                
                # Get petal color
                h, s, v = instate['colors']['sunflower_petal']
                r_petal, g_petal, b_petal = hsv_to_rgb(h, s, v)
                
                for petal in range(num_petals):
                    # Calculate petal direction
                    angle = flower['rotation'] + petal * (2 * np.pi / num_petals)
                    
                    # Calculate petal positions (extending from center)
                    petal_length = int(flower['size'] * 0.8)
                    petal_dir_x = np.cos(angle)
                    
                    # Calculate all positions for this petal at once
                    lengths = np.arange(petal_length)
                    petal_positions = (center + np.round(petal_dir_x * lengths)).astype(int) % strip_length
                    
                    # Set all petal pixels at once
                    buffer[petal_positions] = [r_petal, g_petal, b_petal, 0.8]
                
                # Draw center
                center_size = max(1, int(flower['size'] * 0.3))
                
                # Create range of positions for center
                center_range = np.arange(center - center_size, center + center_size + 1) % strip_length
                
                # Calculate distances from center (normalized)
                rel_positions = np.arange(-center_size, center_size + 1)
                distances = np.abs(rel_positions) / center_size
                
                # Calculate intensity based on distance (vectorized)
                valid_mask = distances <= 1.0
                intensities = np.zeros_like(distances)
                intensities[valid_mask] = 1.0 - distances[valid_mask]**2
                
                # Add texture to center (vectorized)
                texture = 0.1 * np.sin(rel_positions * 2.0 + current_time)
                
                # Get center color
                h, s, v = instate['colors']['sunflower_center']
                
                # Apply texture to color (vectorized)
                v_adjusted = np.clip(v * (1.0 + texture), 0.0, 1.0)
                
                # Convert to RGB and apply to buffer
                for i, pos in enumerate(center_range):
                    if valid_mask[i]:
                        r, g, b = hsv_to_rgb(h, s, v_adjusted[i])
                        buffer[pos] = [r, g, b, 0.9]
            
            # Draw sun rays last (on top) - vectorized where possible
            for ray in instate['sun_rays'][strip_id]:
                # Update position
                ray['position'] += ray['movement_speed'] * ray['direction'] * delta_time
                ray['position'] %= strip_length
                
                # Draw the ray
                center = int(ray['position'])
                width = ray['width']
                
                # Create range of positions for this ray
                ray_range = np.arange(center - width, center + width + 1) % strip_length
                
                # Calculate distances from center (normalized)
                rel_positions = np.arange(-width, width + 1)
                distances = np.abs(rel_positions) / width
                
                # Calculate soft edge profile (vectorized)
                valid_mask = distances <= 1.0
                edge_profiles = np.zeros_like(distances)
                edge_profiles[valid_mask] = (1.0 - distances[valid_mask]**2) * ray['intensity'] * sun_factor
                
                # Add shimmer effect (vectorized)
                shimmer = 0.2 * np.sin(current_time * 5.0 + ray_range * 0.3)
                
                # Get sunshine color
                h, s, v = instate['colors']['sunshine_yellow']
                
                # Apply to buffer
                for i, pos in enumerate(ray_range):
                    if edge_profiles[i] > 0.05:  # Skip very low intensity pixels
                        # Less saturated for sun rays
                        r, g, b = hsv_to_rgb(h, s * 0.7, v * (1.0 + shimmer[i]))
                        
                        # Get current pixel values
                        curr_r, curr_g, curr_b, curr_a = buffer[pos]
                        
                        # Blend additively for light rays
                        edge_profile = edge_profiles[i]
                        buffer[pos] = [
                            min(1.0, curr_r + r * edge_profile * 0.7),
                            min(1.0, curr_g + g * edge_profile * 0.7),
                            min(1.0, curr_b + b * edge_profile * 0.7),
                            max(curr_a, edge_profile * 0.7)
                        ]
                        
def OTO_danger_pulse(instate, outstate):
    """
    Generator function that creates a pulsing effect indicating danger.
    
    Features:
    1. Global alpha controlled by outstate['control_danger'] value
    2. All pixels slowly cycle from 50% to 100% brightness
    3. Uses red color to indicate danger
    
    Simple but effective for indicating danger state.
    """
    name = 'danger_pulse'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['pulse_phase'] = 0.0      # Current phase of the pulse (0.0-1.0)
        instate['pulse_speed'] = 0.5      # Pulses per second (adjust for faster/slower)
        instate['spots_state'] = {}   
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        return

    # Get danger level from outstate (default to 0)
    danger_level = outstate.get('control_danger', 0.0)/100
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = danger_level
    
    # Skip rendering if alpha is too low
    if danger_level < 0.01:
        return
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if remaining_time < 10.0:
        fade_alpha = remaining_time / 10.0
        fade_alpha = max(0.0, fade_alpha)
        buffers.generator_alphas[name] = fade_alpha * danger_level
    
    # Get delta time for animation calculations
    delta_time = outstate['current_time'] - outstate['last_time']
    
    # Update pulse phase
    instate['pulse_phase'] += instate['pulse_speed'] * delta_time
    instate['pulse_phase'] %= 1.0  # Keep within 0-1 range
    
    # Calculate brightness factor using sine wave for smooth pulsing
    # Map sine wave from -1 to 1 → 0.5 to 1.0 brightness
    brightness = 0.5 + 0.25 * (1.0 + np.sin(instate['pulse_phase'] * 2 * np.pi))
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Set danger color (bright red)
    r, g, b = 1.0, 1.0, 1.0
    
    # Process each buffer
    for strip_id, buffer in pattern_buffers.items():
        # Skip if strip doesn't exist in manager
        if strip_id not in strip_manager.strips:
            continue

        strip = strip_manager.get_strip(strip_id)
        strip_groups = strip.groups
        
        # Special handling for spots strips - much slower, gentler pulse
        if 'spots' in strip_groups:
            # Initialize spots state if not already done
            if strip_id not in instate['spots_state']:
                instate['spots_state'][strip_id] = {
                    'pulse_phase': np.random.random() * 2 * np.pi,  # Random starting phase
                    'pulse_speed': 0.1 + np.random.random() * 0.05,  # 0.1-0.15 Hz (very slow)
                    'base_brightness': 0.3,  # Dimmer base level
                    'pulse_amplitude': 0.2   # Smaller variation (30% to 50% brightness)
                }
            
            spots = instate['spots_state'][strip_id]
            
            # Update spots pulse phase (much slower)
            spots['pulse_phase'] += spots['pulse_speed'] * delta_time
            spots['pulse_phase'] %= 1.0
            
            # Calculate spots brightness with smaller variation
            # Using smoother cosine wave for even gentler transition
            spots_brightness = spots['base_brightness'] + spots['pulse_amplitude'] * (
                0.5 + 0.5 * np.cos(spots['pulse_phase'] * 2 * np.pi)
            )
            
            # Apply a warmer, less intense red for spots
            spots_r = 0.8  # Slightly less intense red
            spots_g = 0.1  # Tiny bit of green for warmth
            spots_b = 0.0
            
            # Fill buffer with gentler pulsing
            buffer[:] = [
                spots_r * spots_brightness, 
                spots_g * spots_brightness, 
                spots_b * spots_brightness, 
                0.6  # Lower alpha for spots
            ]
            
        else:
            # Regular strips - normal danger pulse
            # Fill buffer with pulsing red at calculated brightness
            buffer[:] = [r * brightness, g * brightness, b * brightness, 0.8]

            
        # Fill buffer with pulsing red at calculated brightness
        buffer[:] = [r * brightness, g * brightness, b * brightness, 0.8]