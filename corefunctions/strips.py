from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Literal
import numpy as np
import json
import yaml
import math

@dataclass
class LEDStrip:
    """Represents a single LED strip with its properties and metadata"""
    
    # Basic properties
    id: str                           # Unique identifier for the strip
    length: int                       # Number of LEDs in strip
    type: Literal["line", "circle", "arc"]   # Type of strip (line, circle, or arc)
    groups: List[str] = field(default_factory=list)  # All groups this strip belongs to

    # Physical properties
    indices: Optional[np.ndarray] = None  # Physical indices if they differ from sequential
    coordinates: Optional[np.ndarray] = None  # 3D coordinates of each LED
    distance: Optional[np.ndarray] = None  # Normalized distance along the strip (0-1)
    
    # Position data (stored for reference)
    position_data: Dict[str, Any] = field(default_factory=dict)
    
    # Additional custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and initialize derived properties"""
        # Ensure positions has correct shape
        # If indices not provided, create sequential indices
        if self.indices is None:
            self.indices = np.arange(self.length)
        
        # Initialize distance (normalized position along strip)
        self.distance = self.indices / (self.length - 1) if self.length > 1 else np.zeros(1)
        
        # Generate 3D coordinates based on type and position data
        self._generate_coordinates()
    
    def _generate_coordinates(self):
        """Generate 3D coordinates for each LED based on strip type"""
        if self.type == "line":
            self._generate_line_coordinates()
        elif self.type == "circle":
            self._generate_circle_coordinates()
        elif self.type == "arc":
            self._generate_arc_coordinates()
        elif self.type == "concatenated":
            # For concatenated strips, coordinates are directly set during concatenation
            # No need to generate anything here
            pass
        else:
            raise ValueError(f"Unsupported strip type: {self.type}")
    
    def _generate_line_coordinates(self):
        """Generate coordinates for a line strip"""
        if "start" not in self.position_data or "end" not in self.position_data:
            # Create default coordinates if position data not provided
            self.coordinates = np.zeros((self.length, 3))
            return
            
        start = np.array(self.position_data["start"])
        end = np.array(self.position_data["end"])
        
        # Generate points along the line
        self.coordinates = np.zeros((self.length, 3))
        for i in range(self.length):
            t = i / (self.length - 1) if self.length > 1 else 0
            self.coordinates[i] = start + t * (end - start)
    
    def _generate_circle_coordinates(self):
        """Generate coordinates for a circular strip"""
        if "center" not in self.position_data or "radius" not in self.position_data:
            # Create default coordinates if position data not provided
            self.coordinates = np.zeros((self.length, 3))
            return
            
        center = np.array(self.position_data["center"])
        radius = self.position_data["radius"]
        normal = np.array(self.position_data.get("normal", [0, 1, 0]))
        start_angle = self.position_data.get("start_angle", 0.0)
        
        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)
        
        # Find perpendicular vectors to the normal to define the circle plane
        # First perpendicular vector
        if abs(normal[0]) < abs(normal[1]):
            perp1 = np.array([1, 0, 0])
        else:
            perp1 = np.array([0, 1, 0])
            
        perp1 = perp1 - np.dot(perp1, normal) * normal
        perp1 = perp1 / np.linalg.norm(perp1)
        
        # Second perpendicular vector (cross product)
        perp2 = np.cross(normal, perp1)
        
        # Generate points around the circle
        self.coordinates = np.zeros((self.length, 3))
        for i in range(self.length):
            angle = start_angle + (i / self.length) * 2 * np.pi
            point = center + radius * (perp1 * np.cos(angle) + perp2 * np.sin(angle))
            self.coordinates[i] = point
    
    def _generate_arc_coordinates(self):
        """Generate coordinates for an arc strip"""
        if ("start" not in self.position_data or 
            "end" not in self.position_data or 
            "arc_degrees" not in self.position_data):
            # Create default coordinates if position data not provided
            self.coordinates = np.zeros((self.length, 3))
            return
            
        start = np.array(self.position_data["start"])
        end = np.array(self.position_data["end"])
        arc_degrees = self.position_data["arc_degrees"]
        normal = np.array(self.position_data.get("normal", [0, 1, 0]))
        
        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)
        
        # Handle special case: if arc_degrees is close to 0 or 180, special handling
        if abs(arc_degrees) < 0.001:
            # Generate a straight line
            self.coordinates = np.zeros((self.length, 3))
            for i in range(self.length):
                t = i / (self.length - 1) if self.length > 1 else 0
                self.coordinates[i] = start + t * (end - start)
            return
        
        # Calculate chord and midpoint
        chord = end - start
        chord_length = np.linalg.norm(chord)
        midpoint = (start + end) / 2
        
        # We need a direction perpendicular to the chord in the plane of the arc
        # This will be the direction along which the center lies
        
        # First, normalize the chord direction
        chord_unit = chord / chord_length if chord_length > 0 else np.array([1.0, 0.0, 0.0])
        
        # The perpendicular direction is cross product of normal and chord_unit
        bisector_dir = np.cross(normal, chord_unit)
        bisector_dir = bisector_dir / np.linalg.norm(bisector_dir)
        
        # Convert arc angle to radians
        arc_radians = math.radians(abs(arc_degrees))
        
        # Calculate radius using the chord length and arc angle
        # For a circular arc: chord_length = 2 * radius * sin(arc_angle/2)
        radius = chord_length / (2 * math.sin(arc_radians / 2)) if abs(math.sin(arc_radians / 2)) > 1e-10 else 1e10
        
        # Calculate the distance from midpoint to center along the bisector
        # Using: radius^2 = (chord/2)^2 + h^2, where h is the distance
        h = math.sqrt(max(0, radius**2 - (chord_length/2)**2))
        
        # The sign of h depends on the arc angle and direction
        # For arc > 180 degrees, center is on the opposite side of the chord
        if abs(arc_degrees) > 180:
            h = -h
            
        # For negative arc angle, we go in the opposite direction
        if arc_degrees < 0:
            bisector_dir = -bisector_dir
        
        # Calculate the center
        center = midpoint + h * bisector_dir
        
        # Initialize coordinates array
        self.coordinates = np.zeros((self.length, 3))
        
        # Calculate vectors from center to start and end for verification
        vec_to_start = start - center
        #vec_to_end = end - center
        
        # Use the actual radius (distance from center to points)
        radius_actual = np.linalg.norm(vec_to_start)
        
        # Create a basis in the plane for the circle
        # First basis vector is the normalized vector from center to start
        basis1 = vec_to_start / radius_actual
        
        # Second basis vector is perpendicular to both normal and basis1
        basis2 = np.cross(normal, basis1)
        basis2 = basis2 / np.linalg.norm(basis2)
        
        # Calculate the total sweep angle
        total_sweep = arc_radians
        
        # Generate all points along the arc
        for i in range(self.length):
            # Calculate the angle for this point (0 at start, arc_radians at end)
            t = i / (self.length - 1) if self.length > 1 else 0
            angle = t * total_sweep
            
            # For negative arc_degrees, we go clockwise instead of counterclockwise
            if arc_degrees < 0:
                angle = -angle
                
            # Calculate the point using parametric equation of a circle
            x = math.cos(angle)
            y = math.sin(angle)
            
            # Combine the basis vectors to get the point
            point = center + radius_actual * (basis1 * x + basis2 * y)
            self.coordinates[i] = point
class StripManager:
    """Manages a collection of LED strips with query capabilities"""
    
    def __init__(self):
        self.strips: Dict[str, LEDStrip] = {}
        self._groups_cache: Dict[str, List[str]] = {}  # Cache for group lookups
        self.concatenated_strips: Dict[str, LEDStrip] = {} 

    def add_strip(self, strip: LEDStrip) -> None:
        """Add a strip to the collection"""
        self.strips[strip.id] = strip
        self._invalidate_caches()
    
    def remove_strip(self, strip_id: str) -> None:
        """Remove a strip from the collection"""
        if strip_id in self.strips:
            del self.strips[strip_id]
            self._invalidate_caches()
    
    def get_strip(self, strip_id: str) -> LEDStrip:
        """Get a specific strip by ID"""
        if strip_id not in self.strips:
            raise KeyError(f"Strip {strip_id} not found")
        return self.strips[strip_id]
    
    def get_strip_ids(self) -> List[str]:
        """Get all strip IDs"""
        return list(self.strips.keys())
    
    def get_all_strips(self) -> Dict[str, LEDStrip]:
        """Get all strips"""
        return self.strips
    
    def get_strips_by_group(self, group: str) -> Dict[str, LEDStrip]:
        """Get all strips that belong to a specific group"""
        if group not in self._groups_cache:
            self._groups_cache[group] = [
                strip_id for strip_id, strip in self.strips.items()
                if group in strip.groups
            ]
        
        return {
            strip_id: self.strips[strip_id]
            for strip_id in self._groups_cache[group]
        }
    
    def get_spatial_strips(self, center: Tuple[float, float, float], 
                          radius: float) -> Dict[str, LEDStrip]:
        """Get strips that have LEDs within a sphere of specified radius from center"""
        center_array = np.array(center)
        result = {}
        
        for strip_id, strip in self.strips.items():
            if strip.coordinates is None:
                continue
                
            # Calculate distances from center to each LED in the strip
            distances = np.linalg.norm(strip.coordinates - center_array, axis=1)
            
            # If any LED is within radius, include this strip
            if np.any(distances <= radius):
                result[strip_id] = strip
        
        return result
    
    def _invalidate_caches(self) -> None:
        """Clear caches when strips are modified"""
        self._groups_cache.clear()
    
    def create_buffers(self) -> Dict[str, np.ndarray]:
        """Create a buffer dictionary with arrays for each strip (for rendering)"""
        return {
            strip_id: np.zeros((strip.length, 4), dtype=np.float32)  # RGBA
            for strip_id, strip in self.strips.items()
        }

    def create_dmx_senders(self) -> Dict[str, Any]:
        """Create DMX senders for each unique IP address in strip metadata"""
        from corefunctions.ImageToDMX import SACNPixelSender
        
        # Group strips by IP address
        ip_strips = {}
        for strip_id, strip in self.strips.items():
            ip = strip.metadata.get('REC_IP')
            if ip:
                if ip not in ip_strips:
                    ip_strips[ip] = []
                ip_strips[ip].append(strip)
        
        # Create receivers config for each IP
        senders = {}
        for ip, strips in ip_strips.items():
            # Sort strips by Strip_num to ensure correct order
            strips.sort(key=lambda s: s.metadata.get('Strip_num', 0))
            
            # Group strips by type
            rgb_strips = [s for s in strips if s.metadata.get('Type', 'RGB') == 'RGB']
            rgbw_strips = [s for s in strips if s.metadata.get('Type', 'RGBW') == 'RGBW']
            rgbw3_strips = [s for s in strips if s.metadata.get('Type', 'RGBW3') == 'RGBW3']
            dmx_strips = [s for s in strips if s.metadata.get('Type', 'DMX') == 'DMX']
            rgb4_strips = [s for s in strips if s.metadata.get('Type', 'RGB4') == 'RGB4']
            
            # Prepare receivers list to handle multiple types per IP
            receivers = []
            
            # Add RGB receiver if needed
            if rgb_strips:
                # Group RGB strips by output if specified
                output_groups = {}
                for strip in rgb_strips:
                    output = strip.metadata.get('Output', 'default')
                    if output not in output_groups:
                        output_groups[output] = []
                    output_groups[output].append(strip)
                
                for output, group_strips in output_groups.items():
                    group_strips.sort(key=lambda s: s.metadata.get('Strip_num', 0))
                    min_strip_num = min(s.metadata.get('Strip_num', 0) for s in group_strips)
                    
                    rgb_receiver = {
                        'ip': ip,
                        'pixel_count': sum(strip.length for strip in group_strips),
                        'strip_info': [(strip.id, strip.length, strip.metadata.get('Direction', 1), 'RGB') 
                                    for strip in group_strips],
                        'type': 'RGB',
                        'output': output if output != 'default' else None,
                        'strip_ids': [s.id for s in group_strips],
                        'strip_num_start': min_strip_num
                    }
                    receivers.append(rgb_receiver)
            
            # Add RGBW receiver if needed
            if rgbw_strips:
                # Group RGBW strips by output if specified
                output_groups = {}
                for strip in rgbw_strips:
                    output = strip.metadata.get('Output', 'default')
                    if output not in output_groups:
                        output_groups[output] = []
                    output_groups[output].append(strip)
                
                for output, group_strips in output_groups.items():
                    group_strips.sort(key=lambda s: s.metadata.get('Strip_num', 0))
                    min_strip_num = min(s.metadata.get('Strip_num', 0) for s in group_strips)
                    
                    rgbw_receiver = {
                        'ip': ip,
                        'pixel_count': sum(strip.length for strip in group_strips),
                        'strip_info': [(strip.id, strip.length, strip.metadata.get('Direction', 1), 'RGBW') 
                                    for strip in group_strips],
                        'type': 'RGBW',
                        'output': output if output != 'default' else None,
                        'strip_ids': [s.id for s in group_strips],
                        'strip_num_start': min_strip_num
                    }
                    receivers.append(rgbw_receiver)
            
            # Handle RGBW3 strips - group by Output for continuous packing
            if rgbw3_strips:
                # Group RGBW3 strips by Output
                output_groups = {}
                for strip in rgbw3_strips:
                    output = strip.metadata.get('Output', 'default')
                    if output not in output_groups:
                        output_groups[output] = []
                    output_groups[output].append(strip)
                
                # Create a receiver for each output group
                for output, group_strips in output_groups.items():
                    # Sort strips within output group by Strip_num
                    group_strips.sort(key=lambda s: s.metadata.get('Strip_num', 0))
                    min_strip_num = min(s.metadata.get('Strip_num', 0) for s in group_strips)
                    
                    rgbw3_receiver = {
                        'ip': ip,
                        'pixel_count': sum(strip.length for strip in group_strips),
                        'strip_info': [(strip.id, strip.length, strip.metadata.get('Direction', 1), 'RGBW3') 
                                    for strip in group_strips],
                        'type': 'RGBW3',
                        'output': output,
                        'strip_ids': [strip.id for strip in group_strips],
                        'strip_num_start': min_strip_num
                    }
                    receivers.append(rgbw3_receiver)
            
            # Add DMX receiver if needed
            if dmx_strips:
                # Group DMX strips by output if specified
                output_groups = {}
                for strip in dmx_strips:
                    output = strip.metadata.get('Output', 'default')
                    if output not in output_groups:
                        output_groups[output] = []
                    output_groups[output].append(strip)
                
                for output, group_strips in output_groups.items():
                    group_strips.sort(key=lambda s: s.metadata.get('Strip_num', 0))
                    min_strip_num = min(s.metadata.get('Strip_num', 0) for s in group_strips)
                    
                    dmx_receiver = {
                        'ip': ip,
                        'pixel_count': sum(strip.length for strip in group_strips),
                        'strip_info': [(strip.id, strip.length, strip.metadata.get('Direction', 1), 'DMX') 
                                    for strip in group_strips],
                        'type': 'DMX',
                        'output': output if output != 'default' else None,
                        'strip_ids': [s.id for s in group_strips],
                        'strip_num_start': min_strip_num
                    }
                    receivers.append(dmx_receiver)
            
            # Handle RGB4 strips - group by Output
            if rgb4_strips:
                # Group RGB4 strips by Output
                output_groups = {}
                for strip in rgb4_strips:
                    output = strip.metadata.get('Output', 'default')
                    if output not in output_groups:
                        output_groups[output] = []
                    output_groups[output].append(strip)
                
                # Create a receiver for each output group
                for output, group_strips in output_groups.items():
                    # Sort strips within output group by Strip_num
                    group_strips.sort(key=lambda s: s.metadata.get('Strip_num', 0))
                    min_strip_num = min(s.metadata.get('Strip_num', 0) for s in group_strips)
                    
                    rgb4_receiver = {
                        'ip': ip,
                        'pixel_count': sum(strip.length for strip in group_strips),
                        'strip_info': [(strip.id, strip.length, strip.metadata.get('Direction', 1), 'RGB4') 
                                    for strip in group_strips],
                        'type': 'RGB4',
                        'output': output,
                        'strip_ids': [strip.id for strip in group_strips],
                        'strip_num_start': min_strip_num
                    }
                    receivers.append(rgb4_receiver)
            
            # Sort receivers by the minimum Strip_num to maintain order
            receivers.sort(key=lambda r: r.get('strip_num_start', float('inf')))
            
            # Create strip info for all strips on this IP
            all_strip_info = []
            for strip in strips:
                strip_type = strip.metadata.get('Type', 'RGB')
                all_strip_info.append((
                    strip.id, 
                    strip.length, 
                    strip.metadata.get('Direction', 1), 
                    strip_type
                ))
            
            # Print info about this IP's strips
            strip_types = []
            if rgb_strips: 
                outputs = {}
                for s in rgb_strips:
                    output = s.metadata.get('Output', 'default')
                    if output not in outputs:
                        outputs[output] = 0
                    outputs[output] += s.length
                for output, count in outputs.items():
                    strip_types.append(f"RGB{f' Output {output}' if output != 'default' else ''} ({count} pixels)")
                    
            if rgbw_strips:
                outputs = {}
                for s in rgbw_strips:
                    output = s.metadata.get('Output', 'default')
                    if output not in outputs:
                        outputs[output] = 0
                    outputs[output] += s.length
                for output, count in outputs.items():
                    strip_types.append(f"RGBW{f' Output {output}' if output != 'default' else ''} ({count} pixels)")
                    
            if rgbw3_strips:
                outputs = {}
                for s in rgbw3_strips:
                    output = s.metadata.get('Output', 'default')
                    if output not in outputs:
                        outputs[output] = 0
                    outputs[output] += s.length
                for output, count in outputs.items():
                    strip_types.append(f"RGBW3 Output {output} ({count} pixels)")
                    
            if dmx_strips:
                outputs = {}
                for s in dmx_strips:
                    output = s.metadata.get('Output', 'default')
                    if output not in outputs:
                        outputs[output] = 0
                    outputs[output] += s.length
                for output, count in outputs.items():
                    strip_types.append(f"DMX{f' Output {output}' if output != 'default' else ''} ({count} pixels)")
            
            # Show RGB4 strips by output
            if rgb4_strips:
                outputs = {}
                for s in rgb4_strips:
                    output = s.metadata.get('Output', 'default')
                    if output not in outputs:
                        outputs[output] = 0
                    outputs[output] += s.length
                
                for output, count in outputs.items():
                    strip_types.append(f"RGB4 Output {output} ({count} pixels)")
            
            print(f"IP {ip} has: {', '.join(strip_types)}")
            
            try:
                # Create a single sender for all receivers on this IP
                sender = SACNPixelSender(receivers)
                
                # Store sender with combined strip info
                senders[ip] = {
                    'sender': sender,
                    'config': {
                        'ip': ip,
                        'strip_info': all_strip_info
                    }
                }
            except Exception as e:
                print(f"Error creating DMX sender for {ip}: {e}")
        
        return senders


    def concatenate_strips(self, new_id: str, strip_ids: List[str], join_group: str = None) -> None:
        """
        Concatenate multiple strips into a single logical strip.
        
        Args:
            new_id: ID for the new concatenated strip
            strip_ids: List of strip IDs to concatenate (in order)
            join_group: Optional group name that these strips belong to
        """
        if new_id in self.strips or new_id in self.concatenated_strips:
            raise ValueError(f"Strip ID '{new_id}' already exists")
            
        # Collect strips to concatenate
        strips_to_join = []
        for strip_id in strip_ids:
            if strip_id not in self.strips:
                raise KeyError(f"Strip '{strip_id}' not found")
            strips_to_join.append(self.strips[strip_id])
        
        if not strips_to_join:
            raise ValueError("No strips provided for concatenation")
            
        # Calculate total length
        total_length = sum(strip.length for strip in strips_to_join)
        
        # Create a new strip
        concatenated = LEDStrip(
            id=new_id,
            length=total_length,
            type="concatenated",  # Use a special type for concatenated strips
            groups=strips_to_join[0].groups.copy()  # Start with groups from first strip
        )
        
        # If a join group was specified, add it to the groups
        if join_group:
            if join_group not in concatenated.groups:
                concatenated.groups.append(join_group)
        
        # Create combined coordinates array
        concatenated.coordinates = np.zeros((total_length, 3))
        
        # Combine indices and coordinates
        concatenated.indices = np.arange(total_length)
        
        # Store the source strips and their offset in the concatenated strip
        concatenated.metadata["source_strips"] = []
        
        # Fill in coordinates and metadata
        offset = 0
        for strip in strips_to_join:
            # Copy coordinates
            concatenated.coordinates[offset:offset+strip.length] = strip.coordinates
            
            # Store mapping information
            concatenated.metadata["source_strips"].append({
                "id": strip.id,
                "offset": offset,
                "length": strip.length
            })
            
            # Update offset for next strip
            offset += strip.length
            
            # Merge groups (union)
            for group in strip.groups:
                if group not in concatenated.groups:
                    concatenated.groups.append(group)
        
        # Create distance array (normalized position along the strip)
        concatenated.distance = np.linspace(0, 1, total_length)
        
        # Store the concatenated strip
        self.concatenated_strips[new_id] = concatenated
        
        # Also add it to the regular strips collection for unified access
        self.strips[new_id] = concatenated
        
        # Invalidate caches
        self._invalidate_caches()

    def send_dmx(self, output_buffers, dmx_senders):
        """Send buffer data to DMX receivers, handling concatenated strips"""
        # Create a copy of the buffers to modify for concatenated strips
        processed_buffers = {}
        
        # For each strip buffer
        for strip_id, buffer in output_buffers.items():
            # If this is a regular strip, just copy the buffer
            if strip_id not in self.concatenated_strips:
                processed_buffers[strip_id] = buffer
            else:
                # For concatenated strips, we need to copy data to the source strips
                concatenated = self.concatenated_strips[strip_id]
                
                # For each source strip, copy the relevant portion of the buffer
                for source_info in concatenated.metadata["source_strips"]:
                    source_id = source_info["id"]
                    offset = source_info["offset"]
                    length = source_info["length"]
                    
                    # Create a buffer for this source strip if it doesn't exist
                    if source_id not in processed_buffers:
                        processed_buffers[source_id] = np.zeros((length, 4), dtype=np.float32)
                    
                    # Copy the data from the concatenated buffer to the source buffer
                    processed_buffers[source_id][:] = buffer[offset:offset+length]
        
        # Now send the processed buffers to DMX
        for ip, sender_info in dmx_senders.items():
            sender = sender_info['sender']
            config = sender_info['config']
            
            # Send to DMX using the processed buffers
            try:
                sender.send_from_buffers(processed_buffers, config['strip_info'])
            except Exception as e:
                print(f"Error sending DMX data to {ip}: {e}")


class StripLoader:
    """Utilities for loading strip definitions from files"""
    
    @staticmethod
    def from_json(file_path: str) -> StripManager:
        """Load strip definitions from a JSON file"""
        manager = StripManager()
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for strip_data in data.get('strips', []):
            manager.add_strip(StripLoader._create_strip_from_dict(strip_data))
        
        return manager
    
    @staticmethod
    def from_yaml(file_path: str) -> StripManager:
        """Load strip definitions from a YAML file"""
        manager = StripManager()
        
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        for strip_data in data.get('strips', []):
            manager.add_strip(StripLoader._create_strip_from_dict(strip_data))
        
        return manager
    
    @staticmethod
    def _create_strip_from_dict(data: Dict[str, Any]) -> LEDStrip:
        """Create a LEDStrip instance from a dictionary"""
        # Extract required fields
        strip_id = data['id']
        length = data['length']
        strip_type = data.get('type', 'line')  # Default to line if type not specified
        
        # Get position data
        position_data = data.get('position', {})
        
        # Create the strip with other metadata
        return LEDStrip(
            id=strip_id,
            length=length,
            type=strip_type,
            groups=data.get('groups', []),
            position_data=position_data,
            metadata=data.get('metadata', {})
        )


class BufferManager:
    """Manages multiple output buffers for different graphic generators"""
    
    def __init__(self, strip_manager: StripManager):
        self.strip_manager = strip_manager
        self.generators = {}  # Dictionary to store generator names and their buffers
        self.generator_alphas = {}
        self.generator_priorities = {}

    def register_generator(self, generator_name: str, priority: int = 0) -> None:
        """
        Register a new graphic generator and create its buffers
        
        Args:
            generator_name: Name of the generator
        """
        # Skip if generator already exists
        if generator_name in self.generators:
            # Update priority if generator already exists
            self.generator_priorities[generator_name] = priority
            return
            
        # Create buffers for this generator (one per strip)
        self.generators[generator_name] = {
            strip_id: np.zeros((strip.length, 4), dtype=np.float32)  # RGBA
            for strip_id, strip in self.strip_manager.strips.items()
        }
        
        # Store alpha value for this generator
        self.generator_alphas[generator_name] = 1
        self.generator_priorities[generator_name] = priority

    def get_buffer(self, generator_name: str, strip_id: str) -> np.ndarray:
        """Get a specific buffer for a generator and strip"""
        if generator_name not in self.generators:
            raise KeyError(f"Generator '{generator_name}' not registered")
        
        if strip_id not in self.generators[generator_name]:
            raise KeyError(f"Strip '{strip_id}' not found in generator '{generator_name}'")
        
        return self.generators[generator_name][strip_id]
    
    def get_all_buffers(self, generator_name: str) -> Dict[str, np.ndarray]:
        """Get all buffers for a specific generator"""
        if generator_name not in self.generators:
            raise KeyError(f"Generator '{generator_name}' not registered")
        
        return self.generators[generator_name]
    
    def clear_buffer(self, generator_name: str, strip_id: str = None) -> None:
        """Clear buffer(s) for a generator, optionally for a specific strip only"""
        if generator_name not in self.generators:
            raise KeyError(f"Generator '{generator_name}' not registered")
        
        if strip_id is not None:
            # Clear specific strip buffer
            if strip_id in self.generators[generator_name]:
                self.generators[generator_name][strip_id].fill(0)
        else:
            # Clear all buffers for this generator
            for buffer in self.generators[generator_name].values():
                buffer.fill(0)
    
    def merge_buffers(self, output_buffers: Dict[str, np.ndarray], blend_mode: str = 'alpha') -> None:
        """
        Merge all generator buffers into the output buffers using specified blend mode
        
        Args:
            output_buffers: Target buffers to merge into (typically the final output)
            blend_mode: How to combine pixels ('alpha', 'add', 'max', etc.)
        """
        # For each strip
        for strip_id in output_buffers:
            if strip_id not in self.strip_manager.strips:
                continue
                
            # Clear the output buffer
            output_buffers[strip_id].fill(0)

            # Sort generators by priority (lower to higher)
            sorted_generators = sorted(
                self.generators.items(), 
                key=lambda x: self.generator_priorities.get(x[0], 0)
            )
            

            # For each generator, blend its buffer into the output
            for generator_name, gen_buffers in sorted_generators:
                # Skip generators with zero alpha
                generator_alpha = self.generator_alphas.get(generator_name, 1.0)
                if generator_alpha <= 0.01:
                    continue
                    
                if strip_id not in gen_buffers:
                    continue
                    
                source_buffer = gen_buffers[strip_id]
                target_buffer = output_buffers[strip_id]
                
                if blend_mode == 'alpha':
                    # Vectorized alpha blending
                    # Calculate combined alpha values (source alpha * generator alpha)
                    alphas = source_buffer[:, 3] * generator_alpha
                    
                    # Create a mask for pixels with alpha > 0
                    mask = alphas > 0
                    
                    if np.any(mask):  # Only process if there are pixels to blend
                        # Calculate the blended colors for RGB channels
                        one_minus_alpha = 1 - alphas[mask, np.newaxis]
                        target_buffer[mask, :3] = one_minus_alpha * target_buffer[mask, :3] + \
                                                alphas[mask, np.newaxis] * source_buffer[mask, :3]
                        
                        # Update the alpha channel with the maximum value
                        target_buffer[mask, 3] = np.maximum(target_buffer[mask, 3], alphas[mask])
                            
                elif blend_mode == 'add':
                    # Additive blending with generator alpha
                    if generator_alpha < 1.0:
                        # Scale source by generator alpha
                        scaled_source = source_buffer * generator_alpha
                        target_buffer += scaled_source
                    else:
                        target_buffer += source_buffer
                    np.clip(target_buffer, 0, 1, out=target_buffer)
                    
                elif blend_mode == 'max':
                    # Maximum value blending with generator alpha
                    if generator_alpha < 1.0:
                        # Scale source by generator alpha
                        scaled_source = source_buffer * generator_alpha
                        np.maximum(target_buffer, scaled_source, out=target_buffer)
                    else:
                        np.maximum(target_buffer, source_buffer, out=target_buffer)


# Helper functions for testing
def hsv_to_rgb(h, s, v):
    """Simple HSV to RGB conversion"""
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    if h_i == 0:
        return v, t, p
    elif h_i == 1:
        return q, v, p
    elif h_i == 2:
        return p, v, t
    elif h_i == 3:
        return p, q, v
    elif h_i == 4:
        return t, p, v
    else:
        return v, p, q

def generate_rainbow(buffers):
    """Generate a rainbow pattern in the buffers"""
    for strip_id, buffer in buffers.items():
        # Fill buffer with rainbow pattern
        for i in range(len(buffer)):
            hue = (i / len(buffer)) % 1.0
            # Convert HSV to RGB
            r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
            buffer[i] = [r, g, b, 1.0]  # Full opacity