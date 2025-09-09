import numpy as np
from sacn import sACNsender
import math


class SACNPixelSender:
    def __init__(self, receivers, start_universe=1):
        """
        Initialize the SACNPixelSender with receiver configurations.
        :param receivers: List of dicts, each with 'ip', 'pixel_count', and 'addressing_array' keys.
        """
        self.receivers = receivers
        self.sender = sACNsender()
        self.sender.start()

        # Set up universes for each receiver
        self.receiver_universes = []
        self.receiver_to_output_map = []  # Map each receiver to its output group
        universe_counter = start_universe
        
        # Sort receivers by strip number (or any other specified order)
        sorted_receivers = sorted(receivers, key=lambda r: r.get('strip_num_start', float('inf')))
        
        # Process all receivers in order
        for receiver in sorted_receivers:
            receiver_type = receiver.get('type', 'RGB')
            output = receiver.get('output', None)  # Get output group for any type
            
            if receiver_type == 'RGB':
                # Each RGB universe can hold 170 pixels (510 bytes)
                universe_count = math.ceil(receiver['pixel_count'] / 170)
                print(f"Setting up RGB universes for strips {receiver.get('strip_ids', [])} starting at universe {universe_counter}")
                receiver_universes = list(range(universe_counter, universe_counter + universe_count))
                self.receiver_universes.append(receiver_universes)
                self.receiver_to_output_map.append(output)  # Support output grouping for RGB
                universe_counter += universe_count

            elif receiver_type == 'RGBW':
                # Each RGBW universe can hold 128 pixels (512 bytes)
                universe_count = math.ceil(receiver['pixel_count'] / 128)
                print(f"Setting up RGBW universes for strips {receiver.get('strip_ids', [])} starting at universe {universe_counter}")
                receiver_universes = list(range(universe_counter, universe_counter + universe_count))
                self.receiver_universes.append(receiver_universes)
                self.receiver_to_output_map.append(output)  # Support output grouping for RGBW
                universe_counter += universe_count
                
            elif receiver_type == 'RGBW3':
                # RGBW3 packs RGBW data continuously across universes (510 bytes per universe)
                print(f"Setting up RGBW3 universes for output {output} strips {receiver.get('strip_ids', [])} starting at universe {universe_counter}")
                
                # Calculate total bytes needed for this RGBW3 receiver
                receiver_bytes = receiver['pixel_count'] * 4  # 4 bytes per RGBW pixel
                universe_count = math.ceil(receiver_bytes / 510)  # 510 bytes per universe
                
                receiver_universes = list(range(universe_counter, universe_counter + universe_count))
                self.receiver_universes.append(receiver_universes)
                self.receiver_to_output_map.append(output)  # Store output group
                universe_counter += universe_count

            elif receiver_type == 'DMX':
                # Each DMX universe can hold 73 pixels (511 bytes, leaving 1 byte unused)
                universe_count = math.ceil(receiver['pixel_count'] / 73)
                print(f"Setting up DMX universes for strips {receiver.get('strip_ids', [])} starting at universe {universe_counter}")
                receiver_universes = list(range(universe_counter, universe_counter + universe_count))
                self.receiver_universes.append(receiver_universes)
                self.receiver_to_output_map.append(output)  # Support output grouping for DMX
                universe_counter += universe_count
                
            elif receiver_type == 'RGB4':
                print(f"Setting up RGB4 universes for output {output} strips {receiver.get('strip_ids', [])} starting at universe {universe_counter}")
                
                # Calculate total bytes needed for this RGB4 receiver
                receiver_bytes = receiver['pixel_count'] * 3
                universe_count = math.ceil(receiver_bytes / 510)  # 510 bytes per universe
                
                receiver_universes = list(range(universe_counter, universe_counter + universe_count))
                self.receiver_universes.append(receiver_universes)
                self.receiver_to_output_map.append(output)  # Store output group
                universe_counter += universe_count
            
            # Activate universes for this receiver
            for universe in receiver_universes:
                self.sender.activate_output(universe)
                self.sender[universe].destination = receiver['ip']


        
        # # Process RGB4 strips - group by output
        # rgb4_start_universe = universe_counter
        # rgb4_receivers = [r for r in receivers if r.get('type', 'RGB4') == 'RGB4']
        
        # if rgb4_receivers:
        #     # Group RGB4 receivers by output
        #     output_groups = {}
        #     for receiver in rgb4_receivers:
        #         output = receiver.get('output', 'default')
        #         if output not in output_groups:
        #             output_groups[output] = []
        #         output_groups[output].append(receiver)
            
        #     # Process each output group
        #     for output, group_receivers in output_groups.items():
        #         print(f"Setting up RGB4 universes for output {output} starting at universe {universe_counter}")
                
        #         # Calculate total bytes needed for this output group
        #         total_bytes = sum(receiver['pixel_count'] * 3 for receiver in group_receivers)
        #         universe_count = math.ceil(total_bytes / 510)  # 510 bytes per universe (170 pixels * 3)
                
        #         # Assign universes to each receiver in this output group
        #         bytes_used = 0
        #         for receiver in group_receivers:
        #             receiver_bytes = receiver['pixel_count'] * 3
                    
        #             # Calculate which universes this receiver spans
        #             start_universe = universe_counter + (bytes_used // 510)
        #             end_universe = universe_counter + ((bytes_used + receiver_bytes - 1) // 510)
                    
        #             receiver_universes = list(range(start_universe, end_universe + 1))
        #             self.receiver_universes.append(receiver_universes)
        #             self.receiver_to_output_map.append(output)  # Store output group
                    
        #             # Activate universes for this receiver
        #             for universe in receiver_universes:
        #                 if universe not in self.sender.get_active_outputs():
        #                     self.sender.activate_output(universe)
        #                     self.sender[universe].destination = receiver['ip']
                    
        #             bytes_used += receiver_bytes
                
        #         # Move universe counter to next available universe
        #         universe_counter += universe_count
                
    def create_mask(self, height, width):
        """
        Creates a binary mask showing which pixels are mapped by receivers.
        
        :param height: Height of the source image
        :param width: Width of the source image
        :return: numpy array of shape (height, width) with 1s where pixels are mapped
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for receiver in self.receivers:
            # Clip coordinates to valid image boundaries
            x_coords = np.clip(receiver['addressing_array'][:, 0], 0, height - 1)
            y_coords = np.clip(receiver['addressing_array'][:, 1], 0, width - 1)
            
            # Set mapped pixels to 1
            mask[x_coords, y_coords] = 1
            
        return mask

    def send(self, source_array):
        """
        Send pixel data to all configured receivers based on their addressing arrays.
        :param source_array: numpy array of shape (strips, pixels, 3) containing source pixel data.
        """
        for receiver, universes in zip(self.receivers, self.receiver_universes):
            # Vectorized extraction of pixel data
            x_coords = np.clip(receiver['addressing_array'][:, 0], 0, source_array.shape[0] - 1)
            y_coords = np.clip(receiver['addressing_array'][:, 1], 0, source_array.shape[1] - 1)
            
            # Extract the RGB values for each pixel
            receiver_data = source_array[x_coords, y_coords]

            # Send data in 170-pixel chunks
            for i, universe in enumerate(universes):
                start = i * 170
                end = min(start + 170, receiver['pixel_count'])
                universe_data = receiver_data[start:end].flatten()
                # Pad the last universe if necessary
                if universe_data.size < 510:
                    universe_data = np.pad(universe_data, (0, 510 - universe_data.size), 'constant')
                self.sender[universe].dmx_data = universe_data.tobytes()

    def send_from_buffers(self, output_buffers, strip_info):
        """
        Send pixel data directly from output buffers.
        
        :param output_buffers: Dictionary mapping strip_id to buffer arrays
        :param strip_info: List of tuples (strip_id, length, direction, type)
        """
        gamma = 2
        
        # Group receivers by output for types that support it
        output_grouped_receivers = {}
        output_grouped_indices = []
        
        # Identify receivers that use output grouping
        for i, (receiver, output_group) in enumerate(zip(self.receivers, self.receiver_to_output_map)):
            receiver_type = receiver.get('type')
            # Types that support continuous packing across universes
            if receiver_type in ['RGB4', 'RGBW3'] and output_group is not None:
                if output_group not in output_grouped_receivers:
                    output_grouped_receivers[output_group] = {'RGB4': [], 'RGBW3': []}
                output_grouped_receivers[output_group][receiver_type].append(i)
                output_grouped_indices.append(i)
        
        # Process normal receivers (RGB, RGBW, DMX) that don't use output grouping
        for i, (receiver, universes) in enumerate(zip(self.receivers, self.receiver_universes)):
            # Skip output-grouped receivers
            if i in output_grouped_indices:
                continue
                
            # Get the receiver type
            receiver_type = receiver.get('type', 'RGB')
            
            # Concatenate all strip data for this receiver
            all_pixel_data = []
            
            for strip_id, length, direction, strip_type in strip_info:
                # Skip if strip type doesn't match receiver type
                if strip_type != receiver_type:
                    continue
                    
                # Get the buffer for this strip
                buffer = output_buffers[strip_id]
                
                # Convert from float (0-1) to uint8 (0-255)
                rgb_data = (np.power(buffer[:, :3], gamma) * 255).astype(np.uint8)
                
                # Reverse the strip if direction is -1
                if direction == -1:
                    rgb_data = rgb_data[::-1]
                    
                # Add to the concatenated data
                all_pixel_data.append(rgb_data)
                
            if not all_pixel_data:
                continue  # Skip if no strips match this receiver
                
            # Concatenate all strips
            receiver_data = np.concatenate(all_pixel_data)
            
            # Handle different pixel types
            if receiver_type == 'RGB':
                pixels_per_universe = 170
                self._send_rgb_data(receiver_data, universes, pixels_per_universe)
            elif receiver_type == 'RGBW':
                pixels_per_universe = 128
                self._send_rgbw_data(receiver_data, universes, pixels_per_universe)
            elif receiver_type == 'DMX':
                pixels_per_universe = 73
                self._send_dmx_data(receiver_data, universes, pixels_per_universe)
        
        # Process output-grouped receivers
        for output, type_indices in output_grouped_receivers.items():
            # Process RGB4 strips for this output
            if type_indices['RGB4']:
                self._send_rgb4_output_group(output, type_indices['RGB4'], output_buffers, strip_info, gamma)
            
            # Process RGBW3 strips for this output
            if type_indices['RGBW3']:
                self._send_rgbw3_output_group(output, type_indices['RGBW3'], output_buffers, strip_info, gamma)
    
    def _send_rgb4_output_group(self, output, indices, output_buffers, strip_info, gamma):
        """Send RGB4 data for an output group"""
        # Collect all RGB4 data for this output group
        output_data = []
        
        for receiver_idx in indices:
            receiver = self.receivers[receiver_idx]
            
            # Get all strips for this receiver
            receiver_strips = []
            for strip_id, length, direction, strip_type in strip_info:
                if strip_type == 'RGB4' and strip_id in receiver.get('strip_ids', []):
                    buffer = output_buffers[strip_id]
                    rgb_data = (np.power(buffer[:, :3], gamma) * 255).astype(np.uint8)
                    
                    if direction == -1:
                        rgb_data = rgb_data[::-1]
                        
                    receiver_strips.append(rgb_data)
            
            if receiver_strips:
                output_data.append(np.concatenate(receiver_strips))
        
        if not output_data:
            return
            
        # Concatenate all data for this output group
        all_output_data = np.concatenate(output_data)
        
        # Flatten the RGB data
        all_bytes = all_output_data.flatten()
        
        # Send across universes (up to 510 bytes per universe)
        universes = []
        for receiver_idx in indices:
            universes.extend(self.receiver_universes[receiver_idx])
        universes = sorted(set(universes))  # Unique, sorted universes
        
        for i, universe in enumerate(universes):
            start_byte = i * 510
            end_byte = min(start_byte + 510, len(all_bytes))
            
            if start_byte < len(all_bytes):
                # Get bytes for this universe
                universe_bytes = all_bytes[start_byte:end_byte]
                
                # Pad if needed
                if len(universe_bytes) < 510:
                    universe_bytes = np.pad(universe_bytes, (0, 510 - len(universe_bytes)), 'constant')
                
                # Send the data
                self.sender[universe].dmx_data = universe_bytes.tobytes()
    

    def _send_rgbw3_output_group(self, output, indices, output_buffers, strip_info, gamma):
        """Send RGBW3 data for an output group (RGBW data packed continuously)"""
        # Collect all RGBW3 data for this output group
        output_data = []
        
        for receiver_idx in indices:
            receiver = self.receivers[receiver_idx]
            
            # Get all strips for this receiver
            receiver_strips = []
            for strip_id, length, direction, strip_type in strip_info:
                if strip_type == 'RGBW3' and strip_id in receiver.get('strip_ids', []):
                    buffer = output_buffers[strip_id]
                    rgb_data = (np.power(buffer[:, :3], gamma) * 255).astype(np.uint8)
                    
                    if direction == -1:
                        rgb_data = rgb_data[::-1]
                    
                    # Create RGBW data with W set to 0
                    rgbw_data = np.zeros((rgb_data.shape[0], 4), dtype=np.uint8)
                    rgbw_data[:, [1,0,2]] = rgb_data  # Set RGB components (with channel swap)
                    # W channel remains 0
                    
                    receiver_strips.append(rgbw_data)
            
            if receiver_strips:
                output_data.append(np.concatenate(receiver_strips))
        
        if not output_data:
            return
            
        # Concatenate all data for this output group
        all_output_data = np.concatenate(output_data)
        
        # Flatten the RGBW data
        all_bytes = all_output_data.flatten()
        
        # Send across universes (up to 510 bytes per universe)
        universes = []
        for receiver_idx in indices:
            universes.extend(self.receiver_universes[receiver_idx])
        universes = sorted(set(universes))  # Unique, sorted universes
        
        for i, universe in enumerate(universes):
            start_byte = i * 510
            end_byte = min(start_byte + 510, len(all_bytes))
            
            if start_byte < len(all_bytes):
                # Get bytes for this universe
                universe_bytes = all_bytes[start_byte:end_byte]
                
                # Pad if needed (last 2 bytes to 0 to reach 512)
                if len(universe_bytes) < 510:
                    universe_bytes = np.pad(universe_bytes, (0, 510 - len(universe_bytes)), 'constant')
                
                # Send the data
                self.sender[universe].dmx_data = universe_bytes.tobytes()



    def _send_rgb_data(self, receiver_data, universes, pixels_per_universe):
        """Send RGB data (helper method)"""
        for i, universe in enumerate(universes):
            start = i * pixels_per_universe
            end = min(start + pixels_per_universe, len(receiver_data))
            
            if start < len(receiver_data):
                # Get data for this universe
                universe_data = receiver_data[start:end].flatten()
                
                # Pad if necessary
                if universe_data.size < 510:  # 510 = 170 pixels * 3 bytes
                    universe_data = np.pad(universe_data, (0, 510 - universe_data.size), 'constant')
                
                # Send the data
                self.sender[universe].dmx_data = universe_data.tobytes()

    def _send_rgbw_data(self, receiver_data, universes, pixels_per_universe):
        """Send RGBW data (helper method)"""
        for i, universe in enumerate(universes):
            start = i * pixels_per_universe
            end = min(start + pixels_per_universe, len(receiver_data))
            
            if start < len(receiver_data):
                # Get data for this universe
                universe_data = receiver_data[start:end]
                
                # Create RGBW data with W set to 0
                rgbw_data = np.zeros((universe_data.shape[0], 4), dtype=np.uint8)
                #rgbw_data[:, :3] = universe_data  # Set RGB components
                rgbw_data[:, [1,0,2]] = universe_data  # Set RGB components
                # Flatten the RGBW data
                universe_data = rgbw_data.flatten()
                
                # Pad if necessary
                if universe_data.size < 512:  # 512 = 128 pixels * 4 bytes
                    universe_data = np.pad(universe_data, (0, 512 - universe_data.size), 'constant')
                
                # Send the data
                self.sender[universe].dmx_data = universe_data.tobytes()

    def _send_dmx_data(self, receiver_data, universes, pixels_per_universe):
        """Send DMX data (helper method)"""
        for i, universe in enumerate(universes):
            start = i * pixels_per_universe
            end = min(start + pixels_per_universe, len(receiver_data))
            
            if start < len(receiver_data):
                # Get data for this universe
                universe_data = receiver_data[start:end]
                
                # Create DMX data: [255, R, G, B, 0, 0, 0] for each pixel
                dmx_data = np.zeros((universe_data.shape[0], 7), dtype=np.uint8)
                dmx_data[:, 0] = 255  # First byte is always 255
                dmx_data[:, 1:4] = universe_data  # Set RGB components
                # Bytes 4-6 are already 0
                
                # Flatten the DMX data
                universe_data = dmx_data.flatten()
                
                # Pad if necessary
                if universe_data.size < 512:
                    universe_data = np.pad(universe_data, (0, 512 - universe_data.size), 'constant')
                
                # Send the data
                self.sender[universe].dmx_data = universe_data.tobytes()

    def close(self):
        """
        Properly close the sACN sender
        """
        self.sender.stop()

    def analyze_row_groups(self, max_pixels_per_group=170):
        """
        Analyze and group pixels in rows that belong to the same receiver.
        
        :param max_pixels_per_group: Maximum number of pixels per group (default 170 for sACN universe limit)
        :return: Dictionary mapping receiver indices to lists of pixel groups
        """
        receiver_groups = {}
        
        for idx, receiver in enumerate(self.receivers):
            # Get coordinates from addressing array
            coordinates = receiver['addressing_array']
            #find the unique rows
            rows = np.unique(coordinates[:, 0])
            #find the number of pixels in each row
            row_counts = {row: np.sum(coordinates[:, 0] == row) for row in rows}         
            #initialize the group list
            groups = []
            #initialize the current group
            current_group = []
            #initialize the current row
            current_row = None
            #iterate through the rows
            group_pixel_count = 0
            pixels_in_group = []
            for row in rows:
                if current_row is None:
                    current_row = row
                #if the row is different from the current row
                row_count = row_counts[row]
                if (group_pixel_count + row_count) <= max_pixels_per_group:
                    #store the current group
                    current_group.append(row)
                    #reset the group count
                    group_pixel_count += row_count
                    #reset the current group
                else:
                    groups.append(current_group)
                    current_group = [row]
                    pixels_in_group.append(group_pixel_count)
                    group_pixel_count = row_count
            #handle the last group
            groups.append(current_group)
            pixels_in_group.append(group_pixel_count)
            print(groups, pixels_in_group,receiver['ip'])

# The rest of the code (generate_frame_data and main function) remains the same

def generate_frame_data():
    """
    Generate random RGB pixel data for each frame.
    In a real application, this would pull data from your actual source.
    """
    width, height = 100, 100  # Example dimensions
    return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)

def make_indicesHS(filename):
    in_list=np.loadtxt(filename, delimiter=',').tolist()
    indices = []
    for sublist in in_list:       
        if sublist[2]>0:
            for m in range(int(sublist[2])):
                indices.append([sublist[0], m+sublist[1]])
        else:
            for m in range(int(-sublist[2])):
                indices.append([sublist[0], sublist[1]-sublist[2]-1-m])   
    return np.array(indices).astype(int)

def main():
    receivers = [
        {
            'ip': '192.168.68.121',
            'pixel_count': 3500,
            #'addressing_array':make_indicesH()
            'addressing_array':make_indicesHS(r"data.txt")
        }
    ]

    sender = SACNPixelSender(receivers)
    sender.analyze_row_groups(255)
    sender.close()

if __name__ == "__main__":
    main()