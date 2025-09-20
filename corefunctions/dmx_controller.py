#!/usr/bin/env python3
"""
DMX Controller Script for DSD Tech USB to DMX Interface
Supports basic DMX data transmission
"""

import serial
import time
import struct
import glob
import os
from typing import List, Optional

class DMXController:
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 250000):
        """
        Initialize DMX controller
        
        Args:
            port: Serial port (defaults to '/dev/ttyUSB0' for Linux)
            baudrate: Serial communication speed (typically 250000 for DMX)
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.dmx_data = [0] * 512  # DMX universe (512 channels)

        
    def connect(self, port: str = None) -> bool:
        """Connect to the DMX interface"""
        if port:
            self.port = port
            
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_TWO,
                timeout=1
            )
            print(f"Connected to DMX interface on {self.port}")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the DMX interface"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Disconnected from DMX interface")
    
    def set_channel(self, channel: int, value: int):
        """
        Set a single DMX channel value
        
        Args:
            channel: DMX channel (1-512)
            value: Channel value (0-255)
        """
        if 1 <= channel <= 512 and 0 <= value <= 255:
            self.dmx_data[channel - 1] = value
        else:
            raise ValueError("Channel must be 1-512, value must be 0-255")
    
    def set_channels(self, start_channel: int, values: List[int]):
        """
        Set multiple consecutive DMX channels
        
        Args:
            start_channel: Starting channel (1-512)
            values: List of values (0-255 each)
        """
        for i, value in enumerate(values):
            if start_channel + i <= 512:
                self.set_channel(start_channel + i, value)
    
    def send_dmx_packet(self) -> bool:
        """Send DMX data packet using Enttec DMX USB Pro protocol"""
        if not self.serial_conn or not self.serial_conn.is_open:
            print("Not connected to DMX interface")
            return False
        
        try:
            # Enttec DMX USB Pro packet format
            packet = bytearray([])
            
            # Start delimiter
            packet.append(0x7E)
            
            # Label (6 = Send DMX packet)
            packet.append(0x06)
            
            # Data length (LSB, MSB) - 513 bytes (1 start code + 512 data)
            data_length = 513
            packet.extend(struct.pack('<H', data_length))
            
            # DMX start code (0 for standard DMX)
            packet.append(0x00)
            
            # DMX data (512 channels)
            packet.extend(self.dmx_data)
            
            # End delimiter
            packet.append(0xE7)
            
        
            # Send packet
            bytes_written = self.serial_conn.write(packet)
            self.serial_conn.flush()  # Ensure data is sent immediately
            
            
        except Exception as e:
            print(f"Error sending DMX packet: {e}")
            return False


    
    def send_continuous(self, refresh_rate: float = 30.0):
        """
        Send DMX data continuously at specified refresh rate
        
        Args:
            refresh_rate: Updates per second (typically 30-44 Hz for DMX)
        """
        interval = 1.0 / refresh_rate
        print(f"Starting continuous DMX transmission at {refresh_rate} Hz")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                self.send_dmx_packet()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopping continuous transmission")

def find_dmx_ports() -> List[str]:
    """Find available serial ports that might be DMX interfaces"""
    ports = []
    
    # Check common Linux serial device patterns
    possible_patterns = [
        '/dev/ttyUSB*',
        '/dev/ttyACM*', 
        '/dev/ttyS*'
    ]
    
    for pattern in possible_patterns:
        ports.extend(glob.glob(pattern))
    
    # Also try using pyserial's port detection
    try:
        import serial.tools.list_ports
        for port in serial.tools.list_ports.comports():
            device = port.device
            if device not in ports:
                ports.append(device)
            print(f"Found port: {device}")
            print(f"  Description: {port.description}")
            print(f"  Hardware ID: {port.hwid}")
            print(f"  VID:PID: {port.vid:04x}:{port.pid:04x}" if port.vid else "  VID:PID: Unknown")
            print()
    except ImportError:
        print("pyserial.tools not available, using glob patterns only")
    
    return sorted(set(ports))

def check_ftdi_devices():
    """Check for FTDI devices using system information"""
    print("Checking for FTDI devices...")
    
    # Check lsusb output for FTDI devices
    try:
        import subprocess
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        ftdi_lines = [line for line in result.stdout.split('\n') if '0403:6001' in line or 'FTDI' in line]
        for line in ftdi_lines:
            print(f"FTDI device found: {line}")
    except:
        print("Could not run lsusb")
    
    # Check /sys/bus/usb-serial/devices/
    usb_serial_path = "/sys/bus/usb-serial/devices/"
    if os.path.exists(usb_serial_path):
        devices = os.listdir(usb_serial_path)
        print(f"USB serial devices in {usb_serial_path}: {devices}")
    
    # Check dmesg for recent FTDI connections
    try:
        import subprocess
        result = subprocess.run(['dmesg', '|', 'tail', '-20'], capture_output=True, text=True, shell=True)
        ftdi_lines = [line for line in result.stdout.split('\n') if 'FTDI' in line or 'ttyUSB' in line]
        if ftdi_lines:
            print("Recent FTDI/USB serial messages:")
            for line in ftdi_lines[-5:]:  # Last 5 relevant lines
                print(f"  {line}")
    except:
        print("Could not check dmesg")

# Example usage and testing
if __name__ == "__main__":
    print("DMX Controller for DSD Tech USB-DMX Interface")
    print("=" * 50)
    
    # Check for FTDI devices
    check_ftdi_devices()
    print()
    
    # Find available ports
    available_ports = find_dmx_ports()
    print("Available serial ports:", available_ports)
    print()
    
    # Initialize controller with default /dev/ttyUSB0
    dmx = DMXController()
    
    # Check if default port exists, otherwise prompt for selection
    if "/dev/ttyUSB0" in available_ports or os.path.exists("/dev/ttyUSB0"):
        port = "/dev/ttyUSB0"
        print(f"Using default port: {port}")
    elif available_ports:
        print("Default /dev/ttyUSB0 not found. Available ports:")
        for i, p in enumerate(available_ports):
            print(f"  {i}: {p}")
        try:
            choice = int(input("Select port number: "))
            port = available_ports[choice]
        except (ValueError, IndexError):
            port = available_ports[0]
            print(f"Invalid choice, using: {port}")
    else:
        print("No serial ports found!")
        print("Make sure your DSD Tech USB-DMX interface is connected.")
        print("You might need to run: sudo chmod 666 /dev/ttyUSB* or add your user to the dialout group")
        print("To add user to dialout group: sudo usermod -a -G dialout $USER (then logout/login)")
        exit(1)
    
    if dmx.connect(port):
        try:
            #dmx.send_continuous()
            n=5
            m=100
            # White
            for n in range(11):
                print([n,m])
                dmx.set_channel(n+1, m)
                dmx.send_dmx_packet()
                time.sleep(5)
            

            
            print("Test complete!")
            
        except Exception as e:
            print(f"Error during testing: {e}")
        finally:
            dmx.disconnect()
    else:
        print("Failed to connect to DMX interface")
        print("\nTroubleshooting tips:")
        print("1. Check if the device is properly connected")
        print("2. Verify permissions: ls -l /dev/ttyUSB*")
        print("3. Add user to dialout group: sudo usermod -a -G dialout $USER")
        print("4. Try running with sudo (not recommended for regular use)")