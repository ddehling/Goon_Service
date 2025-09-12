import threading
import time
import sounddevice as sd
from corefunctions.soundinput import MicrophoneAnalyzer

class AudioAnalysisController:
    def __init__(self, device_id=None, sample_rate=48000, block_size=1024, channels=1):
        """
        Initialize audio analysis controller
        
        Args:
            device_id (int): Specific device ID to use, None for auto-detection
            sample_rate (int): Sample rate for audio capture
            block_size (int): Block size for audio processing
            channels (int): Number of audio channels
        """
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channels = channels
        self.analyzer = None
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        self.is_monitoring = False
        
        # Auto-detect device if not specified
        if self.device_id is None:
            self.device_id = self.find_working_audio_device()
    
    @staticmethod
    def test_device_access(device_id, sample_rate=48000, block_size=1024):
        """
        Test if a device can actually be opened for recording
        
        Args:
            device_id (int): Device ID to test
            sample_rate (int): Sample rate to test with
            block_size (int): Block size to test with
            
        Returns:
            bool: True if device works, False otherwise
        """
        try:
            test_stream = sd.InputStream(
                device=device_id,
                channels=1,
                samplerate=sample_rate,
                blocksize=block_size
            )
            test_stream.start()
            test_stream.stop()
            test_stream.close()
            return True
        except Exception as e:
            return False
    
    @classmethod
    def find_working_audio_device(cls):
        """
        Find available and working audio input device
        
        Returns:
            int or None: Device ID if found, None if no working device found
        """
        devices = sd.query_devices()
        
        print("\nSearching for audio input device...")
        print("Available input devices:")
        
        working_candidates = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  Device ID {i}: {device['name']}")
                
                # Test if device actually works
                if cls.test_device_access(i):
                    status = "✓ Working"
                    working_candidates.append((i, device))
                    print(f"    {status}")
                else:
                    status = "✗ Not accessible"
                    print(f"    {status}")
        
        # Use first working device
        if working_candidates:
            device_id, device_info = working_candidates[0]
            print(f"\nSelected: {device_info['name']} (ID: {device_id})")
            return device_id
        
        print("\n" + "="*60)
        print("NO WORKING AUDIO INPUT DEVICE FOUND")
        print("="*60)
        print("\nTo use audio monitoring on Raspberry Pi:")
        print("\n1. Connect USB microphone or audio interface")
        print("2. Enable audio: sudo raspi-config → Advanced Options → Audio")
        print("3. Check ALSA devices: arecord -l")
        print("4. Install audio packages: sudo apt install alsa-utils pulseaudio")
        
        return None
    
    @staticmethod
    def list_audio_devices():
        """
        List all available audio devices
        
        Returns:
            list: List of device information dictionaries
        """
        devices = sd.query_devices()
        device_list = []
        
        print("\nAll available audio devices:")
        for i, device in enumerate(devices):
            device_info = {
                'id': i,
                'name': device['name'],
                'max_input_channels': device['max_input_channels'],
                'max_output_channels': device['max_output_channels'],
                'default_samplerate': device['default_samplerate'],
                'working': AudioAnalysisController.test_device_access(i) if device['max_input_channels'] > 0 else False
            }
            device_list.append(device_info)
            
            device_type = "Input" if device['max_input_channels'] > 0 else "Output"
            working_status = "✓" if device_info['working'] else "✗" if device['max_input_channels'] > 0 else "-"
            
            print(f"  ID {i}: {device['name']} [{device_type}] {working_status}")
        
        return device_list
    
    def initialize_analyzer(self):
        """
        Initialize the microphone analyzer
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.device_id is None:
            print("No working audio device found. Cannot initialize analyzer.")
            return False
        
        try:
            print(f"Initializing audio analyzer with device ID {self.device_id}...")
            self.analyzer = MicrophoneAnalyzer(device=self.device_id)
            self.analyzer.start()
            print("Audio analyzer initialized successfully!")
            return True
        except Exception as e:
            print(f"Failed to initialize audio analyzer: {e}")
            self.analyzer = None
            return False
    
    def get_bass_level(self):
        """
        Get current bass frequency level
        
        Returns:
            float or None: Bass level, None if analyzer not running
        """
        if self.analyzer is None:
            return None
        
        try:
            return self.analyzer.get_sound()
        except Exception as e:
            print(f"Error getting bass level: {e}")
            return None
    
    def get_overall_level(self):
        """
        Get current overall sound level
        
        Returns:
            float or None: Overall level, None if analyzer not running
        """
        if self.analyzer is None:
            return None
        
        try:
            return self.analyzer.get_all_sound()
        except Exception as e:
            print(f"Error getting overall level: {e}")
            return None
    
    def get_audio_levels(self):
        """
        Get both bass and overall audio levels
        
        Returns:
            dict: Dictionary with 'bass' and 'overall' keys, or None if error
        """
        if self.analyzer is None:
            return None
        
        try:
            bass_level = self.analyzer.get_sound()
            overall_level = self.analyzer.get_all_sound()
            return {
                'bass': bass_level,
                'overall': overall_level
            }
        except Exception as e:
            print(f"Error getting audio levels: {e}")
            return None
    
    def _monitor_audio_levels(self, interval=1.0, callback=None):
        """
        Internal method to monitor and print audio levels
        
        Args:
            interval (float): Seconds between level checks
            callback (callable): Optional callback function to handle levels
        """
        print("Starting audio monitoring...")
        
        while not self.stop_event.is_set():
            try:
                levels = self.get_audio_levels()
                if levels:
                    if callback:
                        callback(levels['bass'], levels['overall'])
                    else:
                        print(f"Audio Level - Bass: {levels['bass']:.3f}, Overall: {levels['overall']:.3f}")
                else:
                    print("Could not read audio levels")
            except Exception as e:
                print(f"Audio monitoring error: {e}")
            
            # Wait for interval or until stop event is set
            if self.stop_event.wait(interval):
                break
        
        print("Audio monitoring stopped.")
    
    def start_monitoring(self, interval=1.0, callback=None):
        """
        Start continuous audio level monitoring in a separate thread
        
        Args:
            interval (float): Seconds between level checks
            callback (callable): Optional callback function(bass_level, overall_level)
            
        Returns:
            bool: True if monitoring started, False otherwise
        """
        if self.analyzer is None:
            if not self.initialize_analyzer():
                return False
        
        if self.is_monitoring:
            print("Audio monitoring already running")
            return True
        
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitor_audio_levels,
            args=(interval, callback)
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        self.is_monitoring = True
        
        return True
    
    def stop_monitoring(self, timeout=2):
        """
        Stop audio level monitoring
        
        Args:
            timeout (float): Timeout in seconds to wait for thread to stop
            
        Returns:
            bool: True if stopped successfully
        """
        if not self.is_monitoring:
            return True
        
        print("Stopping audio monitoring...")
        self.stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=timeout)
            self.monitoring_thread = None
        
        self.is_monitoring = False
        return True
    
    def cleanup(self):
        """
        Clean up resources and stop monitoring
        
        Returns:
            bool: True if cleanup successful
        """
        success = True
        
        # Stop monitoring first
        if self.is_monitoring:
            success &= self.stop_monitoring()
        
        # Stop analyzer
        if self.analyzer:
            try:
                self.analyzer.stop()
                self.analyzer = None
                print("Audio analyzer stopped.")
            except Exception as e:
                print(f"Error stopping audio analyzer: {e}")
                success = False
        
        return success
    
    def __enter__(self):
        """Context manager entry"""
        if not self.initialize_analyzer():
            raise RuntimeError("Failed to initialize audio analyzer")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()

# Example usage and testing
if __name__ == "__main__":
    print("Testing Audio Analysis Controller...")
    
    # List all devices
    AudioAnalysisController.list_audio_devices()
    
    # Test with context manager
    try:
        with AudioAnalysisController() as audio:
            print("\nStarting 10-second audio monitoring test...")
            
            # Start monitoring with custom callback
            def level_callback(bass, overall):
                print(f"Custom callback - Bass: {bass:.3f}, Overall: {overall:.3f}")
            
            audio.start_monitoring(interval=0.5, callback=level_callback)
            
            # Let it run for 10 seconds
            time.sleep(10)
            
            print("Test complete!")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")