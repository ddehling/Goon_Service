from dmxpy import DmxPy
import time

# Initialize DmxPy with the serial port of your DMX device.
# Replace 'COM5' with the actual serial port your device is connected to.
# On Linux, this might be something like '/dev/ttyUSB0'.
dmx = DmxPy.DmxPy('/dev/ttyUSB0')
dmx.set_channel(3,255)
# Set a specific DMX channel to a value.


# To send the DMX values to the device, you must call render().
dmx.render()
time.sleep(2)
# You can also set all channels to 0 (off)
# dmx.all_off()
# dmx.render()