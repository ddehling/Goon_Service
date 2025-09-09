# osc_test_sender.py
from pythonosc import udp_client
import time
import random
import argparse

def main():
    parser = argparse.ArgumentParser(description='Send test OSC messages to Events.py')
    parser.add_argument('--ip', default='127.0.0.1', help='The IP to send OSC messages to')
    parser.add_argument('--port', type=int, default=5005, help='The port to send OSC messages to')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between messages in seconds')
    args = parser.parse_args()

    # Create OSC client
    client = udp_client.SimpleUDPClient(args.ip, args.port)
    
    print(f"Sending OSC test messages to {args.ip}:{args.port}")
    print("Press Ctrl+C to stop")
    
    message_types = [
        ("/emotions/brightness", lambda: random.randint(0, 10)),
        #("/test/color", lambda: [random.random(), random.random(), random.random()]),
        #("/test/trigger", lambda: "effect_" + str(random.randint(1, 5))),
        ("/state/tempo", lambda: random.uniform(0, 1.0)),
    ]
    
    try:
        count = 0
        while True:
            # Choose a random message type
            address, value_generator = random.choice(message_types)
            value = value_generator()
            
            # Send the message
            client.send_message(address, value)
            
            #print(f"Sent: {address} {value}")
            count += 1
            
            # Every 10 messages, send a batch of messages rapidly
            if count % 10 == 0:
                print("Sending message burst...")
                for _ in range(5):
                    addr, val_gen = random.choice(message_types)
                    client.send_message(addr, val_gen())
                    time.sleep(0.05)
            
            time.sleep(args.delay)
            
    except KeyboardInterrupt:
        print("\nStopping OSC test sender")

if __name__ == "__main__":
    main()