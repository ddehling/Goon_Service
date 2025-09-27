import colorsys

def hex_to_rgb_hsv(hex_color):
    """
    Convert a 6-digit hex color code to RGB and HSV values (all 0-1)
    
    Args:
        hex_color (str): Hex color code (with or without #)
    
    Returns:
        dict: Contains 'rgb' tuple (0-1) and 'hsv' tuple (all 0-1)
    """
    # Remove # if present
    hex_color = hex_color.lstrip('#')
    
    # Validate input
    if len(hex_color) != 6:
        raise ValueError("Hex color must be 6 digits")
    
    # Convert hex to RGB (0-255) then to 0-1 range
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    
    # Convert RGB to HSV (colorsys returns h as 0-1, which is what we want)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    return {
        'rgb': (r, g, b),
        'hsv': (h, s, v)
    }

# Example usage
if __name__ == "__main__":
    # Test with various hex colors
    colors = ["C27E35","757008","1E3226","0E568F","A23742","762908"]
    print("Hex Color -> RGB (0-1) -> HSV (all 0-1)")
    print("-" * 50)
    
    for color in colors:
        result = hex_to_rgb_hsv(color)
        rgb = result['rgb']
        hsv = result['hsv']
        
        print(f"{color:>7} -> RGB({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f}) -> HSV({hsv[0]:.3f}, {hsv[1]:.3f}, {hsv[2]:.3f})")
    
    # Interactive input
    print("\nEnter hex color codes (or 'quit' to exit):")
    while True:
        user_input = input("Hex color: ").strip()
        if user_input.lower() == 'quit':
            break
        
        try:
            result = hex_to_rgb_hsv(user_input)
            rgb = result['rgb']
            hsv = result['hsv']
            
            print(f"RGB: ({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})")
            print(f"HSV: ({hsv[0]:.3f}, {hsv[1]:.3f}, {hsv[2]:.3f})")
            print()
        except ValueError as e:
            print(f"Error: {e}")

# Alternative function if you just want the HSV values
def hex_to_hsv(hex_color):
    """Simple function that returns just HSV values (all 0-1)"""
    result = hex_to_rgb_hsv(hex_color)
    return result['hsv']