def hex_to_rgb(hex_color):
    """
    Convert a 6-digit hex color code to RGB values between 0 and 1
    
    Args:
        hex_color (str): Hex color code (with or without #)
    
    Returns:
        tuple: (r, g, b) values between 0 and 1
    """
    # Remove # if present
    hex_color = hex_color.lstrip('#')
    
    # Validate input
    if len(hex_color) != 6:
        raise ValueError("Hex color must be 6 digits")
    
    # Convert hex to RGB (0-255)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    # Convert to 0-1 range
    return (r/255.0, g/255.0, b/255.0)

# Example usage
if __name__ == "__main__":
    # Test with various hex colors
    colors = ["C27E35","757008","1E3226","0E568F","A23742","762908"]
    
    for color in colors:
        rgb = hex_to_rgb(color)
        print(f"{color:>7} -> RGB({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})")
    
    # Interactive input
    print("\nEnter hex color codes (or 'quit' to exit):")
    while True:
        user_input = input("Hex color: ").strip()
        if user_input.lower() == 'quit':
            break
        
        try:
            rgb = hex_to_rgb(user_input)
            print(f"RGB: ({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})")
        except ValueError as e:
            print(f"Error: {e}")