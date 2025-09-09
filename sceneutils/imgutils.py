
import numpy as np

def hsv_to_rgb(h, s, v):
    """Convert HSV color to RGB color"""
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



def hsv_to_rgb_vectorized(h, s, v):
    """
    Vectorized conversion of HSV colors to RGB colors.
    
    Parameters:
    h, s, v : numpy arrays or scalars
        Hue, Saturation, Value components in range [0,1]
        
    Returns:
    r, g, b : numpy arrays or scalars
        Red, Green, Blue components in range [0,1]
    """
    h = np.asarray(h)
    s = np.asarray(s)
    v = np.asarray(v)
    
    h_i = (h * 6).astype(int)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    # Initialize RGB arrays with zeros
    shape = np.broadcast(h, s, v).shape
    r = np.zeros(shape)
    g = np.zeros(shape)
    b = np.zeros(shape)
    
    # Use boolean masks to handle each h_i case
    mask = (h_i == 0)
    r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]
    
    mask = (h_i == 1)
    r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]
    
    mask = (h_i == 2)
    r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]
    
    mask = (h_i == 3)
    r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]
    
    mask = (h_i == 4)
    r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]
    
    mask = (h_i == 5) | (h_i == 6)  # Handle both 5 and potential 6 due to rounding
    r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]
    
    return r, g, b