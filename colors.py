import random

def random_color():
    t = random.random()

    max_intensity = 0.7  # Adjust this value to control the saturation

    if t < 0.5:
        # Choose a shade of red
        r,g,b = interpolate_color((1, 0, 0), (1, 1, 1), t / 0.5)
    else:
        # Choose a shade of blue
        r,g,b = interpolate_color((1, 1, 1), (0, 1, 0.7), (t - 0.5) / 0.5)
    a = abs(2*(t - 0.5))
    return r,g,b,a

def interpolate_color(color1, color2, t):
    """ Interpolate between two colors """
    tint = 0.7
    return tuple(a + (b - a) * t for a, b in zip(color1, color2))