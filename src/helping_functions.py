from PIL import Image, ImageDraw
import numpy as np

def get_ball_color():
    """
    Define the color for the balls as a lighter shade of #b3ade6.
    """
    return (150, 216, 230)  # Light blue color in RGB format

def get_line_color(pt1, pt2):
    """
    Define the color for the lines based on the connection.
    """
    # Colors for different parts of the skeleton
    color_map = {
        (15, 13): (173, 216, 230),  # #ADD8E6
        (13, 11): (173, 216, 230),  # #ADD8E6
        (16, 14): (173, 216, 230),  # #ADD8E6
        (14, 12): (173, 216, 230),  # #ADD8E6
        (11, 12): (173, 216, 230),  # #ADD8E6
        (5, 11): (230, 173, 173),   # #E6ADAD
        (6, 12): (230, 173, 173),   # #E6ADAD
        (5, 6): (230, 173, 173),    # #E6ADAD
        (5, 7): (230, 173, 173),    # #E6ADAD
        (6, 8): (230, 173, 173),    # #E6ADAD
        (7, 9): (230, 216, 173),    # #E6D8AD
        (8, 10): (230, 216, 173),   # #E6D8AD
        (1, 2): (230, 216, 173),    # #E6D8AD
        (0, 1): (230, 216, 173),    # #E6D8AD
        (0, 2): (230, 216, 173),    # #E6D8AD
        (1, 3): (203, 230, 173),    # #CBE6AD
        (2, 4): (203, 230, 173),    # #CBE6AD
        (3, 5): (203, 230, 173),    # #CBE6AD
        (4, 6): (203, 230, 173)     # #CBE6AD
    }
    # Ensure the connection is in the map, otherwise default to a grey color
    return color_map.get((pt1, pt2), (128, 128, 128))

def draw_3d_ball(draw, center, radius, color):
    """
    Draws a 3D-like ball effect with gradient shading at the specified center and radius.
    """
    for i in range(radius, 0, -1):  # Draw concentric circles for shading effect
            # Reduce gradient effect by making shading less pronounced
            alpha = (radius - i) / radius
            # Use a constant factor to reduce the gradient effect
            shading_factor = 0.4  # Adjust this value to control the shading intensity
            shaded_color = tuple(int(c * (1 - shading_factor * alpha)) for c in color)
            draw.ellipse([center[0] - i, center[1] - i, center[0] + i, center[1] + i], fill=shaded_color)

def draw_full_skeleton(image, keypoints, connections):
    """
    Draws the full skeleton on the image using the keypoints and skeleton structure.
    """
    line_width = 4
    ball_size = 12
    conf_thresh = 0.5
    width, height = 1024, 720

    draw = ImageDraw.Draw(image)

    # Draw the skeleton lines
    for pt1, pt2 in connections:
        if pt1 < len(keypoints) and pt2 < len(keypoints):
            if keypoints[pt1][2] > conf_thresh and keypoints[pt2][2] > conf_thresh:
                x1, y1 = int(keypoints[pt1][0] * width), int(keypoints[pt1][1] * height)
                x2, y2 = int(keypoints[pt2][0] * width), int(keypoints[pt2][1] * height)
                line_color = get_line_color(pt1, pt2)
                draw.line([x1, y1, x2, y2], fill=line_color, width=line_width)  # Thicker line for better visibility

    # Draw the keypoints with 3D ball effect
    ball_color = get_ball_color()
    for point in keypoints:
        if point[2] > conf_thresh:  # Confidence threshold check
            x, y = int(point[0] * width), int(point[1] * height)
            draw_3d_ball(draw, (x, y), ball_size, ball_color)  # Draw larger 3D ball effect

    return image

def plot_keypoints_on_black_background(keypoints, width, height):
    """
    Plot keypoints on a black background with 3D-like balls.
    """
    # Create a black background
    black_background = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(black_background)

    # Define the color for the balls
    ball_color = get_ball_color()

    # Draw the keypoints with a 3D ball effect
    ball_size = 12
    for point in keypoints:
        if len(point) == 2:  # Ensure point has x and y coordinates
            x, y = int(point[0] * width), int(point[1] * height)
            draw_3d_ball(draw, (x, y), ball_size, ball_color)  # Draw 3D ball effect

    return black_background


def main():
    # Define the skeleton and keypoints
    skeleton = [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
        [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
        [1, 3], [2, 4], [3, 5], [4, 6]
    ]

    keypoints = np.array([
        [0.5441, 0.53131],
        [0.58238, 0.45586],
        [0.50191, 0.45157],
        [0.62804, 0.50917],
        [0.44395, 0.50335],
        [0.70178, 0.84081],
        [0.33874, 0.86800]
    ])
    keypoints = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])

    while len(keypoints) < 17:
        keypoints = np.vstack([keypoints, [0, 0, 0]])

    # Initialize image dimensions
    global width, height
    width, height = 1024, 720

    # Create a blank image with black background
    image = Image.new("RGB", (width, height), (0, 0, 0))

    # Draw the skeleton on the image
    image_with_skeleton = draw_full_skeleton(image, keypoints, skeleton)

    # Display the resulting image
    image_with_skeleton.show()
