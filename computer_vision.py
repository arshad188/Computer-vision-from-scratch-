# computer_vision_from_scratch.py
# Pure Python classical computer vision (no OpenCV / NumPy required)

def create_test_image(width=20, height=15):
    """Create a simple synthetic image with a white square on black background."""
    img = [[0 for _ in range(width)] for _ in range(height)]
    for y in range(4, 11):
        for x in range(5, 15):
            img[y][x] = 255
    return img

def print_image(img, title="Image"):
    """Simple ASCII visualization (useful on mobile terminals)."""
    print(f"\n=== {title} ===")
    chars = " .:-=+*#%@"
    for row in img:
        line = ""
        for val in row:
            idx = min(len(chars) - 1, val * (len(chars) - 1) // 255)
            line += chars[idx]
        print(line)

def to_grayscale_rgb(rgb_img):
    """Convert RGB image (list of lists of (R,G,B)) to grayscale."""
    gray = []
    for row in rgb_img:
        gray_row = []
        for r, g, b in row:
            # Luminance formula (human-eye weighted)
            val = int(0.299 * r + 0.587 * g + 0.114 * b)
            gray_row.append(max(0, min(255, val)))
        gray.append(gray_row)
    return gray

def convolve(image, kernel):
    """
    2-D convolution (the heart of almost all classical CV filters).
    image  : 2-D list of ints
    kernel : 2-D list (e.g. 3x3)
    Returns new image of same size (border pixels kept simple).
    """
    h = len(image)
    w = len(image[0])
    kh = len(kernel)
    kw = len(kernel[0])
    pad_h = kh // 2
    pad_w = kw // 2

    # Output image
    result = [[0 for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            acc = 0.0
            for ky in range(kh):
                for kx in range(kw):
                    iy = y + ky - pad_h
                    ix = x + kx - pad_w
                    if 0 <= iy < h and 0 <= ix < w:
                        acc += image[iy][ix] * kernel[ky][kx]
            result[y][x] = int(round(acc))
    return result

def box_blur(image, size=3):
    """Simple averaging blur (noise reduction)."""
    kernel = [[1.0 / (size * size) for _ in range(size)] for _ in range(size)]
    return convolve(image, kernel)

def sobel_edge_detection(image):
    """
    Classic Sobel edge detector (from scratch).
    Returns gradient magnitude image.
    """
    # Sobel kernels
    sobel_x = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
    sobel_y = [
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]

    gx = convolve(image, sobel_x)
    gy = convolve(image, sobel_y)

    # Magnitude
    magnitude = []
    max_val = 0
    for y in range(len(image)):
        row = []
        for x in range(len(image[0])):
            mag = int((gx[y][x]**2 + gy[y][x]**2)**0.5)
            row.append(mag)
            if mag > max_val:
                max_val = mag
        magnitude.append(row)

    # Normalize to 0-255 for display
    if max_val == 0:
        return magnitude
    normalized = []
    for row in magnitude:
        normalized.append([int(v * 255 / max_val) for v in row])
    return normalized

def threshold(image, thresh=128):
    """Simple binary thresholding."""
    return [[255 if pixel >= thresh else 0 for pixel in row] for row in image]

# -------------------- Demo --------------------
if __name__ == "__main__":
    print("Computer Vision from Scratch – Demo")

    # 1. Create synthetic image
    img = create_test_image(30, 20)
    print_image(img, "Original (white square)")

    # 2. Blur (noise reduction / smoothing)
    blurred = box_blur(img, size=3)
    print_image(blurred, "After Box Blur")

    # 3. Edge detection
    edges = sobel_edge_detection(blurred)
    print_image(edges, "Sobel Edges")

    # 4. Threshold the edges
    binary = threshold(edges, thresh=80)
    print_image(binary, "Thresholded Edges")