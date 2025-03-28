import numpy as np
import cv2
from io import BytesIO

# Load the recorded episode
file_path = "demos/20250324T154528_0_498.npz"  # Replace with your actual file path
data = np.load(file_path, allow_pickle=True)

# Get the compressed camera images
compressed_images = data['camera_image_3']

# Display a specific image (e.g., the first one)
image_bytes = compressed_images[0]
image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
print(image.shape)
cv2.imshow('Camera Image', image)
cv2.waitKey(0)

image_bytes = compressed_images[-1]
image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
cv2.imshow('Camera Image', image)
cv2.waitKey(0)

cv2.destroyAllWindows()

# To display all images sequentially
# for i, img_bytes in enumerate(compressed_images):
#     image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
#     cv2.imshow(f'Image {i}', image)
#     # key = cv2.waitKey(100)  # Wait 100ms between frames
#     key = cv2.waitKey(0)  # Wait for a key press to show the next image
#     if key == 27:  # ESC key to exit
#         break
cv2.destroyAllWindows()
