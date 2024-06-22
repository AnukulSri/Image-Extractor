import cv2
import numpy as np
import keras_ocr

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return (x_mid, y_mid)

# Initialize the keras-ocr pipeline
pipeline = keras_ocr.pipeline.Pipeline()

def remove_text_and_emojis(img_path, pipeline):
    # Read image
    img = cv2.imread(img_path)
    original_img = img.copy()  # Make a copy for final comparison
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Generate (word, box) tuples using keras-ocr
    prediction_groups = pipeline.recognize([original_img])
    
    # Create a mask to identify text regions
    mask = np.zeros(gray.shape, dtype="uint8")
    
    # Iterate through each detected text box
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]
        
        # Calculate midpoints of opposite sides of the bounding box
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mid1 = midpoint(x0, y0, x3, y3)
        
        # Calculate the thickness of the line
        thickness = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        
        # Draw lines on the mask to cover the text region
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mid1), 255, thickness)
    
    # Inpaint the image using the mask to remove text
    # Using cv2.INPAINT_TELEA for potentially better results
    img = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # Save the inpainted image
    output_path = img_path.split('.')[0] + '_text_removed.png'
    cv2.imwrite(output_path, img)
    
    return output_path

# Example usage
img_path = 'F:\\Images\\pics\\pic2.jpg'
output_image_path = remove_text_and_emojis(img_path, pipeline)
print(f'Removed text and emojis image saved as {output_image_path}')
