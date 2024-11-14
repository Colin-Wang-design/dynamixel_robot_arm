import cv2
import pytesseract

# Initialize camera
cap = cv2.VideoCapture(0)  # Change 0 if using an external camera

# Set up Tesseract configuration for single character detection
tesseract_config = '--psm 10 -c tessedit_char_whitelist=H'  # --psm 10: treat as single character, whitelist 'H'

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame for easier processing (optional)
    frame = cv2.resize(frame, (640, 480))

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection to locate the keyboard
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edges, contours, -1, (0, 0, 255), 2)
    # Filter contours to find keys
    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the contour is a rectangle, assume it's a key
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 20:  # Threshold for the size of the keys
                key_roi = gray[y:y+h, x:x+w]

                # Use Tesseract to recognize the text on each key
                text = pytesseract.image_to_string(key_roi, config=tesseract_config).strip()
                
                # Check if recognized text is the letter 'H'
                if text == "E":
                    # Highlight the key with a red rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "E", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display the resulting frame
    cv2.imshow('Keyboard Recognition - Detecting H', frame)
    cv2.imshow('Show contours',edges)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
