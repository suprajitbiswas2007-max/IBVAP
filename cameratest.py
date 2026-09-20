import cv2

cap = cv2.VideoCapture(0)
cv2.namedWindow("Registration Capture", cv2.WINDOW_NORMAL)

count = 0
print("Press 's' to save a photo, 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Registration Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        count += 1
        filename = f"webcam_reg_{count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()