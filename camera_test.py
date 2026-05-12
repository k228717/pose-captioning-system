import cv2

for index in range(5):
    print(f"Trying camera index {index}...")
    cap = cv2.VideoCapture(index)

    if cap.isOpened():
        print(f"Camera found at index {index}")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Frame not received")
                break

            cv2.imshow("Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        break
    else:
        cap.release()

cv2.destroyAllWindows()