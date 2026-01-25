import cv2

VIDEO_PATH = "assets/videos/demo.mp4"  # put your demo video here
# VIDEO_PATH = 0  # use this later for webcam test

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Cannot open video source:", VIDEO_PATH)
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            print("End of video or cannot read frame.")
            break

        cv2.imshow("Input Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
