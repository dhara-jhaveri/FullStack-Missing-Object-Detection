import cv2
cap = cv2.VideoCapture("rtsp://admin:@RTSP-ATPL-908612-AIPTZ.torqueverse.dev:8612/ch0_0.264")
print(cap.isOpened())  # Should return True