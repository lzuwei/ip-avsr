from __future__ import print_function
import numpy as np
import cv2

for idx in range(31, 61):
    videofile = '../examples/data/s30_v1_u{}.mp4'.format(idx)
    print('video file: {}'.format(videofile))
    cap = cv2.VideoCapture(videofile)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', gray)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()