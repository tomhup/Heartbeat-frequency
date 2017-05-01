#coding utf8
#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import Queue

# local modules
from video import create_capture
from common import clock, draw_str
Sample_Num = 64

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (int((x1+x2)/2 - (x2-x1)*0.1), int(y1+(y2-y1)*0.1)), ( int((x1+x2)/2+(x2-x1)*0.1), int(y1+(y2-y1)*0.2)), (0, 0, 255), 2)

if __name__ == '__main__':
    import sys, getopt

    q = Queue.Queue(maxsize=Sample_Num)
    q2 = Queue.Queue(maxsize=10)
    q3 = Queue.Queue(maxsize=10)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "haarcascade_frontalface_alt.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)
    cam = create_capture(video_src)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        t = clock()
        rects = detect(gray, cascade)
        vis = img.copy()
        if len(rects) > 0:
            x1, y1, x2, y2 = rects[0]
            xx1, yy1, xx2, yy2 = int((x1+x2)/2 - (x2-x1)*0.1), int(y1+(y2-y1)*0.1),  int((x1+x2)/2+(x2-x1)*0.1), int(y1+(y2-y1)*0.2)

            gg = img[x1:x2, y1:y2, 1]

            if q.full():
                q.get()
                q.put(gg)
            else:
                q.put(gg)

        if len(rects) == 0 and not q.empty():
            q.get()

        zz = map(lambda x: np.sum(x.ravel()), np.array(q.queue))
        draw_rects(vis, rects, (0, 255, 0))
        dt = clock() - t
        tf = 0
        ft = 1000.0 / (dt * 1000 + 10+5)

        if q3.full():
            q3.get(); q3.put(ft)
        else:
            q3.put(ft)
        ft = np.average(np.array(q3.queue))

        if q.full():
            frez = np.abs(np.fft.fft(zz, Sample_Num))
            frez[0:20] = 0
            tf = frez[0:len(frez)/2]
            tf = np.where(tf == max(tf))
            tf = tf[0]*ft/q.qsize()*10
            if q2.full():
                q2.get()
                q2.put(tf)
            else:
                q2.put(tf)

            tf = np.average(np.array(q2.queue))

        draw_str(vis, (20, 20), 'Sample Freq: %.0f Heartbeat Freq: %.0f Sample Num: %d '%(ft,tf, len(zz)))
        cv2.imshow('Heartbeat frequency', vis)
        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()
