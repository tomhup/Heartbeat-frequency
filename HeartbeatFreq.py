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
Sample_Num = 128

xx1 = lambda x1, x2: int((x1+x2)/2-(x2-x1)*0.2)
xx2 = lambda x1, x2: int((x1+x2)/2+(x2-x1)*0.2)
yy1 = lambda y1, y2: int(y1+(y2-y1)*0.1)
yy2 = lambda y1, y2: int(y1+(y2-y1)*0.2)

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
        cv2.rectangle(img, (xx1(x1,x2), yy1(y1,y2)),
                           (xx2(x1,x2), yy2(y1,y2)),
                           (0, 0, 255), 2)

if __name__ == '__main__':
    import sys, getopt

    q_data = Queue.Queue(maxsize=Sample_Num)
    q_heart = Queue.Queue(maxsize=10)
    q_samplefreq = Queue.Queue(maxsize=10)

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
            xxx1, yyy1, xxx2, yyy2 = xx1(x1,x2), yy1(y1,y2),  \
                                     xx2(x1,x2), yy2(y1,y2)
            gg = img[xxx1:xxx2, yyy1:yyy2, 1]
            if q_data.full():
                q_data.get()
                q_data.put(gg)
            else:
                q_data.put(gg)

        if len(rects) == 0 and not q_data.empty():
            q_data.get()

        zz = map(lambda x: np.sum(x.ravel()), np.array(q_data.queue))
        draw_rects(vis, rects, (0, 255, 0))
        dt = clock() - t
        tf = 0
        ft = 1000.0 / (dt * 1000 + 10+5)
        if q_samplefreq.full():
            q_samplefreq.get(); q_samplefreq.put(ft)
        else:
            q_samplefreq.put(ft)
        ft = np.average(np.array(q_samplefreq.queue))

        if q_data.full():
            frez = np.abs(np.fft.fft(zz, Sample_Num))
            frez[0:20] = 0
            tf = frez[0:len(frez)/2]
            tf = np.where(tf == max(tf))
            tf = tf[0]*ft/Sample_Num*10
            if q_heart.full():
                q_heart.get()
                q_heart.put(tf)
            else:
                q_heart.put(tf)
            tf = np.average(np.array(q_heart.queue))
        draw_str(vis, (20, 20), 'Sample Freq: %.0f Heartbeat Freq: %.0f Sample Num: %d '%(ft,tf, len(zz)))
        cv2.imshow('Heartbeat frequency', vis)
        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()
