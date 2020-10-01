#!/usr/bin/env python
# coding: utf-8

# In[2]:


import time
import cv2
import mss
import numpy
from PIL import ImageGrab
import multiprocessing
# from grabscreen import grab_screen


# In[3]:


mon = (0,40,800,640)
monitor = {"top": 40, "left": 0, "width": 800, "height": 640}
sct = mss.mss()
title = "FPS benchmark"
display_time = 2
fps = 0
start_time = time.time()


# In[ ]:


def screen_recordPIL():
    # set variables as global, that we could change them
    global fps, start_time
    # begin our loop
    while True:
        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.asarray(ImageGrab.grab(bbox=mon))
        # Display the picture
        cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # add one to fps
        fps+=1
        # calculate time difference
        TIME = time.time() - start_time
        # check if our 2 seconds passed
        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            # set fps again to zero
            fps = 0
            # set start time to current time again
            start_time = time.time()
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


# In[ ]:


# screen_recordPIL()


# In[ ]:


def screen_grab():
    global fps, start_time
    while True:
        # Get raw pixels from the screen 
        img = grab_screen(region=mon)
        # Display the picture
        cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        fps+=1
        TIME = time.time() - start_time
        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


# In[ ]:


# screen_grab()


# In[ ]:


def screen_recordMSS():
    global fps, start_time
    while True:
        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))
        # to ger real color we do this:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        fps+=1
        TIME = time.time() - start_time
        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


# In[ ]:





# In[ ]:


q = multiprocessing.JoinableQueue()


# In[4]:


def grab_using_mss(q):
    
    global fps, start_time
    while True:
        
        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))
        # to ger real color we do this:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q.put_nowait(img)
        q.join()

    


# In[5]:


def show_using_mss():
    global fps , start_time
    while True:
        if not q.empty():
            img = q.get_nowait()
            q.task_done()
            
            cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            fps+=1
            TIME = time.time() - start_time
            if (TIME) >= display_time :
                print("FPS: ", fps / (TIME))
                fps = 0
                start_time = time.time()
        # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
    


# In[9]:


if __name__=="__main__":
# Queue
    q = multiprocessing.JoinableQueue()

    # creating new processes
    p1 = multiprocessing.Process(target=grab_using_mss, args=(q, ))
    p2 = multiprocessing.Process(target=show_using_mss, args=(q, ))

    # starting our processes
    p1.start()
    p2.start()


# In[ ]:





# In[ ]:




