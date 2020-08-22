import numpy as np
from collections import deque
from PIL import Image

class RingBuffer:
    def __init__(self, buffer_len):
        self.queue = deque([],maxlen=buffer_len)
    def insert_obs(self,obs):
        self.queue.append(obs)
    def generate_arr(self):
        arr = np.array(list(self.queue))
        arr = np.transpose(arr,axes=(1,2,0))
        return arr
    def print_arr(self):
        arr = self.generate_arr()
        arr = arr * 255
        p1 = np.array(arr[:,:,0], dtype=np.uint8)
        p2 = np.array(arr[:,:,1], dtype=np.uint8)
        p3 = np.array(arr[:,:,2], dtype=np.uint8)
        p4 = np.array(arr[:,:,3], dtype=np.uint8)
        img1 = Image.fromarray(p1)
        img2 = Image.fromarray(p2)
        img3 = Image.fromarray(p3)
        img4 = Image.fromarray(p4)
        img1.save("frame_1.png")
        img2.save("frame_2.png")
        img3.save("frame_3.png")
        img4.save("frame_4.png")
 