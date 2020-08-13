import numpy as np
import collections
from PIL import Image


class RingBuffer:
    def __init__(self, buffer_len):
        self.queue = collections.deque([],maxlen=buffer_len)
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
        img1.save("1.png")
        img2.save("2.png")
        img3.save("3.png")
        img4.save("4.png")
 