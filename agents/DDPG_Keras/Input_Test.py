from agents.DDPG_Keras.rbuffer import RingBuffer
import numpy as np

buffer =  RingBuffer(4)

baseArr = np.random.randint(10,90,(16,64,4))

buffer.insert_obs(baseArr[:,:,0])
buffer.insert_obs(baseArr[:,:,1])
buffer.insert_obs(baseArr[:,:,2])
buffer.insert_obs(baseArr[:,:,3])
targetArr = buffer.generate_arr()

print(np.array_equal(baseArr,targetArr))

baseArr = np.random.randint(10,90,(1,16,64,4))

if baseArr[0].shape == (16,64,4):
    buffer.insert_obs(baseArr[0][:,:,0])
else:
    buffer.insert_obs(baseArr[0][0][:,:,0])


buffer.generate_arr()

#(1,2,16,64,4) == BAD
#(1,16,64,4) == GOOD
