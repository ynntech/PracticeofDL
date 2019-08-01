# coding: utf-8


import numpy as np
np.random.seed(0)

xs = np.array(((0,0), (0,1), (1,0), (1,1)), dtype=np.float32)
ts = np.array(((-1), (-1), (-1), (1)), dtype=np.float32)

#学習率 0~1の範囲を取る。
lr = 0.1

w = np.random.normal(0., 1, (3))
print ("weight >>",w)

# add bias
_xs = np.hstack([xs, [[1] for _ in range(4)]])


#iteration
ite = 0;
while True :
    ite+=1;
    # feed forward
    ys = np.dot (_xs, w)
    
    print ("iteration:", ite, "y>>", ys)
    
    #update parameters
    if len(np.where(ys * ts < 0)[0]) < 1:
        break
    _ys = ys.copy()
    _ts  = ts.copy()
    
    _ys [ys * ts >= 0] = 0
    _ts [ts * ys >= 0] = 0
    En = np.dot(_ts, _xs)
    
    w += lr * En
print("training finished!")
print("weight >>", w)


# test
ys = np.array(list(map(lambda x: np.dot(w, x), _xs)))

for i in range(4):
    ys = np.dot(w, _xs[i])
    print("in >>", xs[i], ", out >>", ys) 

