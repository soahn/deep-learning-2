import numpy as np
import matplotlib.pyplot as plt


N = 2   # mini-arrangement size
H = 3   # dimension of hidden vector
T = 20  # length of time data

dh = np.ones((N, H))

np.random.seed(3) # fixing random seed to re-do

Wh = np.random.randn(H, H)
#Wh = np.random.randn(H, H) * 0.5

norm_list = []
for t in range(T):
    dh = np.dot(dh, Wh.T)
    norm = np.sqrt(np.sum(dh**2)) / N
    norm_list.append(norm)

print(norm_list)

# drawing graph
plt.plot(np.arange(len(norm_list)), norm_list)
plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
plt.xlabel('time step(시간 크기)')
plt.ylabel('norm(노름)')
plt.show()
