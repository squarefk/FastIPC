import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(19680801)

sigmaFRef = 10
m = 2
N = 10000 #we can simply use number of particles for Vref / V

R = np.random.rand(N)
for i in range(len(R)):
    R[i] = math.log(R[i]) / math.log(0.5)

sigmaF = sigmaFRef * (N * R)**(1.0/m)

n, bins, patches = plt.hist(sigmaF, 50, density=True, facecolor='g', alpha=0.75)

plt.xlabel('SigmaF')
plt.ylabel('Probability')
plt.title('Histogram of Weibull Distributed SigmaF (sigmaFRef = 10, m = 3)')
#plt.xlim(0, 20)
#plt.ylim(0, 0.3)
plt.grid(True)
plt.show()
