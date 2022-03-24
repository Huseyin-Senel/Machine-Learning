import numpy as np
from numpy import random
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter
import time
import math


x = random.randint(200, size=(200, 3))
m = random.randint(200, size=(3, 3))
dist = np.zeros((len(x),len(m)))
b = np.zeros((len(x),len(m)))
new_centers = np.zeros((len(m), 3))


plt.ion()
fig1 = plt.figure(1)
ax = fig1.add_subplot(111, projection='3d')

fig1.tight_layout()
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

for lo in range(0,10):
    b = np.zeros((len(x), len(m)))
    ax.cla()
    #dots, ms, = plt.plot(x[:, 0], x[:, 1], 'ro', m[:, 0], m[:, 1], 'bs')
    plt.plot(x[:, 0], x[:, 1],  x[:, 2],'ro', markersize=2)
    plt.plot( m[:, 0], m[:, 1],m[:, 2],'bs',markersize=7)
    for w, i in zip(x, np.arange(0, len(x))):
        for q, j in zip(m, np.arange(0, len(m))):                   # MESAFE HESAPLAMA
            dist[i, j] = format(math.dist(w, q), '.8f')


    minInRows = (numpy.amin(dist, axis=1))                        # minimum değerleri bulma
    for min, i in zip(minInRows, np.arange(0, len(minInRows))):
        z = numpy.where(dist[i] == min)                               # B değerlerini yazma
        b[i][z[0]] = 1

    '''
    print("---------------------------------------------------------------------------------------------------")
    print("coordinates / distances / min distance / min distance position / row")
    for i, d, min, b1, j in zip(x, dist, minInRows, b, np.arange(0, len(x))):           #Debug
        # print(i,d,min,b1,j)
        print(f'{i} / {d} / {min: .8f} / {b1} / {j}')'''

    for w, i in zip(x, np.arange(0, len(x))):
        qz = int(numpy.where(b[i] == 1)[0][0])                  # Çizgileri oluşturmna
        #plt.plot([w[0], m[qz, 0]], [w[1], m[qz, 1]])
        plt.plot([w[0], m[qz, 0]], [w[1], m[qz, 1]], zs=[w[2] , m[qz,2]],linewidth=1.5)

    for c in range(0, len(b[0])):
        k = 0
        sum_x = 0
        sum_y = 0
        sum_z = 0
        for j in range(0, len(b)):
            if (b[j, c] == 1):
                k += 1
                sum_x += x[j, 0]
                sum_y += x[j, 1]
                sum_z += x[j, 2]
        if (k == 0):
            k = 1
        new_centers[c] = (sum_x / k, sum_y / k,sum_z / k)

    m = new_centers
    fig1.canvas.draw()
    fig1.canvas.flush_events()
    time.sleep(0.5)
plt.ioff()
ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
ax.set_zlabel('feature 3')
plt.show()



'''
def animate(i):
    x1 = np.linspace(0, 4, 1000)
    y1 = i*1
    dots.set_data(x1, y1)
    return dots,

#anim = FuncAnimation(fig1, animate,frames=200, interval=20, blit=True)
#anim.save('s_wave.gif', writer=PillowWriter(fps=30))
#plt.show()


fig2 = plt.figure(2)
ax = plt.axes(xlim=(0, 4), ylim=(-2, 2))
line, = ax.plot([], [], lw=3)
def init():
    line.set_data([], [])
    return line,
def animate(i):
    x = np.linspace(0, 4, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,

anim = FuncAnimation(fig2, animate, init_func=init,frames=200, interval=20, blit=True)
anim.save('sine_wave.gif', writer=PillowWriter(fps=30))

'''


