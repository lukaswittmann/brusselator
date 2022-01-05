import numpy as np
import matplotlib.pyplot as plt
import time
import random
from scipy.ndimage.interpolation import rotate


'''Definition der Laufvariablen'''
t_tot = 500
dt = 0.01
l = 200
t = np.arange(0, t_tot+0.01, dt)
n = len(t)

'''Nebenbedingungsmatrix erstellen'''

A = np.empty(shape=(l, l), dtype=float)
B = np.empty(shape=(l, l), dtype=float)

start_A = 0.2
stop_A = 1.5

start_B = 1.5
stop_B = 4.5

A_in = np.arange(start_A, stop_A, (stop_A-start_A)/l)
B_in = np.arange(start_B, stop_B, (stop_B-start_B)/l)

A[:, :] = 1
B[:, :] = 3

#A[:,:] = A_in
#B[:,:] = B_in
#B = rotate(B,angle=90)

'''Arrays fuer die beiden Konzentrationen: (Zeit, y-Laenge, x-Laenge)'''
Cx = np.empty(shape=(l, l), dtype=float)
Cy = np.empty(shape=(l, l), dtype=float)

'''Verschobene Konzentrationsarrays fuer die Diffusion'''
Cx_oben = np.empty(shape=(l, l), dtype=float)
Cx_unten = np.empty(shape=(l, l), dtype=float)
Cx_links = np.empty(shape=(l, l), dtype=float)
Cx_rechts = np.empty(shape=(l, l), dtype=float)
Cy_oben = np.empty(shape=(l, l), dtype=float)
Cy_unten = np.empty(shape=(l, l), dtype=float)
Cy_links = np.empty(shape=(l, l), dtype=float)
Cy_rechts = np.empty(shape=(l, l), dtype=float)

Cx_diff_gesamt = np.empty(shape=(l, l), dtype=float)
Cy_diff_gesamt = np.empty(shape=(l, l), dtype=float)

k1 = 1.0
k2 = 1.0
k3 = 1.0
k4 = 1.0

'''Diffusionskoeffizientenmatrix erstellen'''

Dx = np.empty(shape=(l, l), dtype=float)
Dy = np.empty(shape=(l, l), dtype=float)

start_Dx = 1.0
stop_Dx = 3.0

start_Dy = 3.0
stop_Dy = 9.00

Dx_in = np.arange(start_Dx, stop_Dx, (stop_Dx-start_Dx)/l)
Dy_in = np.arange(start_Dy, stop_Dy, (stop_Dy-start_Dy)/l)

Dx[:,:] = 2
Dy[:,:] = 0.2

#Dx[:,:] = Dx_in
#Dy[:,:] = Dy_in
#Dy = rotate(Dy,angle=90)



betrachtungsintervall = 10

'''Einlesen der Startkonzentrationen (optionale wenn benötigt)'''
Cx[:, :] = A
Cy[:, :] = B/A

h = 0

plt.ion()
figure = plt.imshow(Cx[:,:], cmap='RdBu',interpolation="bilinear")  # , cmap='hot', interpolation="nearest", cmap='hsv', interpolation="lanczos"
plt.title("Bereit..", fontsize=8)
plt.axis('off')
plt.clim(0.2, 3.5)
plt.tight_layout()

def draw_figure(i,h):
    figure.set_data(Cx[:,:])
    plt.title("t=" + str(np.round((dt * i), decimals=1)) + ", Iteration " + str(i),fontsize=8)
    #plt.draw()
    #plt.savefig("render/image" + str(h) + ".png", dpi=350)
    plt.pause(0.001)
    return

start = time.time()

print("Start der Berechnung...")
for i in range(1, n):

    if i % int(random.random() * 100 + 1) == 0 and i < 1000:
        f1 = random.random()
        f2 = random.random()
        x1 = int(random.random() * l)
        y1 = int(random.random() * l)
        x2 = int(random.random() * l)
        y2 = int(random.random() * l)
        Cx[x1, y1] += f1
        Cy[x1, y1] -= f1
        Cx[x2, y2] += f2
        Cy[x2, y2] -= f2


    Cx_oben[:, :] = np.roll(Cx[:, :], -1, axis=0)
    Cx_oben[-1, :] = Cx[-1, :]

    Cx_unten[:, :] = np.roll(Cx[:, :], 1, axis=0)
    Cx_unten[0, :] = Cx[0, :]

    Cx_links[:, :] = np.roll(Cx[:, :], -1, axis=1)
    Cx_links[:, -1] = Cx[:, -1]

    Cx_rechts[:, :] = np.roll(Cx[:, :], 1, axis=1)
    Cx_rechts[:, 0] = Cx[:, 0]

    Cy_oben[:, :] = np.roll(Cy[:, :], -1, axis=0)
    Cy_oben[-1, :] = Cy[-1, :]

    Cy_unten[:, :] = np.roll(Cy[:, :], 1, axis=0)
    Cy_unten[0, :] = Cy[0, :]

    Cy_links[:, :] = np.roll(Cy[:, :], -1, axis=1)
    Cy_links[:, -1] = Cy[:, -1]

    Cy_rechts[:, :] = np.roll(Cy[:, :], 1, axis=1)
    Cy_rechts[:, 0] = Cy[:, 0]


    Cx_diff_gesamt[:, :] = Dx[:,:] * (- 4 * Cx[:, :] + Cx_oben[:, :] + Cx_unten[:, :] + Cx_links[:, :] + Cx_rechts[:, :])
    Cy_diff_gesamt[:, :] = Dy[:,:] * (- 4 * Cy[:, :] + Cy_oben[:, :] + Cy_unten[:, :] + Cy_links[:, :] + Cy_rechts[:, :])

    X = (k1 * A[:, :] - k2 * B[:, :] * Cx[:, :] + k3 * Cy[:, :] * Cx[:, :] ** 2 - k4 * Cx[:, :])*dt + Cx[:, :] + Cx_diff_gesamt[:, :] * dt
    Y = (k2 * B[:, :] * Cx[:, :] - k3 * Cy[:, :] * Cx[:, :] ** 2)*dt + Cy[:, :] + Cy_diff_gesamt[:, :] * dt

    Cx[:, :] = X
    Cy[:, :] = Y

    end = time.time()
    if i % ((t_tot/dt)/1000) == 0:
        end = time.time()
        print("Iteration " + str(i) + " " + str(np.round((i/n)*100, decimals=1)) + "% erledigt (" + str(np.round(i / (end - start), decimals=1)) + " I/s, " + str(np.round(i / (end - start)/betrachtungsintervall, decimals=1)) + " f/s) Restdauer: " + str(np.round((((t_tot/dt)-i)/(i / (end - start)))/60, decimals=1)) + " min")

    if i == 1:
        draw_figure(i,h)
        h += 1

    if i % betrachtungsintervall == 0:
        draw_figure(i,h)
        h += 1
