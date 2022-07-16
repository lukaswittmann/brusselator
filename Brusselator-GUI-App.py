import numpy as np
import matplotlib.pyplot as plt
import time
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox as mb
import random

t_tot = 500
dt = 0.05
l_tot = 200  # Breite
n = len(np.arange(0, t_tot + 0.01, dt))
t = np.arange(0, t_tot + 0.01, dt)

A = 1
B = 3
k1 = 1.0
k2 = 1.0
k3 = 1.0
k4 = 1.0
Dx = 0.1
Dy = 0.01
stop_var = 0
betrachtungsintervall = 20
Cx = np.ones(shape=(l_tot + 1, l_tot + 1), dtype=float)
Cx[:, :] = (k1*A)/k2

plt.ion()


def var_initialisieren():
    global t_tot,dt,l_tot,n,t,A,B,k1,k2,k3,k4,stop_var,betrachtungsintervall,Dx,Dy,Cx_oben,Cx_unten,Cx_links,Cx_rechts,Cy_oben,Cy_unten,Cy_links,Cy_rechts,Cx_diff_gesamt,Cy_diff_gesamt,Cx0,Cy0
    Cx = np.ones(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cy = np.ones(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cx_oben = np.zeros(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cx_unten = np.zeros(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cx_links = np.zeros(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cx_rechts = np.zeros(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cy_oben = np.zeros(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cy_unten = np.zeros(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cy_links = np.zeros(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cy_rechts = np.zeros(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cx_diff_gesamt = np.zeros(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cy_diff_gesamt = np.zeros(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cx0 = np.random.rand(l_tot + 1, l_tot + 1) ** 2 * 2
    Cy0 = ((np.random.rand(l_tot + 1, l_tot + 1) + 0.3) ** 4) * 2
    return t_tot,dt,l_tot,n,t,A,B,k1,k2,k3,k4,stop_var,betrachtungsintervall,Dx,Dy,Cx_oben,Cx_unten,Cx_links,Cx_rechts,Cy_oben,Cy_unten,Cy_links,Cy_rechts,Cx_diff_gesamt,Cy_diff_gesamt,Cx0,Cy0

def werte_annehmen():
    global switch_variable
    switch_variable = switch_variable_eingabe.get()
    global fluktuationshaeufigkeit
    fluktuationshaeufigkeit = 10000 - int(eingabe_fluktuationshaeufigkeit.get())
    global fluktuationsgroesse
    fluktuationsgroesse = int(eingabe_fluktuationsgroesse.get())
    global l_tot
    l_tot = int(eingabe_gitter.get())
    global stop_var
    stop_var=0
    global betrachtungsintervall
    betrachtungsintervall = int(eingabe_betrachtungsintervall.get())
    global t_tot
    t_tot = int(eingabe_t.get())
    global dt
    dt = float(eingabe_dt.get())
    global k1
    k1 = float(eingabe_k1.get())
    global k2
    k2 = float(eingabe_k2.get())
    global k3
    k3 = float(eingabe_k3.get())
    global k4
    k4 = float(eingabe_k4.get())
    global A
    A = float(eingabe_A.get())
    global B
    B = float(eingabe_B.get())
    global Dx
    Dx = float(eingabe_Dx.get())
    global Dy
    Dy = float(eingabe_Dy.get())
    global n
    n = len(np.arange(0, t_tot + 0.01, dt))
    global t
    t = np.arange(0, t_tot + 0.01, dt)
    global Cx,Cy,Cx_oben,Cx_unten,Cx_links,Cx_rechts,Cy_oben,Cy_unten,Cy_links,Cy_rechts,Cx_diff_gesamt,Cy_diff_gesamt,Cx0,Cy0
    Cx = np.ones(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cy = np.ones(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cx_oben = np.zeros(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cx_unten = np.zeros(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cx_links = np.zeros(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cx_rechts = np.zeros(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cy_oben = np.zeros(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cy_unten = np.zeros(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cy_links = np.zeros(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cy_rechts = np.zeros(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cx_diff_gesamt = np.zeros(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cy_diff_gesamt = np.zeros(shape=(l_tot + 1, l_tot + 1), dtype=float)
    Cx0 = np.random.rand(l_tot + 1, l_tot + 1) ** 2 * 2
    Cy0 = ((np.random.rand(l_tot + 1, l_tot + 1) + 0.3) ** 4) * 2
    return fluktuationshaeufigkeit, fluktuationsgroesse, t_tot, dt, k1, k2, k3, k4, A, B, Dx, Dy, n, t, betrachtungsintervall,Cx_oben,Cx_unten,Cx_links,Cx_rechts,Cy_oben,Cy_unten,Cy_links,Cy_rechts,Cx_diff_gesamt,Cy_diff_gesamt,Cx0,Cy0,Cx,Cy

def berechnung_dC_ist_0():
    werte_annehmen()
    global Cx0, Cy0, Cx, Cy, Cx_oben, Cx_unten, Cx_links, Cx_rechts, Cx_diff_gesamt, Cy_diff_gesamt, Cy_oben, Cy_unten, Cy_links, Cy_rechts

    start = time.time()

    print(switch_variable)

    if switch_variable == 0:
        Cx[:, :] = (k1 * A) / k4
        Cy[:, :] = (k2 * k4 * B) / (k3 * k1 * A)
    elif switch_variable == 1:
        Cx[:, :] = Cx0[:, :]
        Cy[:, :] = Cy0[:, :]

    print("Start der Berechnung...")
    h = 0
    for i in range(1, n):  # loop von 1 bis n

        if switch_variable == 0:
            if i % int(random.random() * fluktuationshaeufigkeit + 1) == 0:
                f1 = (random.random() / 10000) * fluktuationsgroesse
                f2 = (random.random() / 10000) * fluktuationsgroesse
                Cx[int(random.random() * l_tot), int(random.random() * l_tot)] += f1
                Cy[int(random.random() * l_tot), int(random.random() * l_tot)] += f2
                Cx[int(random.random() * l_tot), int(random.random() * l_tot)] -= f1
                Cy[int(random.random() * l_tot), int(random.random() * l_tot)] -= f2

        Cx_oben[:, :] = np.roll(Cx[:, :], -1, axis=0)
        Cx_oben[-1, :] = 0
        Cx_oben[-2, :] *= (4 / 3)
        Cx_oben[:, 0] *= (4 / 3)
        Cx_oben[:, -1] *= (4 / 3)
        Cx_oben[-2, 0] *= (18 / 16)
        Cx_oben[-2, -1] *= (18 / 16)

        Cx_unten[:, :] = np.roll(Cx[:, :], 1, axis=0)
        Cx_unten[0, :] = 0
        Cx_unten[1, :] *= (4 / 3)
        Cx_unten[:, 0] *= (4 / 3)
        Cx_unten[:, -1] *= (4 / 3)
        Cx_unten[1, 0] *= (18 / 16)
        Cx_unten[1, -1] *= (18 / 16)

        Cx_links[:, :] = np.roll(Cx[:, :], -1, axis=1)
        Cx_links[:, -1] = 0
        Cx_links[:, -2] *= (4 / 3)
        Cx_links[0, :] *= (4 / 3)
        Cx_links[-1, :] *= (4 / 3)
        Cx_links[0, -2] *= (18 / 16)
        Cx_links[-1, -2] *= (18 / 16)

        Cx_rechts[:, :] = np.roll(Cx[:, :], 1, axis=1)
        Cx_rechts[:, 0] = 0
        Cx_rechts[:, 1] *= (4 / 3)
        Cx_rechts[0, :] *= (4 / 3)
        Cx_rechts[-1, :] *= (4 / 3)
        Cx_rechts[0, 1] *= (18 / 16)
        Cx_rechts[-1, 1] *= (18 / 16)

        Cy_oben[:, :] = np.roll(Cy[:, :], -1, axis=0)
        Cy_oben[-1, :] = 0
        Cy_oben[-2, :] *= (4 / 3)
        Cy_oben[:, 0] *= (4 / 3)
        Cy_oben[:, -1] *= (4 / 3)
        Cy_oben[-2, 0] *= (18 / 16)
        Cy_oben[-2, -1] *= (18 / 16)

        Cy_unten[:, :] = np.roll(Cy[:, :], 1, axis=0)
        Cy_unten[0, :] = 0
        Cy_unten[1, :] *= (4 / 3)
        Cy_unten[:, 0] *= (4 / 3)
        Cy_unten[:, -1] *= (4 / 3)
        Cy_unten[1, 0] *= (18 / 16)
        Cy_unten[1, -1] *= (18 / 16)

        Cy_links[:, :] = np.roll(Cy[:, :], -1, axis=1)
        Cy_links[:, -1] = 0
        Cy_links[:, -2] *= (4 / 3)
        Cy_links[0, :] *= (4 / 3)
        Cy_links[-1, :] *= (4 / 3)
        Cy_links[0, -2] *= (18 / 16)
        Cy_links[-1, -2] *= (18 / 16)

        Cy_rechts[:, :] = np.roll(Cy[:, :], 1, axis=1)
        Cy_rechts[:, 0] = 0
        Cy_rechts[:, 1] *= (4 / 3)
        Cy_rechts[0, :] *= (4 / 3)
        Cy_rechts[-1, :] *= (4 / 3)
        Cy_rechts[0, 1] *= (18 / 16)
        Cy_rechts[-1, 1] *= (18 / 16)

        Cx_diff_gesamt[:, :] = Dx * (- 4 * Cx[:, :] + Cx_oben[:, :] + Cx_unten[:, :] + Cx_links[:, :] + Cx_rechts[:, :])
        Cy_diff_gesamt[:, :] = Dy * (- 4 * Cy[:, :] + Cy_oben[:, :] + Cy_unten[:, :] + Cy_links[:, :] + Cy_rechts[:, :])

        Cx[:, :] = (k1 * A - k2 * B * Cx[:, :] + k3 * Cy[:, :] * Cx[:, :] ** 2 - k4 * Cx[:, :]) * dt + Cx[:,:] + Cx_diff_gesamt[:,:] * dt
        Cy[:, :] = (k2 * B * Cx[:, :] - k3 * Cy[:, :] * Cx[:, :] ** 2) * dt + Cy[:, :] + Cy_diff_gesamt[:, :] * dt

        end = time.time()

        if i % ((t_tot / dt) / 1000) == 0:
            Fortschritt(i, n, end, start)

        if i % betrachtungsintervall == 0:
            draw_figure(i, start, end, betrachtungsintervall)
            h += 1

        if stop_var == 1:
            break
    return

def berechnung_C_ist_C0():
    werte_annehmen()
    global Cx0, Cy0, Cx, Cy, Cx_oben, Cx_unten, Cx_links, Cx_rechts, Cx_diff_gesamt, Cy_diff_gesamt, Cy_oben, Cy_unten, Cy_links, Cy_rechts

    if switch_variable == 0:
        Cx[:, :] = (k1 * A) / k4
        Cy[:, :] = (k2 * k4 * B) / (k3 * k1 * A)
    elif switch_variable == 1:
        Cx[:, :] = Cx0[:, :]
        Cy[:, :] = Cy0[:, :]

    start = time.time()

    print("Start der Berechnung...")
    h = 0
    for i in range(1, n):  # loop von 1 bis n

        if switch_variable == 0:
            if i % int(random.random() * fluktuationshaeufigkeit + 1) == 0:
                f1 = (random.random() / 10000) * fluktuationsgroesse
                f2 = (random.random() / 10000) * fluktuationsgroesse
                Cx[int(random.random() * l_tot), int(random.random() * l_tot)] += f1
                Cy[int(random.random() * l_tot), int(random.random() * l_tot)] += f2
                Cx[int(random.random() * l_tot), int(random.random() * l_tot)] -= f1
                Cy[int(random.random() * l_tot), int(random.random() * l_tot)] -= f2

        Cx[0, :] = Cx0[0, :]
        Cx[:, 0] = Cx0[:, 0]
        Cx[-1, :] = Cx0[-1, :]
        Cx[:, -1] = Cx0[:, -1]
        Cy[0, :] = Cy0[0, :]
        Cy[:, 0] = Cy0[:, 0]
        Cy[-1, :] = Cy0[-1, :]
        Cy[:, -1] = Cy0[:, -1]

        Cx_oben[:, :] = np.roll(Cx[:, :], -1, axis=0)
        Cx_oben[-1, :] = 0

        Cx_unten[:, :] = np.roll(Cx[:, :], 1, axis=0)
        Cx_unten[0, :] = 0

        Cx_links[:, :] = np.roll(Cx[:, :], -1, axis=1)
        Cx_links[:, -1] = 0

        Cx_rechts[:, :] = np.roll(Cx[:, :], 1, axis=1)
        Cx_rechts[:, 0] = 0

        Cy_oben[:, :] = np.roll(Cy[:, :], -1, axis=0)
        Cy_oben[-1, :] = 0

        Cy_unten[:, :] = np.roll(Cy[:, :], 1, axis=0)
        Cy_unten[0, :] = 0

        Cy_links[:, :] = np.roll(Cy[:, :], -1, axis=1)
        Cy_links[:, -1] = 0

        Cy_rechts[:, :] = np.roll(Cy[:, :], 1, axis=1)
        Cy_rechts[:, 0] = 0

        Cx_diff_gesamt[:, :] = Dx * (- 4 * Cx[:, :] + Cx_oben[:, :] + Cx_unten[:, :] + Cx_links[:, :] + Cx_rechts[:, :])
        Cy_diff_gesamt[:, :] = Dy * (- 4 * Cy[:, :] + Cy_oben[:, :] + Cy_unten[:, :] + Cy_links[:, :] + Cy_rechts[:, :])

        Cx[:, :] = (k1 * A - k2 * B * Cx[:, :] + k3 * Cy[:, :] * Cx[:, :] ** 2 - k4 * Cx[:, :]) * dt + Cx[:,:] + Cx_diff_gesamt[:,:] * dt
        Cy[:, :] = (k2 * B * Cx[:, :] - k3 * Cy[:, :] * Cx[:, :] ** 2) * dt + Cy[:, :] + Cy_diff_gesamt[:, :] * dt

        end = time.time()

        if i % ((t_tot / dt) / 1000) == 0:
            Fortschritt(i, n, end, start)

        if i % betrachtungsintervall == 0:
            draw_figure(i, start, end, betrachtungsintervall)
            h += 1

        if stop_var == 1:
            break
    return

def berechnung_periodisch():
    werte_annehmen()
    global Cx0, Cy0, Cx, Cy, Cx_oben, Cx_unten, Cx_links, Cx_rechts, Cx_diff_gesamt, Cy_diff_gesamt, Cy_oben, Cy_unten, Cy_links, Cy_rechts

    if switch_variable == 0:
        Cx[:, :] = (k1 * A) / k4
        Cy[:, :] = (k2 * k4 * B) / (k3 * k1 * A)
    elif switch_variable == 1:
        Cx[:, :] = Cx0[:, :]
        Cy[:, :] = Cy0[:, :]

    start = time.time()

    print("Start der Berechnung...")
    h = 0
    for i in range(1, n):  # loop von 1 bis n

        if switch_variable == 0:
            if i % int(random.random() * fluktuationshaeufigkeit + 1) == 0:
                f1 = (random.random() / 10000) * fluktuationsgroesse
                f2 = (random.random() / 10000) * fluktuationsgroesse
                Cx[int(random.random() * l_tot), int(random.random() * l_tot)] += f1
                Cy[int(random.random() * l_tot), int(random.random() * l_tot)] += f2
                Cx[int(random.random() * l_tot), int(random.random() * l_tot)] -= f1
                Cy[int(random.random() * l_tot), int(random.random() * l_tot)] -= f2

        Cx_oben[:, :] = np.roll(Cx[:, :], -1, axis=0)
        Cx_oben[-1, :] = 0 + Cx[0, :]

        Cx_unten[:, :] = np.roll(Cx[:, :], 1, axis=0)
        Cx_unten[0, :] = 0 + Cx[-1, :]

        Cx_links[:, :] = np.roll(Cx[:, :], -1, axis=1)
        Cx_links[:, -1] = 0 + Cx[:, 0]

        Cx_rechts[:, :] = np.roll(Cx[:, :], 1, axis=1)
        Cx_rechts[:, 0] = 0 + Cx[:, -1]


        Cy_oben[:, :] = np.roll(Cy[:, :], -1, axis=0)
        Cy_oben[-1, :] = 0 + Cy[0, :]

        Cy_unten[:, :] = np.roll(Cy[:, :], 1, axis=0)
        Cy_unten[0, :] = 0 + Cy[-1, :]

        Cy_links[:, :] = np.roll(Cy[:, :], -1, axis=1)
        Cy_links[:, -1] = 0 + Cy[:, 0]

        Cy_rechts[:, :] = np.roll(Cy[:, :], 1, axis=1)
        Cy_rechts[:, 0] = 0 + Cy[:, -1]

        Cx_diff_gesamt[:, :] = Dx * (- 4 * Cx[:, :] + Cx_oben[:, :] + Cx_unten[:, :] + Cx_links[:, :] + Cx_rechts[:, :])
        Cy_diff_gesamt[:, :] = Dy * (- 4 * Cy[:, :] + Cy_oben[:, :] + Cy_unten[:, :] + Cy_links[:, :] + Cy_rechts[:, :])

        Cx[:, :] = (k1 * A - k2 * B * Cx[:, :] + k3 * Cy[:, :] * Cx[:, :] ** 2 - k4 * Cx[:, :]) * dt + Cx[:,:] + Cx_diff_gesamt[:,:] * dt
        Cy[:, :] = (k2 * B * Cx[:, :] - k3 * Cy[:, :] * Cx[:, :] ** 2) * dt + Cy[:, :] + Cy_diff_gesamt[:, :] * dt

        end = time.time()

        if i % ((t_tot / dt) / 1000) == 0:

            Fortschritt(i, n, end, start)

        if i % betrachtungsintervall == 0:
            draw_figure(i, start, end, betrachtungsintervall)
            h += 1

        if stop_var == 1:
            break
    return

figure = plt.imshow(Cx[:, :], cmap='RdBu', interpolation="none")  # , cmap='hot', interpolation="nearest", cmap='hsv', interpolation="lanczos"
plt.axis('off')
plt.title("Bereit..")
plt.clim(-0.3, 2.5)
plt.tight_layout()

def draw_figure(i, start, end, betrachtungsintervall):
    plt.show()
    figure.set_data(Cx[:, :])
    if i % (betrachtungsintervall*10) == 0:
        plt.title("t=" + str(np.round((dt * i), decimals=1)) + ", " + str(
        np.round((i / n) * 100, decimals=1)) + "% erledigt,  " + str(
        np.round(i / (end - start), decimals=1)) + " Iters/s und " + str(
        np.round(i / (end - start) / betrachtungsintervall, decimals=1)) + " fps, Restdauer: " + str(
        np.round((((t_tot / dt) - i) / (i / (end - start))) / 60, decimals=1)) + " min")
    plt.draw()
    plt.pause(0.001)
    return

def Fortschritt(i, n, end, start):
    print("Berechnung: " + str(np.round((i / n) * 100, decimals=1)) + "% erledigt,  " + str(
                np.round(i / (end - start+0.01), decimals=1)) + " Iters/s und " + str(
                np.round(i / (end - start+0.01)/ betrachtungsintervall, decimals=1)) +  " fps, Restdauer: " + str(
                np.round((((t_tot / dt) - i) / (i / (end - start+0.01))) / 60, decimals=1)) + " min")
    return

'''Werte für die UI'''
spalte1=5
spalte2=220
spalte3=450
Zeilenabstand=46
offset=10

window = tk.Tk()
window.tk.call('tk', 'scaling', 2.0)
window.title("Brüsselator Applikation")
window.geometry("950x830")
#window.option_add('*Font', 'fixed 12')
#window.tk.call('tk', 'scaling', 2.0)

tk.Label(window, text="Gittergröße =").place(x=spalte1, y=Zeilenabstand)
eingabe_gitter = tk.Entry(width=15)
eingabe_gitter.insert(0, 200)
eingabe_gitter.place(x=spalte2, y=Zeilenabstand)
def update_slider_eingabe_gitter(val):
    eingabe_gitter.delete(0,"end")
    eingabe_gitter.insert(0, val)
    return
slider_eingabe_gitter = Scale(window,command=update_slider_eingabe_gitter, from_=100,to=400,length=300,digits=2,orient=HORIZONTAL)
slider_eingabe_gitter.set(200)
slider_eingabe_gitter.place(x=spalte3, y=Zeilenabstand*1-offset)

tk.Label(window, text="Gesamtzeit t =").place(x=spalte1, y=Zeilenabstand*2)
eingabe_t = tk.Entry(width=15)
eingabe_t.insert(0, 200)
eingabe_t.place(x=spalte2, y=Zeilenabstand*2)
def update_slider_eingabe_t(val):
    eingabe_t.delete(0,"end")
    eingabe_t.insert(0, val)
    return
slider_eingabe_t = Scale(window,command=update_slider_eingabe_t, from_=0,to=1000,length=300,digits=2,orient=HORIZONTAL)
slider_eingabe_t.set(200)
slider_eingabe_t.place(x=spalte3, y=Zeilenabstand*2-offset)

tk.Label(window, text="Zeitintervall dt =").place(x=spalte1, y=Zeilenabstand*3)
eingabe_dt = tk.Entry(width=15)
eingabe_dt.insert(0, 0.02)
eingabe_dt.place(x=spalte2, y=Zeilenabstand*3)
def update_slider_eingabe_dt(val):
    eingabe_dt.delete(0,"end")
    eingabe_dt.insert(0, val)
    return
slider_eingabe_dt = Scale(window,command=update_slider_eingabe_dt, from_=0.001,to=1.000,length=300,digits=4, resolution=0.001,orient=HORIZONTAL)
slider_eingabe_dt.set(0.02)
slider_eingabe_dt.place(x=spalte3, y=Zeilenabstand*3-offset)


tk.Label(window, text="Betrachtungsintervall = ").place(x=spalte1, y=Zeilenabstand*4)
eingabe_betrachtungsintervall = tk.Entry(width=15)
eingabe_betrachtungsintervall.insert(0, 10)
eingabe_betrachtungsintervall.place(x=spalte2, y=Zeilenabstand*4)
def update_slider_eingabe_betrachtungsintervall(val):
    eingabe_betrachtungsintervall.delete(0,"end")
    eingabe_betrachtungsintervall.insert(0, val)
    return
slider_eingabe_betrachtungsintervall = Scale(window,command=update_slider_eingabe_betrachtungsintervall, from_=0,to=50,length=300,digits=2,orient=HORIZONTAL)
slider_eingabe_betrachtungsintervall.set(10)
slider_eingabe_betrachtungsintervall.place(x=spalte3, y=Zeilenabstand*4-offset)

tk.Label(window, text="k1 =").place(x=spalte1, y=Zeilenabstand*5)
eingabe_k1 = tk.Entry(width=15)
eingabe_k1.insert(0, 1.0)
eingabe_k1.place(x=spalte2, y=Zeilenabstand*5)
def update_slider_eingabe_k1(val):
    eingabe_k1.delete(0,"end")
    eingabe_k1.insert(0, val)
    return
slider_eingabe_k1 = Scale(window,command=update_slider_eingabe_k1, from_=0.0,to=5.0,length=300,digits=3, resolution=0.01,orient=HORIZONTAL)
slider_eingabe_k1.set(1.0)
slider_eingabe_k1.place(x=spalte3, y=Zeilenabstand*5-offset)

tk.Label(window, text="k2 =").place(x=spalte1, y=Zeilenabstand*6)
eingabe_k2 = tk.Entry(width=15)
eingabe_k2.insert(0, 1.0)
eingabe_k2.place(x=spalte2, y=Zeilenabstand*6)
def update_slider_eingabe_k2(val):
    eingabe_k2.delete(0,"end")
    eingabe_k2.insert(0, val)
    return
slider_eingabe_k2 = Scale(window,command=update_slider_eingabe_k2, from_=0.0,to=5.0,length=300,digits=3, resolution=0.01,orient=HORIZONTAL)
slider_eingabe_k2.set(1.0)
slider_eingabe_k2.place(x=spalte3, y=Zeilenabstand*6-offset)

tk.Label(window, text="k3 =").place(x=spalte1, y=Zeilenabstand*7)
eingabe_k3 = tk.Entry(width=15)
eingabe_k3.insert(0, 1.0)
eingabe_k3.place(x=spalte2, y=Zeilenabstand*7)
def update_slider_eingabe_k3(val):
    eingabe_k3.delete(0,"end")
    eingabe_k3.insert(0, val)
    return
slider_eingabe_k3 = Scale(window,command=update_slider_eingabe_k3, from_=0.0,to=5.0,length=300,digits=3, resolution=0.01,orient=HORIZONTAL)
slider_eingabe_k3.set(1.0)
slider_eingabe_k3.place(x=spalte3, y=Zeilenabstand*7-offset)

tk.Label(window, text="k4 =").place(x=spalte1, y=Zeilenabstand*8)
eingabe_k4 = tk.Entry(width=15)
eingabe_k4.insert(0, 1.0)
eingabe_k4.place(x=spalte2, y=Zeilenabstand*8)
def update_slider_eingabe_k4(val):
    eingabe_k4.delete(0,"end")
    eingabe_k4.insert(0, val)
    return
slider_eingabe_k4 = Scale(window,command=update_slider_eingabe_k4, from_=0.0,to=5.0,length=300,digits=3, resolution=0.01,orient=HORIZONTAL)
slider_eingabe_k4.set(1.0)
slider_eingabe_k4.place(x=spalte3, y=Zeilenabstand*8-offset)

tk.Label(window, text="A =").place(x=spalte1, y=Zeilenabstand*9)
eingabe_A = tk.Entry(width=15)
eingabe_A.insert(0, 1.0)
eingabe_A.place(x=spalte2, y=Zeilenabstand*9)
def update_slider_eingabe_A(val):
    eingabe_A.delete(0,"end")
    eingabe_A.insert(0, val)
    return
slider_eingabe_A = Scale(window,command=update_slider_eingabe_A, from_=0.0,to=5.0,length=300,digits=2, resolution=0.1,orient=HORIZONTAL)
slider_eingabe_A.set(1.0)
slider_eingabe_A.place(x=spalte3, y=Zeilenabstand*9-offset)

tk.Label(window, text="B =").place(x=spalte1, y=Zeilenabstand*10)
eingabe_B = tk.Entry(width=15)
eingabe_B.insert(0, 3.0)
eingabe_B.place(x=spalte2, y=Zeilenabstand*10)
def update_slider_eingabe_B(val):
    eingabe_B.delete(0,"end")
    eingabe_B.insert(0, val)
    return
slider_eingabe_B = Scale(window,command=update_slider_eingabe_B, from_=0.0,to=10.0,length=300,digits=3, resolution=0.1,orient=HORIZONTAL)
slider_eingabe_B.set(3.0)
slider_eingabe_B.place(x=spalte3, y=Zeilenabstand*10-offset)

tk.Label(window, text="Dx =").place(x=spalte1, y=Zeilenabstand*11)
eingabe_Dx = tk.Entry(width=15)
eingabe_Dx.insert(0, 0.1)
eingabe_Dx.place(x=spalte2, y=Zeilenabstand*11)
def update_slider_eingabe_Dx(val):
    eingabe_Dx.delete(0,"end")
    eingabe_Dx.insert(0, val)
    return
slider_eingabe_Dx = Scale(window,command=update_slider_eingabe_Dx, from_=0.000,to=2.000,length=300,digits=4, resolution=0.001,orient=HORIZONTAL)
slider_eingabe_Dx.set(0.10)
slider_eingabe_Dx.place(x=spalte3, y=Zeilenabstand*11-offset)


tk.Label(window, text="Dy =").place(x=spalte1, y=Zeilenabstand*12)
eingabe_Dy = tk.Entry(width=15)
eingabe_Dy.insert(0, 0.010)
eingabe_Dy.place(x=spalte2, y=Zeilenabstand*12)
def update_slider_eingabe_Dy(val):
    eingabe_Dy.delete(0,"end")
    eingabe_Dy.insert(0, val)
    return
slider_eingabe_Dy = Scale(window, command=update_slider_eingabe_Dy, from_=0.000,to=2.000,length=300,digits=4, resolution=0.001,orient=HORIZONTAL)
slider_eingabe_Dy.set(0.010)
slider_eingabe_Dy.place(x=spalte3, y=Zeilenabstand*12-offset)

tk.Label(window, text="Fluktuationshäufigkeit =").place(x=spalte1, y=Zeilenabstand*13)
eingabe_fluktuationshaeufigkeit = tk.Entry(width=15)
eingabe_fluktuationshaeufigkeit.insert(0, 0.01)
eingabe_fluktuationshaeufigkeit.place(x=spalte2, y=Zeilenabstand*13)
def update_slider_eingabe_fluktuationshaeufigkeit(val):
    eingabe_fluktuationshaeufigkeit.delete(0,"end")
    eingabe_fluktuationshaeufigkeit.insert(0, val)
    return
slider_eingabe_fluktuationshaeufigkeit = Scale(window, command=update_slider_eingabe_fluktuationshaeufigkeit, from_=10,to=10000,length=300,digits=4, resolution=10,orient=HORIZONTAL)
slider_eingabe_fluktuationshaeufigkeit.set(5000)
slider_eingabe_fluktuationshaeufigkeit.place(x=spalte3, y=Zeilenabstand*13-offset)

tk.Label(window, text="Fluktuationsgröße =").place(x=spalte1, y=Zeilenabstand*14)
eingabe_fluktuationsgroesse = tk.Entry(width=15)
eingabe_fluktuationsgroesse.insert(0, 0.01)
eingabe_fluktuationsgroesse.place(x=spalte2, y=Zeilenabstand*14)
def update_slider_eingabe_fluktuationsgroesse(val):
    eingabe_fluktuationsgroesse.delete(0,"end")
    eingabe_fluktuationsgroesse.insert(0, val)
    return
slider_eingabe_fluktuationsgroesse = Scale(window, command=update_slider_eingabe_fluktuationsgroesse, from_=1,to=10000,length=300,digits=4, resolution=1,orient=HORIZONTAL)
slider_eingabe_fluktuationsgroesse.set(500)
slider_eingabe_fluktuationsgroesse.place(x=spalte3, y=Zeilenabstand*14-offset)

tk.Label(window, text="Startbedingung").place(x=spalte1, y=Zeilenabstand*15)

switch_frame = tk.Frame(window)
switch_frame.place(x=spalte2, y=Zeilenabstand*15)

switch_variable_eingabe = IntVar()
heterogen_button = tk.Radiobutton(switch_frame, text="Zufällige Startverteilung", variable=switch_variable_eingabe,
                            indicatoron=False, value=1, width=24)
fluktuationen_button = tk.Radiobutton(switch_frame, text="Fluktuationen", variable=switch_variable_eingabe,
                            indicatoron=False, value=0, width=24)
heterogen_button.pack(side="left")
fluktuationen_button.pack(side="left")

def about():
    mb.showinfo('Brüsselator Applikation', 'Bachelorarbeit von Lukas Wittmann: \r\"Entwicklung eines Python-Programms zur Untersuchung von Reaktions-Diffusions Kopplung in Oszillierenden Reaktionen\"'
                                           '\runter der Aufsicht von Prof. Dr. Dominik Horinek an der Universität Regensburg' + '\rKontakt: Lukas.Wittmann@live.de')
    return

tk.Button(window, text="Start mit dC=0 RB", width=20, command=berechnung_dC_ist_0, bg="white").place(x=spalte1+20, y=Zeilenabstand*16.5)
tk.Button(window, text="Start mit C=C0 RB", width=20, command=berechnung_C_ist_C0, bg="white").place(x=spalte1+290, y=Zeilenabstand*16.5)
tk.Button(window, text="Start mit zyklischer RB", width=20, command=berechnung_periodisch, bg="white").place(x=spalte1+560, y=Zeilenabstand*16.5)
tk.Button(window, text="Über..", width=6,height=1, command=about).place(x=spalte1+840, y=Zeilenabstand*16.5)

skala = ImageTk.PhotoImage(Image.open("skala.png").resize((108, 546), Image.ANTIALIAS))
skala_label = tk.Label(image=skala)
skala_label.image = skala
skala_label.place(x=790, y=115)
tk.Label(window, text="Darstellung der\rKonzentration [X]").place(x=778, y=60)

window.mainloop()


