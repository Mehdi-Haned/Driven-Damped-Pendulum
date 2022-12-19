import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as si

t0, tf = 0, 40 #initial and final times
dt = 0.0001 #times step for the times array
times = np.arange(t0, tf+dt, dt)

#Other parameters
om = 2*np.pi
om_0 = 1.5*om
b = om_0/4

#Value of gamma. A parameter that influences how chaotic the system is
g = 1.5

#Initial conditions for y1, y2, y3 respectively
ic1 = np.array([0,0])
ic2 = np.array([0.001,0])
ic3 = np.array([-0.001,0])

#Rewriting the ODE as a system of linear equations
def flux_ode(t, y, om, om_0, b, g):
    return np.array(
        [
            y[1], 
            g*(om_0**2)*np.cos(om*t) - 2*b*y[1] - (om_0**2)*np.sin(y[0])
            ]
        )

method = 'DOP853'

#Solves the DE at the desired initial condition
y1 = si.solve_ivp(
    flux_ode,
    [t0, tf], ic1, t_eval=times, 
    args=[om, om_0, b, g],
    method=method, 
    vectorized=True, 
    rtol=1e-10,
    atol=1e-10
    ).y

#y2 and y3 are optional as they solve the DE a small distance to the left or right of the 
#first initial condition
y2 = si.solve_ivp(
    flux_ode,
    [t0, tf], ic2, t_eval=times, 
    args=[om, om_0, b, g],
    method=method, 
    vectorized=True, 
    rtol=1e-10,
    atol=1e-10
    ).y

y3 = si.solve_ivp(
    flux_ode,
    [t0, tf], ic3, t_eval=times, 
    args=[om, om_0, b, g],
    method=method, 
    vectorized=True, 
    rtol=1e-10,
    atol=1e-10
    ).y

fig1, ax1 = plt.subplots(1)

#Plot the solution curves for the angle at the initial conditions
ax1.plot(times, y1[0], label=f"$\phi_0$ = {ic1[0]}", lw=2, c="r")
ax1.plot(times, y2[0], label=f"$\phi_0$ = {ic2[0]}", lw=2, c="g")
ax1.plot(times, y3[0], label=f"$\phi_0$ = {ic3[0]}", lw=2, c="b")
ax1.legend()
ax1.set_title(f"$\phi(t)$ for $\gamma$ = {g}")
ax1.grid()
ax1.set_xlabel("t")
ax1.set_ylabel("$\phi(t)$")
fig1.subplots_adjust(left=0.053, bottom=0.137, top=0.91, right=0.967)

fig2, ax2 = plt.subplots(1)

#Plot the solution curves for the angular velocity at the initial conditions
ax2.plot(times, y1[1], label=f"$\phi'_0$ = {ic1[1]}", lw=2, c="r")
ax2.plot(times, y2[1], label=f"$\phi'_0$ = {ic2[1]}", lw=2, c="g")
ax2.plot(times, y3[1], label=f"$\phi'_0$ = {ic3[1]}", lw=2, c="b")
ax2.legend()
ax2.set_title(f"$\phi'(t)$ for $\gamma$ = {g}")
ax2.grid()
ax2.set_xlabel("t")
ax2.set_ylabel("$\phi'(t)$")
fig2.subplots_adjust(left=0.053, bottom=0.137, top=0.91, right=0.967)

plt.show()