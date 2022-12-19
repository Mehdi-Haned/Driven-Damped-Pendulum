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

#Initial conditions for y
ic = np.array([0,0])

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
    [t0, tf], ic, t_eval=times, 
    args=[om, om_0, b, g],
    method=method, 
    vectorized=True, 
    rtol=1e-10,
    atol=1e-10
    ).y

fig1, ax = plt.subplots(1)

#Plot the phase space diagram for the solution
ax.plot(y1[0], y1[1], lw=2, c="k")
ax.scatter(np.array(ic[0]), np.array(ic[1]), c="k")
ax.legend()
ax.set_title(f"Phase space for $\gamma$ = {g} at $(\phi_0, \phi'_0)$ = {(ic[0], ic[1])} from t = {t0} to t = {tf}")
ax.grid()
ax.set_xlabel("$\phi(t)$")
ax.set_ylabel("$\phi'(t)$")
fig1.subplots_adjust(left=0.053, bottom=0.137, top=0.91, right=0.967)


plt.show()