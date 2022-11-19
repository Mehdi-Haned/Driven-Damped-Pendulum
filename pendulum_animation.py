import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as si

t0, tf, trel = 0, 300, 5 #initial and final times
dt = 0.01 #times step for the times array
times = np.arange(t0, tf+dt, dt)

#Other parameters
om = 2*np.pi
om_0 = 1.5*om
b = om_0/4

#Value of gamma. The parameter that influences how chaotic the system is
g = 1.085

#Initial conditions for y1, y2, y3 respectively
ic1 = np.array([0,0])

#Rewriting the ODE as a system of linear equations - sine version
def flux_ode(t, y, om, om_0, b, g):
    return np.array(
        [
            y[1], 
            g*(om_0**2)*np.cos(om*t) - 2*b*y[1] - (om_0**2)*np.sin(y[0])
            ]
        )

#rewriting the DE using the small angle approximation
def flux_ode2(t, y, om, om_0, b, g):
    return np.array(
        [
            y[1], 
            g*(om_0**2)*np.cos(om*t) - 2*b*y[1] - (om_0**2)*y[0]
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

#Solves the DE with the small angle approximation
y2 = si.solve_ivp(
    flux_ode2,
    [t0, tf], ic1, t_eval=times, 
    args=[om, om_0, b, g],
    method=method, 
    vectorized=True, 
    rtol=1e-10,
    atol=1e-10
    ).y

angle = y1[0][tf-trel:]
angle2 = y2[0][tf-trel:]
L = 1

def get_coords(phi):
    """Return the (x, y) coordinates of the bob at angle th."""
    return L * np.sin(phi), -L * np.cos(phi)

# Initialize the animation plot. Make the aspect ratio equal so it looks right.
fig, (ax, ax2) = plt.subplots(1,2, figsize=(8,8))
fig.subplots_adjust(left=0.053, bottom=0.137, top=0.91, right=0.967)

# ax = fig.add_subplot(aspect='equal')
# ax2 = fig.add_subplot(aspect='equal')
ax.set_aspect("equal")
ax2.set_aspect("equal")

# The pendulum rod, in its initial position.
x0, y0 = get_coords(ic1[0])
line, = ax.plot([0, x0], [0, y0], lw=3, c='k')
line2, = ax2.plot([0, x0], [0, y0], lw=3, c='k')
# The pendulum bob: set zorder so that it is drawn over the pendulum rod.
bob_radius = 0.06

circle = ax.add_patch(plt.Circle(get_coords(angle[0]), bob_radius,
                      fc='r', zorder=3))

circle2 = ax2.add_patch(plt.Circle(get_coords(angle[0]), bob_radius,
                      fc='b', zorder=3))

# Set the plot limits so that the pendulum has room to swing!
ax.set_xlim(-L*1.2, L*1.2)
ax.set_ylim(-L*1.2, L*1.2)

ax2.set_xlim(-L*1.2, L*1.2)
ax2.set_ylim(-L*1.2, L*1.2)

def animate(i):
    """Update the animation at frame i."""
    x, y = get_coords(angle[i])
    line.set_data([0, x], [0, y])
    circle.set_center((x, y))

    x2, y2 = get_coords(angle2[i])
    line2.set_data([0, x2], [0, y2])
    circle2.set_center((x2, y2))

nframes = len(angle) - 1
interval = dt * 1000
ani = animation.FuncAnimation(fig, animate, frames=nframes, repeat=True,
                              interval=interval,cache_frame_data=False)

ax.set_title(f"DDP with chaos \n $\ddot{{\phi}} + 2 {2*b:.2f} \dot{{\phi}} + {om_0**2:.2f} \sin(\phi) = {om_0**2 * g:.2f} \cos({om:.2f} t)$ \n $\gamma$ = {g}, $\phi_0$ = {ic1[0]}, ${{\dot{{\phi}}}}_0$ = {ic1[1]}")
ax2.set_title(f"DDP with small angle approximation \n $\ddot{{\phi}} + 2 {2*b:.2f} \dot{{\phi}} + {om_0**2:.2f} \phi = {om_0**2 * g:.2f} \cos({om:.2f} t)$ \n $\gamma$ = {g}, $\phi_0$ = {ic1[0]}, ${{\dot{{\phi}}}}_0$= {ic1[1]}")

ani.save(f"ddp_anim_{g}.gif")
plt.show()

