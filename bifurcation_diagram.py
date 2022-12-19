
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as si

t0, tf = 0, 400 #Initial and final time
dt = 0.01
times = np.arange(t0, tf, dt)

#Defines DE parameters
om = 2*np.pi #Dampening Frequency
om_0 = 1.5*om #Omega
b = om_0/4 #Beta

#Initial Values
ic = np.array([-np.pi/2,0])

#Defines the ODE as a system of differential equations.
#Note: g is the value gamma that the bifurcation diagram is based on.
def flux_ode(t, y, om, om_0, b, g):
    return np.array(
        [
            y[1], 
            g*(om_0**2)*np.cos(om*t) - 2*b*y[1] - (om_0**2)*np.sin(y[0])
            ]
        )

# The array containing the values for gamma:
gspace = np.arange(1.060, 1.0885, 0.0005)
# gspace=np.array([1.065, 1.075, 1.085])

Y_data = []
L = len(gspace)

phi = 0 # if 0 will graph the angle diagram if 1 will graph the angular velocity

#solves the DE at each value of gamma and stores the solution in an array
for i,g in enumerate(gspace):
    yi = si.odeint(
        flux_ode,
        ic,
        times,
        args=(om, om_0, b, g),
        tfirst=True,
        rtol=1e-6,
        atol=1e-6,
        mxstep=1000
    )
    Y_data.append(yi[:,phi])
    print(f"{i+1}/{L}") #This only counts how many iterations have been done. its to show that the code is running

#The solution array is transposed so all the solutions for a particular value of gamma 
#can be graphs along with that value
D = np.array(Y_data).transpose()
f = int(1/dt)

st = 100 #looks at all the points between
for data in D[(tf-st)*f:tf*f:tf]:
    plt.scatter(gspace, data, c="k", s=0.4) 

plt.title(f"Bifurcation diagram for t={tf-st}...{tf}")
plt.xlabel("$\gamma$")
plt.ylabel("$\phi'(t)$" if phi else "$\phi(t)$")

plt.show()