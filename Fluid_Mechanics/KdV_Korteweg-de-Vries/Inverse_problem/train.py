import deepxde as dde
from data import *
from net import *
from pinn_solver import pde, get_bc_ic

# Generate data
gen_exact_solution()
observe_x, y = gen_testdata()

# Define geometry and time domain
geom = dde.geometry.Interval(-20, 20)
timedomain = dde.geometry.TimeDomain(0, 4)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Define PDE problem
data = dde.data.TimePDE(
    geomtime,
    pde,
    get_bc_ic(geomtime, observe_x, y),
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
    num_test=2540,
)

# Build and train the model
net = create_network()
model = dde.Model(data, net)
model.compile("adam", lr=1e-3, external_trainable_variables=[b, rho1, rho2])
variable = dde.callbacks.VariableValue([b, rho1, rho2], period=1, filename="./test2.txt", precision=6)

losshistory, train_state = model.train(iterations=25000, callbacks=[variable])

# Save and plot results
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
