# import numpy as np
# import deepxde as dde
#
#
# if __name__ == '__main__':
#     d = 2
#     precision_train = 10
#     precision_test = 30
#     hard_constraint = True
#     weights = 100  # if hard_constraint == False
#     iterations = 5000
#     parameters = [1e-3, 3, 150, "sin"]
#
#     k0 = dde.Variable(1.)  # Truth is:
#     k1 = 2 * np.pi * d
#
#     if dde.backend.backend_name == "pytorch":
#         sin = dde.backend.pytorch.sin
#     else:
#         from deepxde.backend import tf
#         sin = tf.sin
#
#     learning_rate, num_dense_layers, num_dense_nodes, activation = parameters
#
#     def pde(x, y):
#         dy_xx = dde.grad.hessian(y, x, i=0, j=0)
#         dy_yy = dde.grad.hessian(y, x, i=1, j=1)
#
#         if dde.backend.backend_name == "jax":
#             y = y[0]
#             dy_xx = dy_xx[0]
#             dy_yy = dy_yy[0]
#
#         f = k0 ** 2 * sin(k0 * x[:, 0:1]) * sin(k0 * x[:, 1:2])
#         return -dy_xx - dy_yy - k0 ** 2 * y - f
#
#
#     def func(x):
#         # Reference solution
#         # print(k1)
#         return np.sin(k1 * x[:, 0:1]) * np.sin(k1 * x[:, 1:2])
#
#
#     def transform(x, y):
#         res = x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])
#         return res * y
#
#
#     def boundary(_, on_boundary):
#         return on_boundary
#
#
#     geom = dde.geometry.Rectangle([0, 0], [1, 1])
#     # k0 = 2 * np.pi * d
#     wave_len = 1 / d
#
#     hx_train = wave_len / precision_train
#     nx_train = int(1 / hx_train)
#
#     hx_test = wave_len / precision_test
#     nx_test = int(1 / hx_test)
#
#     if hard_constraint == True:
#         bc = []
#     else:
#         bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)
#
#     data = dde.data.PDE(
#         geom,
#         pde,
#         bc,
#         num_domain=nx_train ** 2,
#         num_boundary=4 * nx_train,
#         solution=func,
#         num_test=nx_test ** 2,
#     )
#
#     net = dde.nn.FNN(
#         [d] + [num_dense_nodes] * num_dense_layers + [1], activation, "Glorot uniform"
#     )
#
#     if hard_constraint == True:
#         net.apply_output_transform(transform)
#
#     model = dde.Model(data, net)
#     optimizer = 'adam'
#
#     if hard_constraint == True:
#         model.compile(optimizer, lr=learning_rate, metrics=["l2 relative error"], external_trainable_variables=[k0])
#     else:
#         loss_weights = [1, weights]
#         model.compile(
#             optimizer,
#             lr=learning_rate,
#             metrics=["l2 relative error"],
#             loss_weights=loss_weights,
#             external_trainable_variables=[k0]
#         )
#
#     variable = dde.callbacks.VariableValue([k0], period=1000)
#     losshistory, train_state = model.train(iterations=10000, callbacks=[variable])
#
#     # dde.saveplot(losshistory, train_state, issave=True, isplot=True)



if __name__ == '__main__':
    """Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
    import deepxde as dde
    import numpy as np
    # Backend tensorflow.compat.v1 or tensorflow
    from deepxde.backend import tf

    # Backend pytorch
    # import torch
    # Backend paddle
    # import paddle
    dde.config.disable_xla_jit()

    C = dde.Variable(2.0)


    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        # Backend tensorflow.compat.v1 or tensorflow
        return (
                dy_t
                - C * dy_xx
                + tf.exp(-x[:, 1:])
                * (tf.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * tf.sin(np.pi * x[:, 0:1]))
        )


    def func(x):
        return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])


    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

    observe_x = np.vstack((np.linspace(-1, 1, num=10), np.full((10), 1))).T
    observe_y = dde.icbc.PointSetBC(observe_x, func(observe_x), component=0)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic, observe_y],
        num_domain=40,
        num_boundary=20,
        num_initial=10,
        anchors=observe_x,
        num_test=10000,
    )

    layer_size = [2] + [32] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)

    model.compile(
        "adam", lr=0.001, external_trainable_variables=C, loss="MSE"
    )
    variable = dde.callbacks.VariableValue(C, period=1000)
    losshistory, train_state = model.train(iterations=10000, callbacks=[variable])

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)