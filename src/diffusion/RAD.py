import numpy as np
from deepxde.backend import tf
import deepxde as dde


def main(c, k):
    NumDomain = 30

    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, j=1)
        dy_xx = dde.grad.hessian(y, x, j=0)
        return (
                dy_t
                - dy_xx
                + tf.exp(-x[:, 1:])
                * (tf.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * tf.sin(np.pi * x[:, 0:1]))
        )

    def func(x):
        return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    data = dde.data.TimePDE(geomtime, pde, [], num_domain=NumDomain, train_distribution='pseudo',
                            solution=func, num_test=10000)

    layer_size = [2] + [32] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    def output_transform(x, y):
        return tf.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y
    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=10000)

    error = losshistory.metrics_test.copy()

    for i in range(40):
        X = geomtime.random_points(10000) #生成10000个随机点，存储在变量X中
        Y = np.abs(model.predict(X, operator=pde)) #使用模型对这些随机点进行预测，结果保存在变量Y中
        err_eq = np.power(Y, k) / np.power(Y, k).mean() + c #计算误差，具体为残差的k次方除以残差的k次方的均值再加上c
        err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        X_ids = np.random.choice(a=len(X), size=NumDomain, replace=False, p=err_eq_normalized)
        X_selected = X[X_ids]
        data.replace_with_anchors(X_selected)

        losshistory, train_state = model.train(epochs=1000)
        error.append(losshistory.metrics_test[-1])

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    np.savetxt(f'error_RAD_c_{c}_k_{k}.txt', error)
    return error


if __name__ == "__main__":
    main(c=1, k=1)
