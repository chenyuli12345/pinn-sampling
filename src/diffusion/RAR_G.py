import numpy as np
from deepxde.backend import tf
import deepxde as dde
import torch


def main():
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
    data = dde.data.TimePDE(geomtime, pde, [], num_domain=10, train_distribution='pseudo',
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

    error = losshistory.metrics_test[-1:]

    for i in range(40): #迭代40次
        X = geomtime.random_points(10000) #生成10000个随机点，存储在变量X中
        Y = np.abs(model.predict(X, operator=pde))[:, 0] #使用模型对这些随机点进行预测，结果保存在变量Y中
        err_eq = torch.tensor(Y) #将Y转换为张量
        X_ids = torch.topk(err_eq, 1, dim=0)[1].numpy() #找到张量中最大的一个值的索引，存储在X_ids中
        data.add_anchors(X[X_ids]) #将这个点添加到锚点中

        losshistory, train_state = model.train(epochs=1000) #重新训练模型1000次，并讲训练过程的损失和训练状态保存在losshistory和train_state中
        error.append(losshistory.metrics_test[-1]) #将最后一次的测试误差添加到error中

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    np.savetxt(f'error_RAR-G.txt', error)
    return error


if __name__ == "__main__":
    main()
