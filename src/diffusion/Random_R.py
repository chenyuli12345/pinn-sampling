import numpy as np
from deepxde.backend import tf
import deepxde as dde


def main(P):
    NumDomain = 30 #区域的数量

    #定义一个PDE
    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, j=1) #计算y关于x的雅可比矩阵的第二列，即y关于x的第二个变量（t）的偏导数
        dy_xx = dde.grad.hessian(y, x, j=0) #计算y关于x的Hessian矩阵的第一列，即y关于x的第一个变量（x）的二阶偏导数

        #x[:, 0:1]表示x的第一列即x，x[:, 1:]表示x的第二列即t
        return (dy_t - dy_xx + tf.exp(-x[:, 1:]) * (tf.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * tf.sin(np.pi * x[:, 0:1]))) #返回PDE的表达式

    def func(x):
        return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:]) #返回解析解

    geom = dde.geometry.Interval(-1, 1) #定义一个一维的空间区域，区域的范围是[-1, 1]
    timedomain = dde.geometry.TimeDomain(0, 1) #定义一个时间区域，时间区域的范围是[0, 1]
    geomtime = dde.geometry.GeometryXTime(geom, timedomain) #将空间区域和时间区域组合成一个空间时间区域
    data = dde.data.TimePDE(geomtime, pde, [], num_domain=NumDomain, train_distribution='pseudo', solution=func, num_test=10000) #定义一个时空pde问题
    #第一个参数表示空间时间区域，第二个参数表示PDE的表达式，第三个参数表示边界条件，第四个参数表示区域内训练点数量，第五个参数表示训练数据的分布为伪随机，第六个参数表示解析解，第七个参数表示测试点的数量

    layer_size = [2] + [32] * 3 + [1] #神经网络的层结构，第一层有两个神经元，中间有三层每层32个神经元，最后一层一个神经元
    activation = "tanh" #激活函数
    initializer = "Glorot uniform" #初始化方法为Glorot均匀初始化方法，用来减少梯度消失和梯度爆炸的问题
    net = dde.maps.FNN(layer_size, activation, initializer) #创建一个前馈神经网络，包含层结构，激活函数和初始化方法

    def output_transform(x, y):
        return tf.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y
    net.apply_output_transform(output_transform)

    model = dde.Model(data, net) #创建一个模型，包含pde问题和神经网络结构

    resampler = dde.callbacks.PDEResidualResampler(period=P) #定义回调函数，用于在训练过程中重新采样PDE残差较大的点
    model.compile("adam", lr=1e-3, metrics=["l2 relative error"]) #编译模型，使用adam优化器，学习率为1e-3，评价指标为l2相对误差
    losshistory, train_state = model.train(epochs=15000, callbacks=[resampler]) #训练模型，迭代15000次，使用回调函数

    error = np.array(losshistory.metrics_test)[-1]
    print("L2 relative error:", error)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    return error


if __name__ == '__main__':
    main(P=100)
