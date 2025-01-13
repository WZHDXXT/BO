# 1.导入库
import numpy as np
# 用于统计函数的库
import scipy.stats as sps
# 加载鸢尾花数据集的函数
from sklearn.datasets import load_iris
# Gaussian Process Regression
from sklearn.gaussian_process import GaussianProcessRegressor
# Kernel Function
from sklearn.gaussian_process.kernels import Matern
from sklearn.svm import SVC

def main():
    # 3.定义边界函数
    # 每行对应一个超参数，每列对应该超参数的下界和上界，限制超参数的搜索空间
    bounds = np.array([[1e-3, 1e3], [1e-5, 1e-1]])

    x_samples = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(15, bounds.shape[0]))
    y_samples = np.random.rand(15)


    # 5.初始化样本和Surrogate Function
    # x_samples 是从由 bounds 数组定义的搜索空间中随机抽样的初始点。y_samples 是这些初始点对应的目标函数评估。
    # 6.运行贝叶斯优化循环
    iters = 10
    for i in range(iters):
        gp = GaussianProcessRegressor()
        # 现成样本更新高斯模型
        gp.fit(x_samples, y_samples)
        x_next = None
        best_acq_value = -np.inf
        # 在参数空间生成大量随机点x_random_points优化获取函数，选择下一个由目标函数评估的样本
        n_random_points = 10000
        x_random_points = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_random_points, bounds.shape[0]))
        # 所有的EI
        acq_values = np.array([acquisition(x, y_samples) for x in x_random_points])
        max_acq_index = np.argmax(acq_values)
        max_acq_value = acq_values[max_acq_index]

        if max_acq_value > best_acq_value:
            best_acq_value = max_acq_value
            x_next = x_random_points[max_acq_index]
        
        print(f"{i+1}: next x is {x_next} value is {max_acq_value}")
        y_next = objective(x_next)
        x_samples = np.vstack((x_samples, x_next))
        y_samples = np.append(y_samples, y_next)
    # 7.返回结果
    best_index = np.argmin(y_samples)   
    best_x = x_samples[best_index]   
    best_y = y_samples[best_index]   

    print(f"Best parameters: C={best_x[0]}, gamma={best_x[1]}")   
    print(f"Best accuracy: {best_y}")


# 4.定义获取函数 Acquisition Function
def acquisition(x, y_samples):
    # 高斯过程在x处预测均值和标准差
    gp = GaussianProcessRegressor()
    mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
    # 初始样本
    f_best = np.min(y_samples)
    improvement = f_best - mu
    with np.errstate(divide='warn'):
        Z = improvement / sigma if sigma > 0 else 0
        ei = improvement * sps.norm.cdf(Z) + sigma * sps.norm.pdf(Z)
        ei[sigma == 0.0] == 0.0
    return ei

# 2.定义目标函数
def objective(params):
    # C是正则化参数，gamma是RBF、poly和sigmoid核的核系数
    C, gamma = params
    X, y = load_iris(return_X_y = True)
    np.random.seed(0)
    indices = np.random.permutation(len(X))
    X_train = X[indices[:100]]
    y_train = y[indices[:100]]
    X_test = X[indices[100:]]
    y_test = y[indices[100:]]
    # 训练支持向量分类器，返回测试集上的负准确性
    clf = SVC(C=C, gamma=gamma)
    clf.fit(X_train, y_train)
    return -clf.score(X_test, y_test)
main()