import numpy as np


class KalmanFilter(object):
    """
    输入为
        x, y, a, h, vx, vy, va, vh

    中心位置 (x, y),
    长宽比 a,
    高度 h,
    速度 (vx, vy),

    假设物体运动是遵循恒速模型。
    边界框位置（x，y，a，h）通过观察直接获得。
    """

    def __init__(self):
        # ndim 输入数据维度默认为4
        # dt 时间的变化量默认为1
        ndim, dt = 4, 1.

        # 初始化kalman的矩阵
        # motion_mat 运动矩阵8*8.

        self._motion_mat = np.eye(ndim * 2) + np.diag([dt] * ndim, ndim)
        # self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        # for i in range(ndim):
        #     self._motion_mat[i, ndim + i] = dt
        # 注释写法为官方实现，上方我的写法，我感觉这样写更简练。都是为了创建如下矩阵F
        #    [[1, 0, 0, 0, 1, 0, 0, 0],
        #     [0, 1, 0, 0, 0, 1, 0, 0],
        #     [0, 0, 1, 0, 0, 1, 1, 0],
        #     [0, 0, 0, 1, 0, 0, 0, 1],
        #     [0, 0, 0, 0, 1, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 1, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 1, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 1]]

        # 创建如下跟新矩阵
        #    [[1, 0, 0, 0, 1, 0, 0, 0],
        #     [0, 1, 0, 0, 0, 1, 0, 0],
        #     [0, 0, 1, 0, 0, 1, 1, 0],
        #     [0, 0, 0, 1, 0, 0, 0, 1],
        self._update_mat = np.eye(ndim, 2 * ndim)

        # 创建位置标准差和速度标准擦用来初始化协方差矩阵
        # 这个值为预先设定好的来代表不确定性的值
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """初始化一个轨迹.
        Parameters
        ----------
        measurement : ndarray
            测量出的边界框 Bounding box，
            具体包含信息如下：
            中心位置 (x, y),
            长宽比 a,
            高度 h,

        Returns
        -------
        (ndarray, ndarray)
            第一个返回值为一个8维的状态向量
            第二个返回值为一个8*8的协方差矩阵，为对角矩阵
        """
        mean_pos = measurement
        # 最开始速度信息初始化为0
        mean_vel = np.zeros_like(mean_pos)
        # 按行去连接两个向量
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def aas(self,a,b):
        """

        """

        return b
    def predict(self, mean, covariance):

        """通过kalman滤波预测.

        Parameters
        ----------
        mean : ndarray
            一个8维的状态向量
        covariance : ndarray
           一个8*8的协方差矩阵，为对角矩阵


        Returns
        -------
        (ndarray, ndarray)
            返回预测的向量和协方差矩阵，违背观测的速度设置为0
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]

        # 按行去连接两个向量
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance


if __name__ == '__main__':
    kalman = KalmanFilter()
