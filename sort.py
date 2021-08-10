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
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
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

if __name__ == '__main__':
    kalman = KalmanFilter()
