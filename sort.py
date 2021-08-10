import numpy as np
import scipy


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
            返回预测的向量和协方差矩阵，未被观测的速度设置为0
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
        # x = F * x
        mean = np.dot(self._motion_mat, mean)
        # P = F * P * F_T + Q
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """分布转换.

        Parameters
        ----------
        mean : ndarray
             大小8的状态向量
        covariance : ndarray
            大小为8*8的协方差矩阵

        Returns
        -------
        (ndarray, ndarray)
            校正后的向量(4)
            矫正后的协方差矩阵(4*4)

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        # u = H * x
        mean = np.dot(self._update_mat, mean)
        # E = H * P * H_T
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))

        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """更新结果

        Parameters
        ----------
        mean : ndarray
            长为8的输入向量
        covariance : ndarray
            大小为8*8的协方差矩阵.
        measurement : ndarray
            长为4的向量(x,y,a,h)，该向量是测量结果,其中中心位置 (x, y),长宽比 a,高度 h,

        Returns
        -------
        (ndarray, ndarray)
            返回矫正后的值

        """
        projected_mean, projected_cov = self.project(mean, covariance)
        # 此处将求 x =  (a^-1) * b 转换为 a * x = b,求解方程x的形式
        # 其中a对应的为 projected_cov, b对应 covariance * self._update_mat.T
        # 由于不是a*x=b的形式，所以等式两边同时转置再求解，最后再转置得到结果kalman_gain
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))

        # 我原认为new_covariance求解方式为下面的，
        # 但是发现这两个等式是等价的，其中K′=Pk*Hk_t(Hk*Pk*Hk_t+Rk)^−1
        # 于是 projected_cov* kalman_gain.T( (Hk*Pk*Hk_t+Rk)*K'_T ) 刚好等于 self._update_mat * covariance（Hk*Pk）
        # 这样做还可以提升效率
        # new_covariance = covariance - np.linalg.multi_dot((
        #     kalman_gain,  self._update_mat, covariance))

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """计算概率分布和测量值的距离.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha

if __name__ == '__main__':
    kalman = KalmanFilter()
