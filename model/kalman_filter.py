import numpy as np


class AdaptiveKalmanFilter:
    """
    自适应卡尔曼滤波器 (针对鱼群计数优化)
    状态向量 x = [当前数量, 变化速度]^T
    """

    def __init__(self, initial_count=0, fps=1.0):
        # 1. 状态向量 [数量, 速度]
        self.x = np.array([[float(initial_count)], [0.0]])

        # 2. 状态协方差矩阵 P (初始不确定性)
        self.P = np.eye(2) * 1.0

        # 3. 状态转移矩阵 F (物理模型：下一刻数量 = 当前数量 + 速度*时间)
        dt = 1.0 / fps
        self.F = np.array([[1.0, dt],
                           [0.0, 1.0]])

        # 4. 观测矩阵 H (我们只能观测到数量，观测不到速度)
        self.H = np.array([[1.0, 0.0]])

        # 5. 过程噪声 Q (物理规律的误差，越小表示越相信物理惯性)
        self.Q = np.array([[0.01, 0.0],
                           [0.0, 0.01]])

        # 6. 观测噪声 R (初始值，会被自适应逻辑覆盖)
        self.R = np.array([[1.0]])

    def predict(self):
        """预测步：根据物理惯性猜下一帧"""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[0, 0]

    def update(self, measurement, confidence_score=None):
        """
        更新步：根据 Transformer 的观测值修正预测
        :param measurement: VIC 模型预测出的鱼群数量 (sum of density)
        :param confidence_score: (可选) 图像置信度。如果图像很浑浊，R 变大
        """
        # === 核心创新：自适应 R (Adaptive R) ===
        # 如果提供了置信度(比如密度图的方差/能量)，动态调整 R
        # 逻辑：confidence 低 (图像差) -> R 变大 -> 不信观测，信预测
        if confidence_score is not None:
            # 这是一个经验公式，你可以根据论文微调
            # 假设 confidence_score 越大越不可信
            base_R = 0.5
            self.R = np.array([[base_R * np.exp(confidence_score)]])
        else:
            self.R = np.array([[0.5]])  # 固定值

        # === 标准卡尔曼公式 ===
        # 1. 计算残差
        y = measurement - np.dot(self.H, self.x)

        # 2. 计算卡尔曼增益 K
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # 3. 更新状态
        self.x = self.x + np.dot(K, y)

        # 4. 更新协方差 P
        I = np.eye(self.F.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)

        # 返回最优估计的数量 (状态向量的第一个元素)
        return self.x[0, 0]