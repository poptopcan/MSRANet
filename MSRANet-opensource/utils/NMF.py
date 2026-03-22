import numpy as np

#数据降维
def nmf_kl_variants(V, k, max_iter=200, tol=1e-4, variant='standard'):
    """
    基于KL散度的NMF实现（支持两种变体）
    参数:
        V: 输入非负矩阵 (n×m)
        k: 降维维度
        max_iter: 最大迭代次数
        tol: 收敛阈值
        variant: 'standard' 或 'alternative'
    返回:
        W: 基矩阵 (n×k)
        H: 系数矩阵 (k×m)
        loss_history: 损失变化历史
    """
    n, m = V.shape
    W = np.random.rand(n, k) * 0.1
    H = np.random.rand(k, m) * 0.1
    loss_history = []
    
    for i in range(max_iter):
        WH = W @ H
        # 选择KL变体
        if variant == 'standard':
            # 原始KL散度更新规则
            H *= (W.T @ (V / (WH + 1e-9))) / (W.T @ np.ones((n, m)) + 1e-9)
            W *= (((V / (WH + 1e-9)) @ H.T)) / (np.ones((n, m)) @ H.T + 1e-9)
            # 计算损失
            loss = np.sum(V * np.log(V / (WH + 1e-9)) - V + WH)
        else:
            # 改进KL散度（I-divergence）更新规则
            H *= (W.T @ (V / (WH + 1e-9))) / (W.T @ np.ones((n, m)) + 1e-9)
            W *= (((V / (WH + 1e-9)) @ H.T)) / (np.ones((n, m)) @ H.T + 1e-9)
            # 改进损失计算
            loss = np.sum(V * ((WH / (V + 1e-9)) - np.log(WH / (V + 1e-9)) - 1))
        
        loss_history.append(loss)
        if i > 10 and abs(loss_history[-1] - loss_history[-2]) < tol:
            break
    
    return W, H, loss_history