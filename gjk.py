import numpy as np


def support(poly_vertices, direction):
    """获取多面体在指定方向上的最远点"""
    max_dot = -np.inf
    support_point = None
    for v in poly_vertices:
        dot = np.dot(v, direction)
        if dot > max_dot:
            max_dot = dot
            support_point = v
    return support_point


def simplex_contains_origin(simplex, direction):
    """判断单纯形是否包含原点，并更新搜索方向"""
    a = simplex[-1]
    ao = -a  # 从a指向原点的向量

    if len(simplex) == 1:  # 线段
        direction[:] = ao
        return False
    elif len(simplex) == 2:  # 三角形
        b = simplex[0]
        ab = b - a
        ab_perp = np.dot(ab, ao) / np.dot(ab, ab) * ab - ao
        if np.dot(ab_perp, ao) > 0:
            direction[:] = ab_perp
        else:
            direction[:] = ao
            simplex.pop(0)
        return False
    else:  # 四面体
        b, c = simplex[0], simplex[1]
        ab = b - a
        ac = c - a
        abc = np.cross(ab, ac)

        if np.dot(abc, ao) > 0:
            direction[:] = abc
            return False
        else:
            # 检查三个面
            for i, (v1, v2) in enumerate([(b, c), (c, a), (a, b)]):
                face_normal = np.cross(v2 - v1, a - v1)
                if np.dot(face_normal, ao) > 0:
                    direction[:] = face_normal
                    simplex.pop(i)
                    return False
            return True  # 原点在四面体内


def gjk_collision(poly_a, poly_b):
    """检测两个多面体是否碰撞"""
    # 初始化搜索方向（例如从A的第一个点指向B的第一个点）
    direction = np.array(poly_b[0]) - np.array(poly_a[0])
    if np.all(direction == 0):
        direction = np.array([1.0, 0.0, 0.0])  # 避免零向量

    simplex = []
    while True:
        # 获取支持点
        support_a = support(poly_a, direction)
        support_b = support(poly_b, -direction)
        support_point = support_a - support_b

        # 如果支持点与方向相反，则无碰撞
        if np.dot(support_point, direction) <= 0:
            return False

        simplex.append(support_point)

        # 检查单纯形是否包含原点
        if simplex_contains_origin(simplex, direction):
            return True
