import numpy as np


def gjk_collision(poly1, poly2, tolerance=1e-8):
    """修复版GJK碰撞检测（优化边界判断逻辑）"""
    poly1 = np.array(poly1, dtype=np.float64)
    poly2 = np.array(poly2, dtype=np.float64)

    def support(direction):
        """计算闵可夫斯基差集的支撑点"""
        if np.linalg.norm(direction) < tolerance:
            direction = np.array([1.0, 0.0, 0.0])  # 避免零向量
        direction = direction / np.linalg.norm(direction)

        # 分别找两个多面体的支撑点
        p1 = poly1[np.argmax(np.dot(poly1, direction))]
        p2 = poly2[np.argmin(np.dot(poly2, direction))]
        return p1 - p2

    # 初始化
    direction = np.array([1.0, 0.0, 0.0])
    simplex = [support(direction)]
    direction = -simplex[0]

    iteration_count = 0
    while iteration_count < 50:
        iteration_count += 1
        A = support(direction)

        # 若新点与原点的投影小于等于0，说明无碰撞
        if np.dot(A, direction) < -tolerance:  # 放宽边界判断
            return False

        simplex.append(A)

        if process_simplex(simplex, direction, tolerance):
            return True
    return False


def process_simplex(simplex, direction, tolerance):
    if len(simplex) == 2:
        return handle_line(simplex, direction, tolerance)
    elif len(simplex) == 3:
        return handle_triangle(simplex, direction, tolerance)
    elif len(simplex) >= 4:
        return handle_tetrahedron(simplex, direction, tolerance)
    return False


def handle_line(simplex, direction, tolerance):
    A, B = simplex[-1], simplex[-2]
    AB = B - A
    AO = -A

    if np.dot(AB, AO) > tolerance:
        # 原点在AB线段正方向区域
        direction[:] = np.cross(np.cross(AB, AO), AB)
    else:
        # 原点在A点方向
        simplex[:] = [A]
        direction[:] = AO
    return False


def handle_triangle(simplex, direction, tolerance):
    A, B, C = simplex[-1], simplex[-2], simplex[-3]
    AB = B - A
    AC = C - A
    AO = -A

    ABC = np.cross(AB, AC)

    if np.dot(ABC, AO) > tolerance:
        direction[:] = ABC
        return False

    AB_perp = np.cross(AB, ABC)
    if np.dot(AB_perp, AO) > tolerance:
        simplex[:] = [A, B]
        direction[:] = np.cross(np.cross(AB, AO), AB)
        return False

    AC_perp = np.cross(ABC, AC)
    if np.dot(AC_perp, AO) > tolerance:
        simplex[:] = [A, C]
        direction[:] = np.cross(np.cross(AC, AO), AC)
        return False

    return True


def handle_tetrahedron(simplex, direction, tolerance):
    A, B, C, D = simplex[-4], simplex[-3], simplex[-2], simplex[-1]

    # 计算各面法向量（指向外部）
    faces = [
        (B - A, C - A, D - A),  # ABC面
        (C - A, D - A, B - A),  # ACD面
        (D - A, B - A, C - A),  # ADB面
        (C - B, D - B, A - B)  # BCD面
    ]

    # 检查原点是否在四面体内部
    inside = True
    for vectors in faces:
        normal = np.cross(vectors[0], vectors[1])
        # 确保法向量指向外部
        if np.dot(normal, vectors[2]) < 0:
            normal = -normal

        # 若原点在面的外部，则更新方向并缩小单纯形
        if np.dot(normal, -A) < -tolerance:  # 放宽边界判断
            inside = False
            direction[:] = normal
            # 保留与该面相关的三个点
            simplex[:] = [A, B, C] if vectors[2] is D - A else [A, C, D]
            break

    return inside


if __name__ == "__main__":
    # =============== 测试用例1：两个相交的四面体（碰撞）===============
    tetra1 = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    tetra2 = [(0.5, 0.5, 0.5), (1.5, 0.5, 0.5), (0.5, 1.5, 0.5), (0.5, 0.5, 1.5)]
    print("测试用例1（相交）：", gjk_collision(tetra1, tetra2))  # 预期True

    # =============== 测试用例2：两个分离的立方体（不碰撞）===============
    cube1 = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    cube2 = [(2, 0, 0), (3, 0, 0), (3, 1, 0), (2, 1, 0), (2, 0, 1), (3, 0, 1), (3, 1, 1), (2, 1, 1)]
    print("测试用例2（分离）：", gjk_collision(cube1, cube2))  # 预期False

    # =============== 测试用例3：两个相切的球体（碰撞）===============
    # 创建球体1（中心在原点）
    angles = np.linspace(0, 2 * np.pi, 20)
    sphere1 = []
    for theta in angles:
        for phi in angles:
            x = np.cos(theta) * np.sin(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(phi)
            sphere1.append((x, y, z))

    # 创建球体2（与球体1相切）
    sphere2 = [(x + 1, y, z) for x, y, z in sphere1]  # 沿x轴平移1个单位

    print("测试用例3（相切）：", gjk_collision(sphere1, sphere2))  # 预期True

    # =============== 测试用例4：完全重叠的物体（碰撞）===============
    print("测试用例4（重叠）：", gjk_collision(tetra1, tetra1))  # 预期True

    # =============== 测试用例5：点接触（碰撞）===============
    point = [(0.5, 0.5, 0.5)]
    print("测试用例5（点接触）：", gjk_collision(tetra1, point))  # 预期True
