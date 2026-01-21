"""
几何参数定义模块 - 严格遵循文档定义
文档位置: 文档1第1节
"""
# 定义几何参数指标（文档表结构）
GEOMETRIC_PARAMS = {
    "sphericity": {
        "definition": "颗粒表面积与同体积球体表面积之比",
        "ideal_sphere": 1.0,
        "typical_range": "<0.8"
    },
    "flatness": {
        "definition": "短轴/中轴(S/I)",
        "ideal_sphere": 1.0,
        "typical_range": "<0.6 (片状颗粒)"
    },
    "elongation": {
        "definition": "长轴/中轴(L/I)",
        "ideal_sphere": 1.0,
        "typical_range": ">1.5 (杆状颗粒)"
    },
    "roundness": {
        "definition": "轮廓棱角的磨圆程度",
        "ideal_sphere": 1.0,
        "typical_range": "<0.6 (棱角分明颗粒)"
    }
}

# 轴定义（文档术语表）
AXIS_DEFINITIONS = {
    "long_axis": {
        "description": "颗粒的最长尺寸",
        "geometric_meaning": "代表颗粒的最大延伸方向",
        "example": "米粒的长度"
    },
    "intermediate_axis": {
        "description": "颗粒的中间尺寸",
        "geometric_meaning": "描述颗粒的'宽度'",
        "example": "米粒的宽度"
    },
    "short_axis": {
        "description": "颗粒的最短尺寸",
        "geometric_meaning": "代表颗粒的'厚度'或扁平程度",
        "example": "米粒的厚度"
    }
}
