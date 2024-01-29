'''
    By Cathayana
    这个程序可能会很卡(可能会卡掉整个 notebook),考虑单独运行 杂化轨道.py
    这个例子以上面的 sp2 杂化轨道为例，但同样适用于其他杂化轨道可视化
    程序中各参数已调整到最适合展示的程度
'''

import numpy as np
import plotly.graph_objects as go

a_0 = 1

def psi_1s(x, y, z, Z):
    r = np.sqrt(x**2 + y**2 + z**2)
    normalization_factor = (Z**3 / (np.pi * a_0**3))**0.5
    return normalization_factor * np.exp(-Z * r / a_0)

def psi_2s(x, y, z, Z):
    r = np.sqrt(x**2 + y**2 + z**2)
    normalization_factor = (Z**3 / (8 * np.pi * a_0**3))**0.5
    return normalization_factor * (1 - Z * r / (2 * a_0)) * np.exp(-Z * r / (2 * a_0))

def psi_2px(x, y, z, Z):
    r = np.sqrt(x**2 + y**2 + z**2)
    normalization_factor = (Z**5 / (32 * np.pi * a_0**5))**0.5
    return normalization_factor * (x / r) * np.exp(-Z * r / (2 * a_0))

def psi_2py(x, y, z, Z):
    r = np.sqrt(x**2 + y**2 + z**2)
    normalization_factor = (Z**5 / (32 * np.pi * a_0**5))**0.5
    return normalization_factor * (y / r) * np.exp(-Z * r / (2 * a_0))

def psi_2pz(x, y, z, Z):
    r = np.sqrt(x**2 + y**2 + z**2)
    normalization_factor = (Z**5 / (32 * np.pi * a_0**5))**0.5
    return normalization_factor * (z / r) * np.exp(-Z * r / (2 * a_0))

# 创建网格
X, Y, Z = np.mgrid[-2:2:60j, -2:2:60j, -2:2:60j]

# 定义sp²杂化轨道
psi_1 = np.sqrt(1/3) * psi_2s(X, Y, Z, 9) + np.sqrt(2/3) * psi_2px(X, Y, Z, 5)
psi_2 = np.sqrt(1/3) * psi_2s(X, Y, Z, 9) - np.sqrt(1/6) * psi_2px(X, Y, Z, 5) + np.sqrt(1/2) * psi_2py(X, Y, Z, 5)
psi_3 = np.sqrt(1/3) * psi_2s(X, Y, Z, 9) - np.sqrt(1/6) * psi_2px(X, Y, Z, 5) - np.sqrt(1/2) * psi_2py(X, Y, Z, 5)

# 定义颜色映射
colorscale = [[0, 'blue'], [0.5, 'rgba(0,0,0,0)'], [1, 'red']]

# 创建杂化轨道的可视化
for psi, name in zip([psi_1, psi_2, psi_3], ['psi_1', 'psi_2', 'psi_3']):
    fig = go.Figure()

    fig.add_trace(go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=psi.flatten(),
        isomin=-0.2,
        isomax=0.2,
        opacity=0.1,
        surface_count=30,
        colorscale=colorscale
    ))

    fig.update_layout(
        title=f"sp² Hybrid Orbital Visualization: {name}",
        margin=dict(t=0, l=0, b=0, r=0)
    )

    fig.show()
