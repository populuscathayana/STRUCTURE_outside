import numpy as np
import plotly.graph_objects as go
import sympy as sp

# 定义球谐函数
def Y_lm(l, m, theta, phi):
    return sp.Ynm(l, m, theta, phi).expand(func=True)

# 符号变量
theta, phi = sp.symbols('theta phi')

# d轨道的球谐函数
Y_22 = Y_lm(2, 2, theta, phi)
Y_2m2 = Y_lm(2, -2, theta, phi)

# 创建网格
x_range = np.linspace(-1.5, 1.5, 100)
y_range = np.linspace(-1.5, 1.5, 100)
z_range = np.linspace(-1.5, 1.5, 100)
x, y, z = np.meshgrid(x_range, y_range, z_range)

# 将笛卡尔坐标转换为球坐标
r = np.sqrt(x**2 + y**2 + z**2)
theta = np.arccos(z/r)
phi = np.arctan2(y, x)

# 计算波函数值
Y_22_vals = np.nan_to_num(sp.lambdify((theta, phi), Y_22, 'numpy')(theta, phi))
Y_2m2_vals = np.nan_to_num(sp.lambdify((theta, phi), Y_2m2, 'numpy')(theta, phi))

# 波函数的模平方
density_22 = np.abs(Y_22_vals)**2
density_2m2 = np.abs(Y_2m2_vals)**2

# 选择一个等值水平
iso_level = 0.02  # 可以根据需要调整这个值

# 使用plotly绘制等值面
fig = go.Figure()

# d_{x^2-y^2} 轨道
fig.add_trace(go.Isosurface(
    x=x.flatten(), y=y.flatten(), z=z.flatten(),
    value=density_22.flatten(),
    isomin=iso_level,
    isomax=iso_level,
    surface_count=1, # 只显示一个等值面
    colorscale='Viridis',
    caps=dict(x_show=False, y_show=False, z_show=False)
))

# d_{xy} 轨道
fig.add_trace(go.Isosurface(
    x=x.flatten(), y=y.flatten(), z=z.flatten(),
    value=density_2m2.flatten(),
    isomin=iso_level,
    isomax=iso_level,
    surface_count=1, # 只显示一个等值面
    colorscale='Viridis',
    caps=dict(x_show=False, y_show=False, z_show=False)
))

fig.update_layout(title='d Orbitals Isosurfaces')
fig.show()
