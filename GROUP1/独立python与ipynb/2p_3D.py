import plotly.express as px
import numpy as np
import plotly.graph_objects as go
# 常数
a0 = 1  # 一个单位的玻尔半径

# 定义网格
r = np.linspace(0, 20*a0, 200)
theta = np.linspace(0, np.pi, 150)
phi = np.linspace(0, 2*np.pi, 150)
r, theta, phi = np.meshgrid(r, theta, phi)

# 波函数
R_2p = (1/np.sqrt(32*np.pi*a0**3)) * r * np.exp(-r/(2*a0))
Y_10 = np.sqrt(3/(4*np.pi)) * np.abs(np.cos(theta))
psi_2pz = R_2p * Y_10

# 概率密度
probability_density = np.abs(psi_2pz)**2

# 转换为笛卡尔坐标
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

# 展平数组
x = x.flatten()
y = y.flatten()
z = z.flatten()
probability_density = probability_density.flatten()

color_scale = px.colors.sequential.Viridis

# 定义阈值
threshold = 0.0006

# 仅选择大于阈值的点进行绘制
mask = probability_density > threshold
x_selected = x[mask]
y_selected = y[mask]
z_selected = z[mask]
prob_density_selected = probability_density[mask]

# 创建图形
fig = go.Figure(data=go.Scatter3d(
    x=x_selected,
    y=y_selected,
    z=z_selected,
    mode='markers',
    marker=dict(
        size=4,
        color=prob_density_selected,                # 设置颜色为概率密度
        colorscale='Viridis',   # 选择颜色范围
        opacity=0.005,      # 设置全局透明度
        colorbar_title='Probability Density'
    )
))

# 添加标题和坐标轴标签
fig.update_layout(title='Approximate 2p_z Orbital of Hydrogen Atom',
                scene=dict(xaxis_title='X',
                            yaxis_title='Y',
                            zaxis_title='Z'))

# 显示图形
fig.show()