# hydrogen_atom_2d.py

import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# ...[其他导入，和您提供的代码中的导入相同]

def page():
    st.title("Hydrogen Atom Wavefunction Visualization")
    # 符号和函数的定义
    # 定义符号
    r, theta, phi, Z, xi = sp.symbols('r theta phi Z xi')

    # 定义Laguerre多项式
    def L(n, alpha, x_expr):
        x = sp.symbols('x')  # 定义一个新的符号x，用于临时替换
        laguerre_unsub = x**(-alpha) * sp.factorial(n)**(-1) * sp.diff(x**(n + alpha) * sp.exp(-x), x, n) * sp.exp(x)
        laguerre = laguerre_unsub.subs(x, x_expr)  # 使用原始表达式替换临时的x
        return sp.simplify(laguerre)

    # 定义径向波函数
    def R(n, l, r, Z):
        x = 2*Z*r/n
        front = sp.sqrt((2*Z/n)**3 * sp.factorial(n-l-1)/(2*n*sp.factorial(n+l)))
        laguerre = L(n-l-1, 2*l+1, x)
        return front * sp.exp(-Z*r/n) * (2*Z*r/n)**l * laguerre

    # 定义关联勒让德多项式
    def P(l, m, x_expr):
        x = sp.symbols('x')  # 定义一个新的符号x，用于临时替换
        p_unsub = (1-x**2)**(abs(m)/2) * sp.diff((x**2-1)**l, x, l + abs(m)) / (2**l * sp.factorial(l))
        p = p_unsub.subs(x, x_expr)  # 使用原始表达式替换临时的x
        return sp.simplify(p)

    # 定义球谐函数
    def YY(l, m, theta, phi):
        P_lm = P(l, m, sp.cos(theta))
        return sp.sqrt((2*l+1)/(4*sp.pi) * sp.factorial(l-abs(m))/sp.factorial(l+abs(m))) * P_lm * sp.exp(sp.I*m*phi)
    # Streamlit UI的创建
    n = st.number_input('Enter principal quantum number n:', min_value=1, max_value=10, value=1, step=1)
    l = st.number_input('Enter azimuthal quantum number l:', min_value=0, max_value=n-1, value=0, step=1)
    m = st.number_input('Enter magnetic quantum number m:', min_value=-l, max_value=l, value=0, step=1)
    # 波函数的计算与可视化
    # [计算R(n,l,r) 和 Y(l,m,theta,phi) 的部分，以及相关图形的部分
    # 计算表达式
    R_exp = R(n, l, r, Z)
    Y_exp = YY(l, m, theta, phi)
    # 输出
    st.latex(f"R_{{nl}}(r) = {sp.latex(R_exp)}")
    st.latex(f"Y_{{lm}}(\\theta, \\phi) = {sp.latex(Y_exp)}")
    Z_val = 1
    # 替换Z的值
    R_exp_sub = R_exp.subs(Z, Z_val)
    Y_exp_sub = Y_exp.subs(Z, Z_val)

    # 使用lambdify转换符号表达式为数值函数
    variables_R = (r, theta, phi)
    variables_Y = (theta, phi)
    R_num = sp.lambdify(variables_R, R_exp_sub, 'numpy')
    Y_num = sp.lambdify(variables_Y, Y_exp_sub, 'numpy')


    # 函数来计算不同平面的概率密度
    def calculate_probability_density(plane, X, Y, R_num, Y_num):
        R = np.sqrt(X**2 + Y**2)
        if plane == "xy":
            Theta = np.pi / 2 * np.ones_like(R)
            Phi = np.arctan2(Y, X)
        elif plane == "xz":
            Theta = np.arctan2(np.sqrt(Y**2), X) # 这里原来使用的是Y，检查是否应该是Y
            Phi = np.zeros_like(R)
        elif plane == "yz":
            Theta = np.arctan2(np.sqrt(X**2), Y) # 这里原来使用的是X，检查是否应该是X
            Phi = np.pi/2 * np.ones_like(R)
        else:
            raise ValueError("Invalid plane specifier. Choose from 'xy', 'xz', or 'yz'.")
        Psi = R_num(R, Theta, Phi) * Y_num(Theta, Phi)
        return np.abs(Psi)**2



    # 用户输入界限值
    x_max = st.number_input('Maximum x', value=12)
    x_min = x_max * -1
    y_max = st.number_input('Maximum y', value=12)
    y_min = y_max * -1

    # 生成网格
    x = np.linspace(x_min, x_max, 700)
    y = np.linspace(y_min, y_max, 700)
    X, Y = np.meshgrid(x, y)


    # 计算概率密度
    Prob_xy = calculate_probability_density("xy", X, Y, R_num, Y_num)

    # 计算概率密度的径向平均值
    r_values = np.linspace(0, x_max, 700)
    prob_r = np.array([np.mean(Prob_xy[(X**2 + Y**2 >= r1**2) & (X**2 + Y**2 < r2**2)])
                    for r1, r2 in zip(r_values[:-1], r_values[1:])])
    r_values = 0.5 * (r_values[:-1] + r_values[1:])

    # 使用plotly绘制概率密度与r的图形
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=r_values, y=prob_r, mode='lines', name='Prob vs. r'))
    fig_r.update_layout(
        title='Probability Density vs. r in xy-plane',
        xaxis_title='r',
        yaxis_title='Probability Density',
        xaxis=dict(constrain='domain',type='log',  # 设置y轴为对数标度
            range=[np.log10(0.1), np.log10(100)]),
        yaxis=dict(
            type='log',  # 设置y轴为对数标度
            autorange=True
        )
    )
    st.plotly_chart(fig_r)

    fig_r2 = go.Figure()
    fig_r2.add_trace(go.Scatter(x=r_values, y=prob_r*4*np.pi*r_values**2, mode='lines', name='Prob vs. r'))
    fig_r2.update_layout(
        title='Probability vs. r in xy-plane',
        xaxis_title='r',
        yaxis_title='Probability',
        xaxis=dict(constrain='domain',
            autorange=True),
        yaxis=dict(
            autorange=True
        )
    )
    st.plotly_chart(fig_r2)













    # 等值线的起始值、终止值和步长
    start = 0
    def draw_density_contour(Prob, x, y, title, xaxis_title, yaxis_title):
        end = np.max(Prob)
        size = end/200
        fig = go.Figure(data=[go.Contour(z=Prob, x=x, y=y, colorscale='Viridis',
                                        contours=dict(start=start, end=end, size=size),
                                        line=dict(color='white', width=0.1))])
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            autosize=False,
            width=600,
            height=600,
            margin=dict(l=60, r=60, b=60, t=60)
        )
        st.plotly_chart(fig)
    # 绘制不同平面的图形
    for plane, yaxis_title in [("xy", "y"), ("xz", "z"), ("yz", "z")]:
        Prob = calculate_probability_density(plane, X, Y, R_num, Y_num)
        draw_density_contour(Prob, x, y, f"Probability Density in the {plane}-Plane", plane[0], yaxis_title)
    if st.button('Show Code'):
        with open(__file__, 'r') as file:
            code = file.read()
        st.markdown("```python\n" + code + "\n```", unsafe_allow_html=True)
