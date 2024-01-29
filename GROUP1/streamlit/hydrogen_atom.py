# hydrogen_atom.py

import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go

# 定义符号
r, theta, phi, Z = sp.symbols('r theta phi Z')

# 定义Laguerre多项式
def L(n, alpha, x_expr):
    x = sp.symbols('x')
    laguerre_unsub = x**(-alpha) * sp.factorial(n)**(-1) * sp.diff(x**(n + alpha) * sp.exp(-x), x, n) * sp.exp(x)
    laguerre = laguerre_unsub.subs(x, x_expr)
    return sp.simplify(laguerre)

# 定义径向波函数
def RR(n, l, r, Z):
    x = 2*Z*r/n
    front = sp.sqrt((2*Z/n)**3 * sp.factorial(n-l-1)/(2*n*sp.factorial(n+l)))
    laguerre = L(n-l-1, 2*l+1, x)
    return front * sp.exp(-Z*r/n) * (2*Z*r/n)**l * laguerre

# 定义关联勒让德多项式
def P(l, m, x_expr):
    x = sp.symbols('x')
    p_unsub = (1-x**2)**(abs(m)/2) * sp.diff((x**2-1)**l, x, l + abs(m)) / (2**l * sp.factorial(l))
    p = p_unsub.subs(x, x_expr)
    return sp.simplify(p)

# 定义球谐函数
def YY(l, m, theta, phi):
    P_lm = P(l, m, sp.cos(theta))
    spherical_harmonic = sp.sqrt((2*l+1)/(4*sp.pi) * sp.factorial(l-abs(m))/sp.factorial(l+abs(m))) * P_lm
    if m >= 0:
        return spherical_harmonic * sp.cos(m * phi)
    else:
        return spherical_harmonic * sp.sin(m * phi)
def page():
    st.title("Hydrogen Atom Wavefunction 3D Visualization")

    # Streamlit UI的创建
    n = st.number_input('Enter principal quantum number n:', min_value=1, max_value=10, value=1, step=1)
    l = st.number_input('Enter azimuthal quantum number l:', min_value=0, max_value=n-1, value=0, step=1)
    m = st.number_input('Enter magnetic quantum number m:', min_value=-l, max_value=l, value=0, step=1)
    Z = 1
    # 波函数的计算
    R_exp = RR(n, l, r, Z)
    Y_exp = YY(l, m, theta, phi)

    # 替换Z的值
    Z_val = 1
    R_exp_sub = R_exp.subs(Z, Z_val)
    Y_exp_sub = Y_exp.subs(Z, Z_val)

    # 使用lambdify转换符号表达式为数值函数
    R_num = sp.lambdify(r, R_exp_sub, 'numpy')
    Y_num = sp.lambdify((theta, phi), Y_exp_sub, 'numpy')

    # 三维图像的绘制
    a = st.number_input("Enter cube range:", value=20)
    d = st.number_input("Enter sampling density:", min_value=1, format="%d", value=50)
    v = st.number_input("Enter wavefunction value:", min_value=0.0, value=0.0000001, max_value=1.0, format="%e")

    # 创建三维网格
    X, Y, Z = np.mgrid[-a:a:d*1j, -a:a:d*1j, -a:a:d*1j]
    R = np.sqrt(X**2 + Y**2 + Z**2)
    T = np.arccos(Z/R)
    P = np.arctan2(Y, X)

    # 计算波函数的值
    Psi = R_num(R) * (Y_num(T, P))
    positive_Psi = np.maximum(Psi, 0)  # 仅正值
    negative_Psi = np.minimum(Psi, 0)  # 仅负值

    # 初始化 Figure 对象
    fig = go.Figure()

    # 向 Figure 添加正值的三维等值面
    fig.add_trace(go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=positive_Psi.flatten()**2,
        isomin=0,
        isomax=np.max(positive_Psi.flatten())**2,
        opacity=0.2,
        surface_count=40,
        colorscale='Reds',
        caps=dict(x_show=False, y_show=False),
        showscale=False
    ))

    # 向 Figure 添加负值的三维等值面
    fig.add_trace(go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=-negative_Psi.flatten()**2,
        isomin=0,
        isomax=-np.min(negative_Psi.flatten())**2,
        opacity=0.05,
        surface_count=40,
        colorscale='Blues',
        caps=dict(x_show=False, y_show=False),
        showscale=False
    ))

    # 更新图形布局
    fig.update_layout(
        title='Hydrogen Atom Wavefunction 3D Visualization',
        autosize=False,
        width=600,
        height=600,
        margin=dict(l=60, r=60, b=60, t=60)
    )

    # 在 Streamlit 中显示图形
    st.plotly_chart(fig, use_container_width=False)
    if st.button('Show Code'):
        with open(__file__, 'r') as file:
            code = file.read()
        st.markdown("```python\n" + code + "\n```", unsafe_allow_html=True)