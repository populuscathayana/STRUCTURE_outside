# hydrogen_molecule.py

import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
def page():
    st.title("Hydrogen Molecule Wavefunction Visualization")
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
    def Y(l, m, theta, phi):
        P_lm = P(l, m, sp.cos(theta))
        return sp.sqrt((2*l+1)/(4*sp.pi) * sp.factorial(l-abs(m))/sp.factorial(l+abs(m))) * P_lm * sp.exp(sp.I*m*phi)

    # Streamlit UI的创建
    n = st.number_input('Enter principal quantum number n:', min_value=1, max_value=10, value=1, step=1)
    l = st.number_input('Enter azimuthal quantum number l:', min_value=0, max_value=n-1, value=0, step=1)
    m = st.number_input('Enter magnetic quantum number m:', min_value=-l, max_value=l, value=0, step=1)
    # 波函数的计算与可视化
    # 计算表达式
    R_exp = R(n, l, r, Z)
    Y_exp = Y(l, m, theta, phi)
    Z_val = 1
    # 替换Z的值
    R_exp_sub = R_exp.subs(Z, Z_val)
    Y_exp_sub = Y_exp.subs(Z, Z_val)
    # 使用lambdify转换符号表达式为数值函数
    variables_R = (r, theta, phi)
    variables_Y = (theta, phi)
    R_num = sp.lambdify(variables_R, R_exp_sub, 'numpy')
    Y_num = sp.lambdify(variables_Y, Y_exp_sub, 'numpy')
    # 用户输入界限值
    x_max = st.number_input('Maximum x', value=4)
    x_min = x_max * -1
    y_max = st.number_input('Maximum y', value=4)
    y_min = y_max * -1
    # 生成网格
    x = np.linspace(x_min, x_max, 800)
    y = np.linspace(y_min, y_max, 800)
    X, Y = np.meshgrid(x, y)
    # 用户输入H-H距离
    d = st.number_input('Enter distance between Hydrogen atoms:', value=1.0)
    # 计算H-H分子轨道
    def get_direction_from_lm(l, m):
        if l == 0:
            return "s"
        elif l == 1:
            if m == 0:
                return "pz"
            elif m == -1:
                return "px"
            elif m == 1:
                return "py"
        return None

    # ... [之前的代码部分保持不变]

    def molecular_orbital(X, Y, d, n, l, m):
        # 原子A的坐标为(-d/2, 0)，原子B的坐标为(d/2, 0)
        rA = np.sqrt((X + d/2)**2 + Y**2)
        rB = np.sqrt((X - d/2)**2 + Y**2)
        Theta = np.full_like(X, np.pi/2)  # Theta为π/2，因为我们在xy平面上工作
        Phi = np.arctan2(Y, X)

        direction = get_direction_from_lm(l, m)

        psi_A = R_num(rA, Theta, Phi) * Y_num(Theta, Phi)
        psi_B = R_num(rB, Theta, Phi) * Y_num(Theta, Phi)

        # ... [之后的代码部分保持不变]


        if direction == "s":
            sigma = (psi_A + psi_B) / np.sqrt(2)
            sigma_star = (psi_A - psi_B) / np.sqrt(2)
            return sigma, sigma_star
        elif direction == "pz":
            # 这里我们不需要对ψ做任何特别的调整，因为Y(θ, φ)已经处理了m=0的情况
            sigma = (psi_A + psi_B) / np.sqrt(2)
            sigma_star = (psi_A - psi_B) / np.sqrt(2)
            return sigma, sigma_star
        elif direction == "px":
            sigma = (psi_A + psi_B) / np.sqrt(2)
            sigma_star = (psi_A - psi_B) / np.sqrt(2)
            return sigma, sigma_star
        elif direction == "py":
            sigma = (psi_A + psi_B) / np.sqrt(2)
            sigma_star = (psi_A - psi_B) / np.sqrt(2)
            return sigma, sigma_star

        return None, None


    sigma, sigma_star = molecular_orbital(X, Y, d, n, l, m)
    Prob_sigma = sigma**2
    Prob_sigma_star = sigma_star**2



    # 计算在xy平面上的σ和σ*的概率密度
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

    draw_density_contour(Prob_sigma, x, y, "Probability Density of σ in the xy-Plane", "x", "y")
    draw_density_contour(Prob_sigma_star, x, y, "Probability Density of σ* in the xy-Plane", "x", "y")
    if st.button('Show Code'):
        with open(__file__, 'r') as file:
            code = file.read()
        st.markdown("```python\n" + code + "\n```", unsafe_allow_html=True)