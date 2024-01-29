'''
Author: cathayana populuscathayana@gmail.com
Date: 2024-01-24 13:20:30
LastEditors: cathayana populuscathayana@gmail.com
LastEditTime: 2024-01-27 17:11:41
FilePath: /website_structure/STRUCTURE/GROUP1/page_3.py
Description:

Copyright (c) 2024 by Cathayana, All Rights Reserved.
'''

import streamlit as st

def show():
    st.write("下面的页面包含了一些Cathayana个人编写的未在前文出现的/与本章节无关的程序。")
    st.write("在左侧边栏中选择你想要查看的程序，由于服务器性能限制，部分程序加载时间可能较长，请耐心等待，本部署版本已恰好达到服务器性能瓶颈。在本地运行时可以自己自行修改各部分的gridsize取得更高精度结果。")
    # 定义一个侧边栏选择器
    program = st.sidebar.selectbox(
        "选择一个程序",
        ["H Atom Wavefunction 3D Visualization", "H Atom Orbitals 2D Visualization", "H2 Orbitals 2D Visualization", "H2 2pz Forming Sigma/Sigma* Orbitals", "H2 Pi/Pi* Orbitals 3D Visualzation", "黑体辐射绘图", "蒙特卡洛模拟 3D Pi/Pi* Orbitals"]  # 这里列出所有可选程序
    )

    # 根据用户的选择显示不同的内容
    if program == "H Atom Wavefunction 3D Visualization":
        code1()
    elif program == "H Atom Orbitals 2D Visualization":
        code2()
    elif program == "H2 Orbitals 2D Visualization":
        code3()
        #st.write('由于精度要求高，服务器可能无法负载，建议从源代码去掉注释内容，本地运行')
    elif program == "H2 2pz Forming Sigma/Sigma* Orbitals":
        code4()
    elif program == "H2 Pi/Pi* Orbitals 3D Visualzation":
        code5()
    elif program == "黑体辐射绘图":
        code6()
    elif program == "蒙特卡洛模拟 3D Pi/Pi* Orbitals":
        code7()
        #st.write('由于精度要求高，服务器可能无法负载，建议从源代码去掉注释内容，本地运行。')

    if 'show_code' not in st.session_state:
        st.session_state.show_code = False
        # 在侧边栏创建一个复选框来控制代码的显示
    st.session_state.show_code = st.sidebar.checkbox('显示代码', value=st.session_state.show_code)
    if st.session_state.show_code:
        with open(__file__, 'r') as file:
            code = file.read()
        st.markdown("```python\n" + code + "\n```", unsafe_allow_html=True)










def code1():
    # hydrogen_atom.py
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
    d = st.number_input("Enter sampling density:", min_value=1, format="%d", value=20, max_value=40)
    v = st.number_input("Enter wavefunction value:", min_value=0.0, value=0.0000001, max_value=1.0, format="%e")
    if st.button('run'):
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




def code2():
    # hydrogen_atom_2d.py
    import sympy as sp
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    st.title("Hydrogen Atom Orbitals 2D Visualization")
    st.write('由于性能限制，仅展示其中一幅图像')
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
            Theta = np.arctan2(np.sqrt(X**2), Y) 
            Phi = np.zeros_like(R)
        elif plane == "yz":
            Theta = np.arctan2(np.sqrt(X**2), Y)
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
    if st.button('run'):
        # 生成网格
        x = np.linspace(x_min, x_max, 300)
        y = np.linspace(y_min, y_max, 300)
        X, Y = np.meshgrid(x, y)


        # 计算概率密度
        Prob_xy = calculate_probability_density("xy", X, Y, R_num, Y_num)

        # 计算概率密度的径向平均值
        r_values = np.linspace(0, x_max, 500)
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
            size = end/20
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
                width=500,
                height=500,
                margin=dict(l=60, r=60, b=60, t=60)
            )
            st.plotly_chart(fig)
        # 绘制不同平面的图形
        # for plane, yaxis_title in [("xy", "y"), ("xz", "z"), ("yz", "z")]:
        for plane, yaxis_title in [("xz", "z")]:
            Prob = calculate_probability_density(plane, X, Y, R_num, Y_num)
            draw_density_contour(Prob, x, y, f"Probability Density in the {plane}-Plane", plane[0], yaxis_title)

def code3():
    
    #  hydrogen_molecule.py


    import sympy as sp
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.title("Hydrogen Molecule Orbitals Visualization")
    st.write('这里仅考虑s和p轨道')
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
    # 用户输入H-H距离
    d = st.number_input('Enter distance between Hydrogen atoms:', value=1.0,step=0.5, min_value=0.5, max_value=8.0)
    if st.button('run'):
        # 生成网格
        x = np.linspace(x_min, x_max, 200)
        y = np.linspace(y_min, y_max, 200)
        X, Y = np.meshgrid(x, y)
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
            size = end/100
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
                width=500,
                height=500,
                margin=dict(l=60, r=60, b=60, t=60)
            )
            st.plotly_chart(fig)

        draw_density_contour(Prob_sigma, x, y, "Probability Density of σ in the xy-Plane", "x", "y")
        draw_density_contour(Prob_sigma_star, x, y, "Probability Density of σ* in the xy-Plane", "x", "y")



def code4():
    import numpy as np
    import plotly.graph_objects as go
    st.title(r"3D Visualization of 2pz Orbitals Forming $\sigma/\sigma^*$ Bond")
    d = st.slider('Distance between H atoms', min_value=0.1, max_value=16.0, value=8.0, step=0.1)
    grid_size = st.slider('Grid size', min_value=10, max_value=30, value=20, step=5)
    if st.button('run'):
        # Defining the grid
        a0 = 1  # Bohr radius unit
        n, l, m = 2, 1, 0  # Principal quantum number n, azimuthal quantum number l, magnetic quantum number m


        x_range = np.linspace(-20, 20, grid_size)
        y_range = np.linspace(-10, 10, grid_size)
        z_range = np.linspace(-10, 10, grid_size)
        X, Y, Z = np.meshgrid(x_range, y_range, z_range)

        # Calculate r, theta, phi for each atom
        rA = np.sqrt((X + 0.5 * d)**2 + Y**2 + Z**2)
        rB = np.sqrt((X - 0.5 * d)**2 + Y**2 + Z**2)
        thetaA = np.arccos((X + 0.5 * d) / np.sqrt((X + 0.5 * d)**2 + Y**2 + Z**2))
        thetaB = np.arccos((X - 0.5 * d) / np.sqrt((X - 0.5 * d)**2 + Y**2 + Z**2))


        # Wave functions for 2pz orbitals
        R_2p_A = (1/np.sqrt(32*np.pi*a0**3)) * rA * np.exp(-rA/(2*a0))
        R_2p_B = (1/np.sqrt(32*np.pi*a0**3)) * rB * np.exp(-rB/(2*a0))
        Y_10A = np.sqrt(3/(4*np.pi)) * np.cos(thetaA)
        Y_10B = np.sqrt(3/(4*np.pi)) * np.cos(thetaB)
        psi_2pz_A = R_2p_A * Y_10A
        psi_2pz_B = R_2p_B * Y_10B

        p_1=np.abs(psi_2pz_A-psi_2pz_B)**2*np.sign(psi_2pz_A-psi_2pz_B)
        p_2=np.abs(psi_2pz_A+psi_2pz_B)**2*np.sign(psi_2pz_A+psi_2pz_B)
        mmm_1=np.max(np.abs(p_1.flatten()))/np.sqrt(2)
        mmm_2=np.max(np.abs(p_2.flatten()))/np.sqrt(2)


        colorscale = [[0, 'blue'], [0.5, 'white'], [1, 'red']]
        # Modify for sigma bond: using Y_10 (2pz) for psi_2pz and changing isosurface names
        iso_surface_sigma = go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=p_1.flatten() / np.sqrt(2),
            isomin=-0.1*mmm_1,
            isomax=0.1*mmm_1,
            surface_count=30,
            colorscale=colorscale,
            showscale=True,
            name='原子A+B sigma轨道',
            opacity=0.1
        )

        iso_surface_sigma_star = go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=p_2.flatten() / np.sqrt(2),
            isomin=-0.1*mmm_2,
            isomax=0.1*mmm_2,
            surface_count=30,
            colorscale=colorscale,
            showscale=True,
            name='原子A+B sigma*轨道',
            opacity=0.1
        )

        # Create plots for sigma and sigma* orbitals
        fig_sigma = go.Figure(data=[iso_surface_sigma])
        fig_sigma.update_layout(
            title='3D Visualization of 2pz Orbitals Forming Sigma Bond',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )

        fig_sigma_star = go.Figure(data=[iso_surface_sigma_star])
        fig_sigma_star.update_layout(
            title='3D Visualization of 2pz Orbitals Forming Sigma* Bond',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )

        # Display the plots
        #fig_sigma.show()
        #fig_sigma_star.show()
        st.write(fig_sigma)
        st.write(fig_sigma_star)


def code5():
    def load_orbitals(d):
        return load(f'GROUP1/joblibrary/code5/orbitals_d_{d:.1f}.joblib')
    import numpy as np
    import plotly.graph_objects as go
    st.title(r"H2 $\pi/\pi^*$ Orbitals Visualization")
    # 常量
    a0 = 1  # 一个单位的玻尔半径
    n, l, m = 2, 1, 0  # 主量子数n，角量子数l，磁量子数m
    d = st.slider('Distance between H atoms', min_value=0.1, max_value=8.0, value=4.0, step=0.1)
    grid_size = st.slider('Grid size', min_value=10, max_value=30, value=20, step=5)
    if st.button('run'):
        x_range = np.linspace(-7, 7, grid_size)
        y_range = np.linspace(-7, 7, grid_size)
        z_range = np.linspace(-7, 7, grid_size)
        X, Y, Z = np.meshgrid(x_range, y_range, z_range)

        # 计算r, theta, phi
        rA = np.sqrt((X + 0.5 * d)**2 + Y**2 + Z**2)
        rB = np.sqrt((X - 0.5 * d)**2 + Y**2 + Z**2)
        thetaA = np.arccos(Z / np.sqrt((X + 0.5 * d)**2 + Y**2 + Z**2))
        thetaB = np.arccos(Z / np.sqrt((X - 0.5 * d)**2 + Y**2 + Z**2))
        phi = np.arctan2(Y, X)

        # 波函数
        R_2p_A = (1/np.sqrt(32*np.pi*a0**3)) * rA * np.exp(-rA/(2*a0))
        R_2p_B = (1/np.sqrt(32*np.pi*a0**3)) * rB * np.exp(-rB/(2*a0))
        Y_10A = np.sqrt(3/(4*np.pi)) * np.cos(thetaA)
        Y_10B = np.sqrt(3/(4*np.pi)) * np.cos(thetaB)
        psi_2pz_A = R_2p_A * Y_10A
        psi_2pz_B = R_2p_B * Y_10B

        p_1=np.abs(psi_2pz_A+psi_2pz_B)**2*np.sign(psi_2pz_A+psi_2pz_B)
        p_2=np.abs(psi_2pz_A-psi_2pz_B)**2*np.sign(psi_2pz_A-psi_2pz_B)
        mmm_1=np.max(np.abs(p_1.flatten()))/np.sqrt(2)
        mmm_2=np.max(np.abs(p_2.flatten()))/np.sqrt(2)
        colorscale = [[0, 'blue'], [0.5, 'white'], [1, 'red']]



        iso_surface_pi = go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=(p_1.flatten())/np.sqrt(2),
            isomin=-0.5*mmm_1,
            isomax=0.5*mmm_1,
            surface_count=20,
            colorscale=colorscale,
            showscale=True,
            name='原子A+B pi*轨道',
            opacity=0.1
        )

        iso_surface_pi_star = go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=(p_2.flatten())/np.sqrt(2),
            isomin=-0.5*mmm_2,
            isomax=0.5*mmm_2,
            surface_count=20,
            colorscale=colorscale,
            showscale=True,
            name='原子A+B pi*轨道',
            opacity=0.1
        )

        # 创建图形 - 同时展示原子A和B
        fig = go.Figure(data=[iso_surface_pi])
        fig.update_layout(
            title='3D Visualization of pi Orbitals of Two Atoms',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )

        # 创建图形 - 同时展示原子A和B
        fig2 = go.Figure(data=[iso_surface_pi_star])
        fig2.update_layout(
            title='3D Visualization of pi* Orbitals of Two Atoms',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )



        # 显示图形
        #fig.show()
        #fig2.show()
        st.write(fig)
        st.write(fig2)

def code6():

    import numpy as np
    import plotly.graph_objs as go
    from scipy.constants import h, c, k
    import matplotlib.pyplot as plt
    def blackbody_radiation_density(f, T):
        """
        Calculate the blackbody radiation density at a given frequency f and temperature T.
        """
        return (2 * f**3 * h / c**2) / (np.exp(h * f / (k * T)) - 1)

    # 创建Streamlit的输入控件
    st.title('黑体辐射绘图')
    min_temp = st.slider('最小温度', 1, 3000, 25)
    max_temp = st.slider('最大温度', 2, 5000, 1000)
    temp_step = st.number_input('温度步长', min_value=1, max_value=100, value=10)

    # 根据输入计算温度范围
    temperatures = np.arange(min_temp, max_temp, temp_step)
    min_frequencies = st.slider('最小频率', 1e10, 3e14, 1e11)
    max_frequencies = st.slider('最大频率', 1e14, 9e14, 3e14)
    # Frequencies ranging from 1e11 to 3e14 Hz
    frequencies = np.linspace(min_frequencies, max_frequencies, 50)

    # 创建Plotly图表
    fig = go.Figure()

    # 为每个温度生成一个颜色映射

    # 生成颜色映射
    colors = [f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 1)' for r, g, b, _ in plt.cm.plasma(np.linspace(0, 1, len(temperatures)))]

    for i, T in enumerate(temperatures):
        densities = [blackbody_radiation_density(f, T) for f in frequencies]
        fig.add_trace(go.Scatter(x=frequencies, y=densities, mode='lines',
                                name=f'T = {T} K', line=dict(color=colors[-i])))

    # 更新图表布局
    fig.update_layout(
        title='Blackbody Radiation vs Frequency for Different Temperatures',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Energy Density (J/m^3/Hz)',
        template="plotly_dark",
        legend_title="Temperature",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    st.write(fig)


def code7():

    import numpy as np
    import plotly.graph_objects as go
    st.title(r''' 蒙特卡洛模拟$\pi,\pi^*$轨道''')
    a0 = 1  # Bohr 半径
    d = st.slider('键长', 1, 10, 5)
    num_points = st.slider('模拟点数', 10000, 30000, 20000,step=5000)
    if st.button("run"):
        r = np.random.uniform(0, 10 * a0, num_points)
        theta = np.random.uniform(0, np.pi, num_points)
        phi = np.random.uniform(0, 2 * np.pi, num_points)

        # 波函数构造
        # 对 atom A (于 -d/2)
        x_A = r * np.sin(theta) * np.cos(phi) - d / 2
        y_A = r * np.sin(theta) * np.sin(phi)
        z_A = r * np.cos(theta)

        # 对 atom B (于 d/2)
        x_B = r * np.sin(theta) * np.cos(phi) + d / 2
        y_B = y_A  # Same as y_A
        z_B = z_A  # Same as z_A

        # 分别计算 R, Y 
        R_2p_A = (1 / np.sqrt(32 * np.pi * a0**3)) * np.sqrt(x_A**2 + y_A**2 + z_A**2) * np.exp(-np.sqrt(x_A**2 + y_A**2 + z_A**2) / (2 * a0))
        Y_10_A = np.sqrt(3 / (4 * np.pi)) * z_A / np.sqrt(x_A**2 + y_A**2 + z_A**2)

        R_2p_B = (1 / np.sqrt(32 * np.pi * a0**3)) * np.sqrt(x_B**2 + y_B**2 + z_B**2) * np.exp(-np.sqrt(x_B**2 + y_B**2 + z_B**2) / (2 * a0))
        Y_10_B = np.sqrt(3 / (4 * np.pi)) * z_B / np.sqrt(x_B**2 + y_B**2 + z_B**2)

        # 波函数
        psi_A = R_2p_A * Y_10_A
        psi_B = R_2p_B * Y_10_B

        # Bonding orbital (pi)
        pi = (psi_A +psi_B) / np.sqrt(2)

        # Antibonding orbital (pi_star)
        pi_star = (psi_A - psi_B) / np.sqrt(2)

        # 概率密度（为了可视化的区分，对反相区域使用对应的负值）
        prob_density_pi = ((pi)**2)*np.abs(pi)/pi
        prob_density_pi_star = ((pi_star)**2)*np.abs(pi_star)/pi_star

        colorscale = [[0, 'blue'], [0.5, 'white'], [1, 'red']]

        # Bonding orbital (pi)
        scatter_pi = go.Scatter3d(
            x=x_A+ d / 2,
            y=y_A,
            z=z_A,
            mode='markers',
            marker=dict(
                size=2,
                color=prob_density_pi,
                colorscale=colorscale,
                opacity=0.1,
                colorbar=dict(title='Probability Density')
            ),
            name='σ轨道'
        )

        scatter_pi_star = go.Scatter3d(
            x=x_B- d / 2,
            y=y_B,
            z=z_B,
            mode='markers',
            marker=dict(
                size=2,
                color=prob_density_pi_star,
                colorscale=colorscale,
                opacity=0.1,
                colorbar=dict(title='Probability Density')
            ),
            name='σ*轨道'
        )

        fig1 = go.Figure(data=[scatter_pi])
        fig1.update_layout(
            title='2p-2p Pi Orbitals (Bonding) Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        #fig1.show()

        fig2 = go.Figure(data=[scatter_pi_star])
        fig2.update_layout(
            title='2p-2p Pi_star Orbitals (Anti-Bonding) Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        #fig2.show()

        st.write(fig1)
        st.write(fig2)