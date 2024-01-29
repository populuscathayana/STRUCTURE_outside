'''
Author: cathayana populuscathayana@gmail.com
Date: 2024-01-24 13:19:20
LastEditors: cathayana populuscathayana@gmail.com
LastEditTime: 2024-01-25 15:35:32
FilePath: /website_structure/STRUCTURE/GROUP1/page_1.py
Description: 

Copyright (c) 2024 by Cathayana, All Rights Reserved. 
'''

# GROUP1/page_1.py
import importlib
import streamlit as st

def show():
    st.write("#### 4.1 价层电子对互斥理论（VSEPR）")
    st.write('''
<b>原子周围各个价电子对之间由于相互排斥，在键长一定的条件下，互相间的间距越远越稳定。
这就要求分布在中心原子周围的价电子对尽可能相互远离。</b> 这种斥力来源于两部分：
各电子对之间的静电排斥作用和价电子对之间自旋相同的电子互相回避的效应（Pauli排斥）。
VSEPR对分子几何构型的判断规则：
    - 为了使价电子对的排斥最小，可以将价电子对看作等距离的排布在同一球面上，形成规则的多面体形式
    - 中心原子A与m个配位体L之间所形成的键可能是单键，也可能是双键和叁键等多重键，其中排斥力的大小为：
        $$叁键-叁键>叁键-双键>双键-双键>双键-单键>单键-单键$$
    - 成键电子和孤对电子的分布情况并不等价，成键电子受到两个成键原子核的吸引，
    比较集中在键轴的位置，而孤对电子没有这种限制，在空间中更加弥散，因此其对相邻电子对的排斥也会更大一点，
    因此其排斥力大小为：
        $$孤对电子-孤对电子>孤对电子-成键电子>成键电子-成键电子$$
    - 电负性高的配体，对电子的吸引能力强，价电子离中心原子较远，占据空间角度也较小
    <font color = red>价层电子对互斥理论对少数化合物的判断不准。</font>
    例如$CaF_2,SrF_2,BaF_2$等是弯曲形而不是预期的直线形。
    此外价层电子对互斥理论也不能应用于过渡金属化合物，除非具有充满、半充满或全空的d轨道。'''
    , unsafe_allow_html=True)
    st.write("#### 4.2 杂化轨道理论")
    st.write(r'''**原子在组合成分子的过程中，根据原子的成键需求，在周围原子的影响下，
             将原有的原子轨道进一步线性组合成为新的原子轨道，该过程称为原子轨道杂化。
             杂化时轨道的数目不变，轨道在空间的分布方向和分布情况发生改变，能级改变。**
             组合所得的杂化轨道一半都会和其他原子形成较强的$\sigma$键或是安置孤对电子，
             而不会以空的杂化轨道形成存在。
             在某个原子的几个杂化轨道中，参与杂化的s、p、d轨道成分如果相等，
             则称为等性杂化轨道；若不相等，则成为不等性杂化轨道。
             下表列出了一些常见的杂化轨道性质：''')
    st.write(r'''
        |杂化轨道|参与杂化的原子轨道|构型|对称性|示例|
        |:-----:|:---------------:|:--:|:----:|:-:|
        |sp|$s,p_z$|直线形|$D_{\infty h}$|$CO_2,N_3^-$|
        |$sp^2$|$s,p_x,p_y$|平面三角形|$D_{3h}$|$BF_3,SO_3$|
        |$sp^3$|$s,p_x,p_y,p_z$|四面体形|$T_d$|$CH_4$|
        |$dsp^2$  或  $sp^2d$|$d_{x^2-y^2},s,p_x,p_y$|平面四方形|$D_{4h}$|$Ni(CN)_4^{2-}$|
        |$dsp^3$  或  $sp^3d$|$d_{z^2},s,p_x,p_y,p_z$|三角双锥形|$D_{3h}$|$PF_5$|
        |$dsp^3$|$d_{x^2-y^2},s,p_x,p_y,p_z$|四方锥形|$C_{4v}$|$IF_5$|
        |$d^2sp^3$  或  $sp^3d^2$|$d_{z^2},d_{x^2-y_2},s,p_x,p_y,p_z$|正八面体形|$O_h$|$SF_5$|
    ''',unsafe_allow_html=True)

    st.write(r'''
<b>杂化轨道具有和普通原子轨道相同的性质，必须满足正交性和归一性，例如：</b>
''',unsafe_allow_html=True)
    st.write(r'''
由s和p轨道组成的杂化轨道$\psi_i=a_is+b_ip$，由归一化可以得到：
''',unsafe_allow_html=True)
    st.write(r'''
$$\int\psi_i^* \psi_i d \tau =1 \quad \quad \quad a_i^2+b_i^2=1 $$
''',unsafe_allow_html=True)
    st.write(r'''
由正交性可以得到：
''',unsafe_allow_html=True)
    st.write(r'''
$$\int\psi_i^*\psi_jd\tau = 0,\quad \quad i\neq j 时 $$

''',unsafe_allow_html=True)
    st.write(r'''
根据这个性质，考虑杂化轨道的空间分布和咋花钱原子轨道的去向，
就可以写出各个杂化轨道中原子轨道的组合系数，例如由$s,p_x,p_y$组成的平面三角形的$sp^2$杂化轨道，
由等性杂化的概念可知每个杂化轨道中s成分占$1/3$，组合系数为$1/\sqrt3$，其中$\psi_1$与x轴平行，
与y轴垂直，故$p_y$没有贡献，所以$$\psi_1 =\sqrt\frac13s+\sqrt\frac23p_x $$
''',unsafe_allow_html=True)
    st.write(r'''
同理：$$\psi_2=\sqrt\frac13s-\sqrt\frac16p_x+\sqrt\frac12p_y $$
''',unsafe_allow_html=True)
    st.write(r'''
$$\psi_3=\sqrt\frac13s-\sqrt\frac16p_x-\sqrt\frac12p_y $$

<b>原子轨道经过杂化会使键的相对强度增大。因为杂化后的原子轨道在某些方向上的分布更加集中，与其他原子成键时，重叠部分增大城建能力增强</b>

杂化轨道理论可以简明的阐明分子的几何构型以及一部分分子的性质，而在使用杂化轨道理论讨论分子性质时，也通常结合分子的键型以及分子的几何构型来进行。
''',unsafe_allow_html=True)
    st.write('''以下是一个杂化轨道计算的演示程序，以sp2杂化为例(为节省计算资源，仅展示其中一个轨道，读者可自行修改代码)''')
    if st.button('杂化轨道示意图'):
        import numpy as np
        import plotly.graph_objects as go

        a_0 = 1

        #def psi_1s(x, y, z, Z):
        #    r = np.sqrt(x**2 + y**2 + z**2)
        #    normalization_factor = (Z**3 / (np.pi * a_0**3))**0.5
        #    return normalization_factor * np.exp(-Z * r / a_0)

        def psi_2s(x, y, z, Z):
            r = np.sqrt(x**2 + y**2 + z**2)
            normalization_factor = (Z**3 / (8 * np.pi * a_0**3))**0.5
            return normalization_factor * (1 - Z * r / (2 * a_0)) * np.exp(-Z * r / (2 * a_0))

        def psi_2px(x, y, z, Z):
            r = np.sqrt(x**2 + y**2 + z**2)
            normalization_factor = (Z**5 / (32 * np.pi * a_0**5))**0.5
            return normalization_factor * (x / r) * np.exp(-Z * r / (2 * a_0))

        #def psi_2py(x, y, z, Z):
        #    r = np.sqrt(x**2 + y**2 + z**2)
        #    normalization_factor = (Z**5 / (32 * np.pi * a_0**5))**0.5
        #    return normalization_factor * (y / r) * np.exp(-Z * r / (2 * a_0))

        #def psi_2pz(x, y, z, Z):
        #    r = np.sqrt(x**2 + y**2 + z**2)
        #    normalization_factor = (Z**5 / (32 * np.pi * a_0**5))**0.5
        #    return normalization_factor * (z / r) * np.exp(-Z * r / (2 * a_0))

        # 创建网格
        X, Y, Z = np.mgrid[-2:2:60j, -2:2:60j, -2:2:60j]

        # 定义sp²杂化轨道
        psi_1 = np.sqrt(1/3) * psi_2s(X, Y, Z, 9) + np.sqrt(2/3) * psi_2px(X, Y, Z, 5)
        #psi_2 = np.sqrt(1/3) * psi_2s(X, Y, Z, 9) - np.sqrt(1/6) * psi_2px(X, Y, Z, 5) + np.sqrt(1/2) * psi_2py(X, Y, Z, 5)
        #psi_3 = np.sqrt(1/3) * psi_2s(X, Y, Z, 9) - np.sqrt(1/6) * psi_2px(X, Y, Z, 5) - np.sqrt(1/2) * psi_2py(X, Y, Z, 5)

        # 定义颜色映射
        colorscale = [[0, 'blue'], [0.5, 'rgba(0,0,0,0)'], [1, 'red']]

        # 创建杂化轨道的可视化
        #for psi, name in zip([psi_1, psi_2, psi_3], ['psi_1', 'psi_2', 'psi_3']):
        for psi, name in zip([psi_1], ['psi_1']):
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
                #title=f"sp² Hybrid Orbital Visualization: {name}",
                margin=dict(t=0, l=0, b=0, r=0)
            )

            st.write(fig)
    st.write("#### 4.3离域分子轨道理论")
    st.write(r'''
<b>当我们使用分子轨道理论取处理多原子分子时，最常用的办法是用非杂化的原子轨道进行线性组合构成分子轨道，这些分子轨道是离域化的，在几个原子之间离域运动</b>

以$CH_4$分子为例进行讨论：
''',unsafe_allow_html=True)
    st.image('GROUP1/图床/5.3.1.png')
    st.write(r'''
$CH_4$分子的离域分子轨道（MO）由8个原子轨道（AO）（即C原子的2s、2p以及4个H原子的1s轨道）线性组合而成，因此每个MO平均占有两个AO，由图可见，与C原子2s轨道球形对称性匹配的线性组合是$$\frac12(1s_a+1s_b+1s_c+1s_d) $$
与C原子的$2p_x,2p_y,2p_z$轨道对称性匹配的线性组合依次是
''',unsafe_allow_html=True)
    st.write(r'''
$$\frac12(1s_a+1s_b-1s_c-1s_d) $$ 
''',unsafe_allow_html=True)
    st.write(r'''
$$\frac12(1s_a-1s_b-1s_c+1s_d) $$
''',unsafe_allow_html=True)
    st.write(r'''
$$\frac12(1s_a-1s_b+1s_c-1s_d) $$
''',unsafe_allow_html=True)
    st.write(r'''
将中心原子的轨道与周围H原子的轨道进一步组合，就可以得到四个成键轨道和四个反键轨道：
''',unsafe_allow_html=True)
    st.write(r'''
$$\psi_s = s + \frac12(1s_a+1s_b+1s_c+1s_d) \quad \psi_s^* = s - \frac12(1s_a+1s_b+1s_c+1s_d) $$
''',unsafe_allow_html=True)
    st.write(r'''
$$\psi_x = p_x + \frac12(1s_a+1s_b-1s_c-1s_d) \quad \psi_x^* = p_x - \frac12(1s_a+1s_b-1s_c-1s_d) $$
''',unsafe_allow_html=True)
    st.write(r'''
$$\psi_y = p_y + \frac12(1s_a-1s_b-1s_c+1s_d) \quad \psi_y^* = p_y - \frac12(1s_a-1s_b-1s_c+1s_d) $$ 
''',unsafe_allow_html=True)
    st.write(r'''
$$\psi_z = p_z + \frac12(1s_a-1s_b+1s_c-1s_d) \quad \psi_z^* = p_z - \frac12(1s_a-1s_b+1s_c-1s_d)$$
''',unsafe_allow_html=True)
    st.write(r'''
             
''',unsafe_allow_html=True)
    st.write("同样，我们选取前四个轨道进行可视化展示")
    if st.button('甲烷的分子轨道示意图'):
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

        X, Y, Z = np.mgrid[-5:5:50j, -5:5:50j, -5:5:50j]
        '''
            为方便期间使用格点模拟
            修改 j 之前的密度参数来得到更稀/密的格点
        '''

        # Wavefunction combinations for CH4 MOs
        '''
            轨道取值已纠正
        '''

        d = 1
        psi_s =         psi_2s(X, Y, Z, Z=6) - 0.5 * (psi_1s(X+d, Y+d, Z+d, Z=1) + psi_1s(X+d, Y-d, Z-d, Z=1) + psi_1s(X-d, Y+d, Z-d, Z=1) + psi_1s(X-d, Y-d, Z+d, Z=1))
        psi_s_star =    psi_2s(X, Y, Z, Z=6) + 0.5 * (psi_1s(X+d, Y+d, Z+d, Z=1) + psi_1s(X+d, Y-d, Z-d, Z=1) + psi_1s(X-d, Y+d, Z-d, Z=1) + psi_1s(X-d, Y-d, Z+d, Z=1))
        psi_px =        psi_2px(X, Y, Z, Z=6) - 0.5 * (psi_1s(X+d, Y+d, Z+d, Z=1) + psi_1s(X+d, Y-d, Z-d, Z=1) - psi_1s(X-d, Y+d, Z-d, Z=1) - psi_1s(X-d, Y-d, Z+d, Z=1))
        psi_px_star =   psi_2px(X, Y, Z, Z=6) + 0.5 * (psi_1s(X+d, Y+d, Z+d, Z=1) + psi_1s(X+d, Y-d, Z-d, Z=1) - psi_1s(X-d, Y+d, Z-d, Z=1) - psi_1s(X-d, Y-d, Z+d, Z=1))
        #以下取值未修正，不保证正确性
        #psi_py =        psi_2py(X, Y, Z, Z=6) + 0.5 * (psi_1s(X+d, Y+d, Z+d, Z=1) - psi_1s(X+d, Y-d, Z-d, Z=1) - psi_1s(X-d, Y+d, Z-d, Z=1) + psi_1s(X-d, Y-d, Z+d, Z=1))
        #psi_py_star =   psi_2py(X, Y, Z, Z=6) - 0.5 * (psi_1s(X+d, Y+d, Z+d, Z=1) - psi_1s(X+d, Y-d, Z-d, Z=1) - psi_1s(X-d, Y+d, Z-d, Z=1) + psi_1s(X-d, Y-d, Z+d, Z=1))
        #psi_pz =        psi_2pz(X, Y, Z, Z=6) + 0.5 * (psi_1s(X+d, Y+d, Z+d, Z=1) - psi_1s(X+d, Y-d, Z-d, Z=1) + psi_1s(X-d, Y+d, Z-d, Z=1) - psi_1s(X-d, Y-d, Z+d, Z=1))
        #psi_pz_star =   psi_2pz(X, Y, Z, Z=6) - 0.5 * (psi_1s(X+d, Y+d, Z+d, Z=1) - psi_1s(X+d, Y-d, Z-d, Z=1) + psi_1s(X-d, Y+d, Z-d, Z=1) - psi_1s(X-d, Y-d, Z+d, Z=1))


        # Orbitals list
        orbitals = {
            "psi_s": psi_s,
            "psi_s_star": psi_s_star,
            "psi_px": psi_px,
            "psi_px_star": psi_px_star,
            #"psi_py": psi_py,
            #"psi_py_star": psi_py_star,
            #"psi_pz": psi_pz,
            #"psi_pz_star": psi_pz_star
        }

        colorscale = [[0, 'blue'], [0.5, 'rgba(0,0,0,0)'], [1, 'red']]

        # Generating figures for each orbital
        for name, orbital in orbitals.items():
            fig = go.Figure()

            fig.add_trace(go.Isosurface(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                value=orbital.flatten(),
                isomin=-0.14,
                isomax=0.14,
                opacity=0.1,
                surface_count=20,
                colorscale=colorscale
            ))

            fig.update_layout(
                title=f"CH4 Molecular Orbital Visualization",
                margin=dict(t=0, l=0, b=0, r=0)
            )

            st.write(fig)
    st.write("#### 4.4 休克尔分子轨道法（HMO）")
    st.write(r'''
             
''',unsafe_allow_html=True)
    st.write(r'''
##### HMO法基本介绍

**有机平面构型的共轭分子中，$\sigma$键是定域键，与原子核一起组成分子骨架，每个C原子剩余一个垂直于分子平面的p轨道，一并组合成为多中心$\pi$键,又称为离域$\pi$键，所有的$\pi$电子在整个分子骨架范围内运动**

当使用HMO法处理共轭分子结构时，有如下假定：
- 由于$\pi$电子在核和$\sigma$键所形成的整个分子骨架中运动，可以将$\sigma$键和$\pi$键分开处理
- 共轭分子具有相对不变的$\sigma$骨架，而$\pi$电子的状态决定分子的性质
- 用$\psi_k$描述第k个$\pi$个电子的运动状态，其薛定谔方程为：$$\hat{H_{\pi}}\psi_k = E_k\psi_k$$

HMO法规定每个C原子的$\alpha$积分相同，每个相邻C原子的$\beta$积分也相同，而不相邻原子的$\beta$积分和重叠积分D均为0。这样可以不用考虑势能函数V和$\hat{H_{\pi}}$的具体形式。处理步骤如下：
1. 设共轭分子有n个C原子，每个C原子提供一个p轨道$\phi_i$用以组成分子轨道$\psi$。按照LCAO得：$$\psi = c_1\psi_1+c_2\psi_2+ \dots +c_n\psi_n = \sum c_i\phi_i$$ 

其$\psi$是分子轨道，$\phi_i$是组成分子的第i个C原子的p轨道，$c_i$是分子轨道中第i个C原子的原子轨道组合系数。

2. 根据线性变分法，由:$$\frac{\partial E}{\partial c_1} = 0;\quad \frac{\partial E}{\partial c_2} = 0; \dots ;\quad\frac{\partial E}{\partial c_n} = 0$$

可得久期方程式:

$$
\begin{equation}
    \begin{bmatrix} H_{11}-ES_{11} & H_{12}-ES_{12} & \dots & H_{1n}-ES_{1n} \\ H_{21}-ES_{11} & H_{22}-ES_{12} & \dots & H_{2n}-ES_{1n} \\ \dots & \dots & \dots & \dots \\ H_{n1}-ES_{n1} & H_{n2}-ES_{n2} & \dots & H_{nn}-ES_{nn} \end{bmatrix}
    \begin{bmatrix} c_1 \\ c_2 \\ \dots \\ c_n \end{bmatrix}
    = 0
\end{equation}
$$


式中$H_{ij} = \int \phi_i \hat{H_\pi} \phi_j d\tau,\ S_{ij} = \int \phi_i \phi_j d\tau$。此行列式方程是E的一元n次代数方程式。

3. 引入下列基本假设：
$$H_{11}=H_{22}=\dots=H_{nn}=\alpha $$
$$
\begin{equation}
	H_{ij}=\left \{
	\begin{aligned}
		\beta & , & (i和j相邻)\\
		0 & , & (i和j不相邻)
	\end{aligned}
	\right.
\end{equation}
$$

$$
\begin{equation}
	S_{ij}=\left \{
	\begin{aligned}
		1 & , &\quad & (i=j)\\
		0 & , &\quad & (i\neq j)
	\end{aligned}
	\right.
\end{equation}
$$
化简上述行列式方程，求出n个$E_k$，将每个$E_k$值代回久期方程，得到$c_{ki}$和$\psi_k$

4. 画出与分子轨道$\psi_k$相应的能级$E_k$图，排布$\pi$电子，画出$\psi_k$的图形

5. 计算以下数据并作出分子图：	
	* 电荷密度$\rho_i$——第i个原子上出现的$\pi$电子数，就等于离域$\pi$键中$\pi$电子在第i个原子周围出现的概率：
	$$\rho_i = \underset {k}{\sum}n_kc_{ki}^2$$
	式中$n_k$表示在$\psi_k$中的电子数，$c_{ki}$为分子轨道$\psi_i$中第i个原子轨道的组合系数
	
	* 键级$P_{ij}$——原子i和j间$\pi$键的强度：
	$$P_{ij} = \underset{k}{\sum}n_kc_{ki}c_{kj}$$

	* 自由价$F_i$——第i个原子剩余城建能力的相对大小：
	$$F_i = F_{max} - \underset i {\sum} P_{ij}$$ 
	其中$F_{max}为\sqrt3$，是采用了理论上存在的三次甲基甲烷分子的中心碳原子和周围3个碳原子形成$\pi$键键级总和为$\sqrt3$。$\underset i \sum P_{ij}$ 为原子i与其接邻的原子之间$\pi$键键级之和。

	* 分子图——把共轭分子由HMO法求得的电荷密度$\rho_i$、键级$P_{ij}$、自由价$F_i$都标在一张分子结构图上，即组成分子图。            
''',unsafe_allow_html=True)
    st.write(r'''
##### 丁二烯的HMO法处理示例
丁二烯的分子轨道为：
$$\psi = c_1\phi_1+c_2\phi_2+c_3\phi_3+c_4\phi_4$$
式中$\phi_1,\phi_2,\phi_3,\phi_4$为参与共轭的四个C原子的$p_z$轨道；$c_1,c_2,c_3,c_4$为变分参数。按照变分法可得$c_1,c_2,c_3,c_4$应该满足的久期方程式。用上述方程化简后得到：
$$
\begin{equation}
    \begin{bmatrix} \alpha - E & \beta & 0 & 0 \\ \beta & \alpha - E & \beta & 0 \\ 0 & \beta & \alpha - E & \beta\\ 0 & 0 & \beta & \alpha - E \end{bmatrix}
    \begin{bmatrix} c_1 \\ c_2 \\ c_3 \\ c_4 \end{bmatrix}
    = 0
\end{equation}
$$

用$x=\frac{\alpha-E}{\beta}$进行变量替换，可以继续化简为：

$$
\begin{equation}
    \begin{bmatrix} x & 1 & 0 & 0 \\ 1 & x & 1 & 0 \\ 0 & 1 & x & 1 \\ 0 & 0 & 1 & x \end{bmatrix}
    \begin{bmatrix} c_1 \\ c_2 \\ c_3 \\ c_4 \end{bmatrix}
    = 0
\end{equation}
$$

根据丁二烯分子拥有对称中心的性质，$c_1=\pm c_4,\quad c_2=\pm c_3$。当$c_1=c_4,\quad c_2=c_3$时，上式可化简为：
$$
\begin{equation}
\begin{aligned}
    xc_1+c_2&=0\\c_1+(X+d&)c_2=0
\end{aligned}
\end{equation}
$$
解得$x=-1.62/0.62$。

当$c_1=-c_4,\quad c_2=-c_3$时，上式可化简为：
$$
\begin{equation}
\begin{aligned}
    xc_1+c_2&=0\\c_1+(X-d&)c_2=0
\end{aligned}
\end{equation}
$$
解得$x=1.62/-0.62$。
将得到的四个x值分别回代式中，结合归一化条件$$c_1^2+c_2^2+c_3^2+c_4^2=1$$可以得到分子轨道中各原子轨道的组合系数c：             
''',unsafe_allow_html=True)
    st.write(r'''
|分子轨道能级|分子轨道波函数|
|:---------:|:-----------:|
|$E_1=\alpha+1.62\beta$|$\psi_1=0.372\phi_1+0.602\phi_2+0.602\phi_3+0.372\phi_4$|
|$E_2=\alpha+0.62\beta$|$\psi_2=0.602\phi_1+0.372\phi_2-0.372\phi_3-0.602\phi_4$|
|$E_3=\alpha-0.62\beta$|$\psi_2=0.602\phi_1-0.372\phi_2-0.372\phi_3+0.602\phi_4$|
|$E_4=\alpha-1.62\beta$|$\psi_1=0.372\phi_1-0.602\phi_2+0.602\phi_3-0.372\phi_4$|
          
''',unsafe_allow_html=True) 
    st.image('GROUP1/图床/5.4.1.png')
    st.write("关于一个Huckel法自动计算轨道系数的例子你可以在页面2中找到")
    # 初始化session_state
    if 'show_page_2' not in st.session_state:
        st.session_state['show_page_2'] = False

    # 切换页面2显示状态的按钮
    if st.button('显示/隐藏页面2'):
        st.session_state['show_page_2'] = not st.session_state['show_page_2']

    # 根据show_page_2的状态决定是否显示页面2的内容
    if st.session_state['show_page_2']:
        page = '页面 2'
        module = importlib.import_module(f"GROUP1.page_{page.split(' ')[1]}")
        module.show()
    
    st.write(r'''#### 4.5 离域$\pi$键和共轭效应
**形成化学键的$\pi$电子不局限于两个原子的区域，而是在多个成键原子的分子骨架中运动，这种化学键称为离域$\pi$键**
一般地，离域$\pi$键的形成需要满足以下条件：
- 原子共面，每个原子提供一个方向相同的p轨道，或者合适的d轨道
- $\pi$电子数小于成键的轨道数的两倍

当分子形成离域$\pi$键时，分子的物理化学性质不再是简单的各个双键和单键的性质加和，而是会表现出特有的性质，称为共轭效应或离域效应。
- 电性：离域$\pi$键的形成会增加物质的导电能力。
- 颜色：离域$\pi$键的形成会增大$\pi$电子的运动范围，使体系的能量降低，能级间隔变小，光谱由紫外光区向可见光区移动。
- 酸碱性：离域$\pi$键的形成会增大特定结构的稳定性，从而促使氢离子的解离或吸收。
- 化学反应性

**超共轭效应**：C——H键等$\sigma$键轨道和相邻原子的$\pi$键轨道或其他轨道相互叠加，扩大$\sigma$电子的活动范围产生的离域效应。

''',unsafe_allow_html=True)
    st.write(r'''#### 4.6 分子轨道对称性和反应机理
''',unsafe_allow_html=True)
    st.write(r'''
    ##### 前线轨道理论
    **分子中由一系列能级从低到高排列的分子轨道，电子值填充了其中能量较低的一部分。已填电子的能量最高轨道称为最高占据轨道（HOMO），能量最低的空轨道称为最低空轨道（LUMO），这些轨道统称为前线轨道。化学反应的条件和方式主要取决与前线轨道的对称性：**

    - 分子在反应过程中，分子轨道相互作用，优先起作用的是前线轨道。一个分子的HOMO和另一个分子的LUMO必须对称性合适，这样形成的过渡态是活化能较低的状态，称为对称允许状态
    - 互相起作用的HOMO和LUMO能级高低必须接近
    - 随着电子从一个分子的HOMO转移到另一个分子的LUMO，电子转移方向从电负性角度判断应该合理，电子的转移要和旧键的削弱相一致，不能产生矛盾
    ##### 分子轨道对称守恒原理
    **电环化反应是指直链共轭烯烃分子两端的C原子上的轨道相互作用，生成一个单键，形成环形分子的过程，分子轨道对称守恒原理要求：**
    - 反应物的分子轨道与产物的分子轨道一一对应
    - 相关轨道的对称性相同
    - 相关轨道的能量应当相互接近
    - 对称性相同的相关线不能相交
    在能量相关图中，若产物的每个成键轨道都只和反应物的成键轨道相互关联，则反应的活化能低，称为对称允许，加热即可发生反应；若双方有成键轨道和反键轨道相关联，则反应活化能高，称为对称禁阻，需要先把反应物的基态电子激发到激发态。
    ''',unsafe_allow_html=True)
    st.write(r'''#### 4.7缺电子多中心键和硼烷的结构
    ''',unsafe_allow_html=True)
    st.write(r'''     
    <b>Li、Be、B、Al等原子价层的原子轨道数多于价电子数，会倾向于接受电子形成四面体型的配合物，当没有合适的外来原子时，化合物自身可以通过聚合相互提供具有孤对电子的原子形成四面体配合物，从而形成缺电子多中心键</b>

    硼烷是典型的缺电子多中心键典例，乙硼烷（$B_2H_6$）中存在B-H-B三中心二电子键，B原子以$sp_3$杂化轨道参与成键，除了两端形成B-H键之外，每个B原子的一个$sp_3$轨道都和氢原子的1s轨道叠加，共同形成B-H-B三中心键，如图所示:
    ''',unsafe_allow_html=True)
    st.image('GROUP1/图床/5.7.1.png')
    st.write(r'''    在一些其他的硼烷和碳硼烷结构中，还有可能出现BBB 3c-2e键以及BBC 3c-2e键等。


    ''',unsafe_allow_html=True)
    st.write(r'''    ##### 硼烷结构的描述——styx数
    ''',unsafe_allow_html=True)
    st.image('GROUP1/图床/5.7.0.jpg')
    st.write(r''' 我们可以使用styx编码来对一个硼烷分子中四种不同化学键的数目进行标定，s代表B-H-B三中心二电子键；t代表BBB三中心二电子键；y代表B-B键；x代表H-B-H键。使用这四种化学键来表达硼烷的结构式称为styx数码，图中展示了一些硼烷的结构式以及相应的styx数码：
''',unsafe_allow_html=True)
    st.image('GROUP1/图床/5.7.2.png')
    st.write(r''' 我们可以使用styx编码来对一个硼烷分子中四种不同化学键的数目进行标定，s代表B-H-B三中心二电子键；t代表BBB三中心二电子键；y代表B-B键；x代表H-B-H键。使用这四种化学键来表达硼烷的结构式称为styx数码，图中展示了一些硼烷的结构式以及相应的styx数码：
''',unsafe_allow_html=True)
    st.write(r'''
使用styx数码时应当注意：
- 每一对相邻的B原子由一个B-B，BBB或BHB键连接
- 每个B原子利用它的4个价轨道成键，以达到八电子组态
- 两个B原子不能同时通过B-B键和BBB键，或同时通过B-B键和BHB键结合
- 每个B原子至少要和一个端位H原子结合
- 必要时根据分子的对称性写出在共振杂化体之间共振的结构式

由此，我们可以通过八隅律和styx数码进行分子骨干键数的计算：
\
对于一个由n个主族元素的原子组成的分子骨干$M_n$，g为其已有的价电子总数。当骨干中由一个共价单键在两个M原子之间形成时，这两个原子都互相得到一个电子。为了使分子满足八隅律从，原子间应该有$\frac12(8n-g)$对电子形成共价单键，这些成键电子对数目定义为分子骨干的键数b$$b=\frac12(8n-g)$$对于缺电子化合物，其电子数目不足以全部形成共价单键，部分形成三中心二电子键时，由于三个原子共享两个电子，相当于每个3c-2e键补偿了四个电子，即键数为2。
\
g为以下三部分之和：
- 组成分子骨干$M_n$的n个原子的价电子数
- 围绕分子骨干$M_n$的配位体提供的电子数
- 化合物所带有的正负电荷数

> 例如对硼烷$B_{12}H_{12}^{2-}$进行分析：$$g=12\times 3 +12\times 1 +2 =50 $$ 
>$\quad B_{12}$分子骨干的键数为$$b=\frac12(8\times 2 -50 )=23\quad $$ 对于硼烷来说，应当有：$$b=2t+y=23$$ 

由上述例子不难看出，对于封闭式硼烷和碳硼烷，由于$g=4n+2$，故有$$b=\frac12[8n-(4n+2)]=2n-1$$而封闭式硼烷的s和x均为0，故$$b=2t+y=23$$即$$2t+y=2n-1$$而在$B_nH_n^{2-}$中，价电子对总数为$\frac12(4n+2)=2n+1$，其中n个电子对用于形成n个B-B键，剩余n+1个电子对用于$B_n$骨架，即$$t+y=n+1$$即可解出分子的styx数码。
    ''',unsafe_allow_html=True)
    st.write(r''' 使用暴力枚举的方法很容易得到合理的整数解，这里不再作代码展示，读者可以自行练习。
    ''',unsafe_allow_html=True)
    st.write(r''' #### 4.8 非金属元素的结构特征
##### 4.8.1 非金属单质的结构特征
##### 4.8.2 非金属化合物的结构特征

1)单质结构对化合物的成键作用

2)d轨道是否参与成键

3)从分子的几何构型了解成键情况

4)从分子的成键情况了解分子性质

#### 4.9 共价键的键长和键能

通过一系列实验我们可以测定分子中成键原子间的距离，当不同分子中两个原子形成相同类型的化学键时，键长相近。根据键长数据可以获得原子的共价半径。
当我们利用原子共价半径计算键长时，应当考虑以下两种情况：
- 异核原子间键长的计算值通常比实验测定值稍大
- 同一种化学键对不同分子有其特殊性，键长也略有不同

按照化学的观点双原子分子在标准状态下解离的解离能就是它的键能，对于多原子分子而言则不然，因为当分子断开一个键分成两部分时，每一部分都有可能发生键或电子的重排。
从热力学实验得到的键能数据，可以归纳出两条规律：
- 当同种原子键A-A和B-B改组成两个异种原子键A-B时，键能有一定的增加。*这可能于两个原子之间的电负性值之差有关*
- 当C=C双键改组成两个C-C单键时，键能总是有所增加。
    ''',unsafe_allow_html=True)
    if 'show_code' not in st.session_state:
        st.session_state.show_code = False

    # 在侧边栏创建一个复选框来控制代码的显示
    st.session_state.show_code = st.sidebar.checkbox('显示代码', value=st.session_state.show_code)
    if st.session_state.show_code:
        with open(__file__, 'r') as file:
            code = file.read()
        st.markdown("```python\n" + code + "\n```", unsafe_allow_html=True)