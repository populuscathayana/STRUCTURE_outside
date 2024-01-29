'''
Author: cathayana populuscathayana@gmail.com
Date: 2024-01-25 13:16:21
LastEditors: cathayana populuscathayana@gmail.com
LastEditTime: 2024-01-27 16:11:22
FilePath: /website_structure/STRUCTURE/GROUP2/page_2.py
Description: 

Copyright (c) 2024 by Cathayana, All Rights Reserved. 
'''
import importlib
import streamlit as st

def show():
    import streamlit as st
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    st.write(r'''
    
    
    # 2 原子的结构和性质
    ## 2.1 单电子原子的Schrodinger方程及其解
    ### 2.1.1 单电子原子Schrodinger方程形式
    对于单电子原子，Schrodinger方程中的哈密顿算子应当包含势能项：
    $$V = -\frac{ze^2}{4πε_0r}$$

    即：
    $$
    H = -\frac{h^2}{8π^2m}∇^2 + V
    $$

    相应的Schrodinger方程形式为：
    $$
    [-\frac{h^2}{8π^2μ}(\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2}) -  -\frac{ze^2}{4πε_0r}]ψ = Eψ
    $$

    （其中$∇^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{əy^2} + \frac{\partial^2}{\partial z^2}$为laplace算子，$μ = \frac{m_em_H}{m_e + m_H}$为约化质量）

    由于方程中势能项是电子与核距离r的函数，为便于求解，采用球坐标变换：
    $$
    \begin{cases}
    x = r\sinθ\cos\phi\\
    y = r\sinθ\sin\phi\\
    z = r\cos\phi
    \end{cases}
    $$

    相应的Schrodinger方程形式为：
    $$
    -\frac{\overline{h}^2}{2μ}[\frac{1}{r^2}\frac{\partial}{\partial r}(r^2\frac{\partial}{\partial r})+\frac{1}{r^2\sinθ}\frac{\partial}{\partialθ}(\sinθ\frac{\partial}{\partialθ})+\frac{1}{r^2\sin^2θ}\frac{\partial^2}{\partial\phi^2}]Ψ - \frac{ze^2}{4\pi\epsilon_0r}Ψ = EΨ
    $$
    ### 2.1.2 变数分离法和Schrodinger方程的一般解
    **变数分离法**

    变数分离法即将含有多个未知量的方程分解为若干个只含一个未知量的因式的乘积的方程变换。
    极坐标形式的单电子原子Schrodinger方程为一个二阶偏微分方程$Ψ(r,\theta,\phi) = 0$，需要采取变数分离法简化为三个二阶常微分方程以求解：设
    $$
    Ψ(r,\theta,\phi) = R(r)Y(\theta,\phi)
    $$

    代入Schrodinger方程，化简为：
    $$
    \frac{1}{R(r)}\frac{\partial}{\partial r}(r^2\frac{\partial}{\partial r})R(r) + \frac{2\mu ze^2r}{4\pi\epsilon_0\overline{h}^2} + \frac{2\mu r^2}{\overline{h}^2}E \\
    = -\frac{1}{Y(\theta,\phi)}[\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}(\sin\theta\frac{\partial}{\partial\theta}) + \frac{1}{\sin^2θ}\frac{\partial^2}{\partial\phi^2}]Y(\theta,\phi)
    $$

    若方程成立，需方程两边等于常数$k$，即：
    $$
    \frac{1}{R(r)}\frac{d}{dr}(r^2\frac{d}{dr})R(r) + \frac{2\mu ze^2r}{4\pi\epsilon_0\overline{h}^2} + \frac{2\mu r^2}{\overline{h}^2}E =k\\
    -\frac{1}{Y(\theta,\phi)}[\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}(\sin\theta\frac{\partial}{\partial\theta}) + \frac{1}{\sin^2θ}\frac{\partial^2}{\partial\phi^2}]Y(\theta,\phi) = k
    $$

    于是，我们得到了关于关于$r$的二阶常微分方程$R(r) = 0$，只需将$Y(\theta,\phi) = 0$分解为二阶常微分方程的乘积，即$Y(\theta,\phi) = \Theta(\theta)\Phi(\phi)$。参照上述方法，引入常数$m^2$，得：
    $$
    \frac{\sin\theta}{\Theta(\theta)}\frac{d}{d\theta}(\sin\theta \frac{d}{d\theta})\Theta(\theta) + k\sin^2 \theta = m^2\\
    -\frac{1}{\Phi(\phi)}\frac{d^2}{d\phi^2}\Phi(\phi) = m^2
    $$

    至此，我们获得了关于$r$，$\theta$，$\phi$的三个二阶常微分方程，逐个求解即可得到$Ψ(r,\theta,\phi)$

    **Φ方程的解**

    $\Phi$方程为线性微分方程，故两个特解（复数解）$\Phi_{|m|} = Ae^{i|m|\phi},\Phi_{-|m|} = Ae^{-i|m|\phi}$的线性组合任然是方程的解（实数解）：
    $$
    \Phi_{|m|} + \Phi_{-|m|} = Ae^{i|m|\phi} + Ae^{-i|m|\phi} = c_1\cos|m|\phi\\
    \Phi_{|m|} - \Phi_{-|m|} = Ae^{i|m|\phi} - Ae^{-i|m|\phi} = c_2\cos|m|\phi
    $$

    对于常数m，由$\Phi$的周期性，需满足：
    $$
    \Phi_m(\phi) = Ae^{i|m|\phi} = Ae^{i|m|\phi}e^{2\pi im} = Ae^{i|m|(\phi + 2\pi)} = \Phi_m(\phi + 2\pi)\\
    \Leftrightarrow \quad e^{2\pi im} = \cos 2m\pi + i\sin2m\pi = 1
    $$
    即$m = 0,\pm1,\pm2,\cdots$，我们引入了$\Psi$的量子化，并将m称为磁量子数

    由归一化条件$\int |\Psi|^2d\tau = 1$得$A = \frac{1}{\sqrt{2\pi}}$，$c_1 = \frac{1}{\sqrt{\pi}}$，复数解：
    $$
    \Psi_{|m|} = \frac{1}{\sqrt{2\pi}}e^{i|m|\phi},\Psi_{-|m|} = \frac{1}{\sqrt{2\pi}}e^{-i|m|\phi}
    $$
    实数解：
    $$
    \Psi'_{|m|} = \frac{1}{\sqrt{\pi}}\cos|m|\psi,\Psi'_{-|m|} = \frac{1}{\sqrt{\pi}}\sin|m|\psi
    $$

    **Θ方程的解**

    Θ方程为缔合（联属）勒让德（Legender）方程，常数$k$满足条件：
    $$
    k = l(l + 1), l = 0, 1, 2, \cdots, l \geq |m|
    $$

    如此便引入了$\Theta$的量子化，$l$被称为角量子数。由于$l \geq |m|$，$\Theta$写成$\Theta_{l,m}$的形式，下面给出缔合勒让德函数的一般形式，详细推导可参阅常微分方程教材
    $$
    \Theta_{l,m} = [\frac{2l+1}{2}\frac{(l-|m|)!}{(l+|m|)!}]^\frac{1}{2}p_l^{|m|}\cos\theta\\
    p^{|m|}_l\cos\theta = \frac{(1-\cos^2\theta)^{\frac{|m|}{2}}}{2^l l!}\frac{d^{l+|m|}}{d\cos^{l+|m|}}(\cos^2\theta - 1)^l
    $$

    **R方程的解**

    R方程为缔合（联属）拉盖尔（Laguerre）方程，引入常数$n$（$n=1, 2, \cdots, n > l$），可以得到能量的解：
    $$
    E_n = -\frac{z^2}{n^2}\frac{me^2}{2\overline{h}^2} = -\frac{z^2}{n^2}\frac{e^2}{2a_0^2}
    $$

    其中$a_0 = \frac{\overline{h}^2}{me^2}$称为玻尔半径

    径向波函数R的解即缔合拉盖尔函数，一般形式如下（详细推导可参阅常微分方程教材）：
    $$
    R_{n,l}(r) = -[(\frac{2z}{na_0})^3\frac{(n-l-1)!}{2n[(n+l)!]^3}]^{\frac{1}{2}}e^{-\frac{\rho}{2}}\rho^lL_{n+l}^{2l+1}(\rho)\\
    L_{n+l}^{2l+1}(\rho) = \frac{d^{2l+1}}{d\rho^{2l+1}}[e^{\rho}\frac{d^{n+l}}{d\rho^{n+l}}(e^{-\rho}\rho^{n+l})],\quad \rho = \frac{2zr}{na_0}
    $$

    至此，我们引入了第三个常数n，完成了能量E和径向波函数R的量子化，n称为主量子数
    ### 2.1.3 结论：波函数与能量的量子化
    在上一节中，我们得到了单电子原子波函数与能量的解，这些解中含有整数n，l，m，这些整数决定了电子的轨道与能量均是分立的。

    需要强调的是，归一化的波函数$\Psi_{n,l,m}(r,\theta,\phi)$描述了电子的运动状态，$\int\Psi_{n,l,m}^2(r,\theta,\phi)d\tau$描述了电子出现在一定空间范围内的概率，$E_n$则描述了电子具有的能量

    根据量子数的取值规则：
    $$
    n = 1,2,\cdots\\
    l = 0,1,2,\cdots,(n-1)\\
    m = 0,\pm1,\pm2,\cdots,\pm l
    $$

    对$n$的每个取值，有序数组$(n,m,l)$共有$n^2$个取值，即每一种能量状态的原子，其电子由不止一种可能的运动状态，这种现象称为能量简并，这些运动状态称为简并态，其数目称为简并度

    前文中，能量算符作用于氢原子波函数得到决定能量状态的主量子数n，相似地，角量子数l与磁量子数m也具有物理意义。若将角动量平方算符作用与氢原子波函数，可得到角动量绝对值：
    $$
    |M| = \sqrt{l(l+1)}\frac{h}{2\pi},\quad l = 0,1,2,\cdots,(n-1)
    $$

    可见，角量子数决定原子轨道角动量的大小。

    将角动量在z方向的分量算符作用与氢原子波函数，该分量有确定值：
    $$
    M_z = m\frac{h}{2\pi},\quad m = 0,\pm1,\pm2,\cdots,\pm l
    $$

    在磁场中，z方向为磁场方向，因此m决定电子轨道角动量在z方向的分量，也决定磁矩在磁场方向的分量$\mu_z = -m\beta_e$

    将$(n,m,l)$的取值代入归一化波函数$\Psi_{n,l,m}(r,\theta,\phi)$，即可得到描述电子运动状态的具体方程。将$l = 0,1,2,3,\cdots$的状态分别标记为s，p，d，f……，对于p轨道，再根据直角坐标系与球坐标系的变换关系，用x，y，z表示p轨道的三种形式，产生$\Psi_{s},\Psi_{2px},\cdots$的记法
    ### 2.1.4 波函数与电子云图形
    $\Psi_{n,l,m}$是三维空间种的函数，通常用$\Psi - r$和$\Psi^2 - r$图在二维上表示沿通过坐标原点（原子核）的某一直线上的概率密度，引入径向分布函数：
    $$
    D(r) = R^2(r)\cdot r^2
    $$

    表示电子在半径为r的球壳中出现的概率密度，D(r)对r作图，称为径向分布图

    此外，还可以用原子轨道等值线图表示某一截面上的概率密度，用原子轨道等值面图表示概率密度值为一定值的曲面形状，进而抽象出完整的电子云图形
    # 2.2 量子数的意义
    ## 2.2.1 主量子数***n***
    在解R方程过程中获得主量子数$n$，其只能取正整数。为使
    $$E_n=-\frac{\mu e^4}{8\epsilon _0^2 h^2} \frac{Z^2}{n^2}$$
    对于H原子，$Z=1$，各状态能量$E_n=-13.595/n^2eV$
    > - 对于氢原子基态能量为$-13.595eV$而有零点能
    > - 解释：位力定理——对势能服从$r^n$规律的体系，其平均势能$〈V〉$与平均动能$〈T〉$符合：$〈T〉=\frac{1}{2}n〈V〉$
    > - 对于氢原子，平均动能$〈T〉=-\frac{1}{2}〈V〉=13.595eV＞0$
    ## 2.2.2 角量子数***l***
    将角动量平方算符作用于氢原子波函数$\psi _{nlm}$，获得
    $$
    M^2 =l(l+1)(\frac{h}{2\pi})^2，l=0,1,2,…,n-1
    $$
    原子角动量与磁矩有关，磁矩$\vec{\mu}=-\frac{e}{2m_e}\vec{M}$，磁矩大小|$μ$|=$\sqrt{l(l+1)}\frac{eh}{4\pi m_e}$=$\sqrt{l(l+1)}\beta _e$，其中$\beta _e$为Bohr磁子，说明磁矩方向量子化。
    ## 2.2.3 磁量子数***m***
    角动量在z方向分量算符作用于氢原子***Φ***方程获得
    $$
    M_z=m\frac{h}{2\pi},m=0,±1,±2,…,±l
    $$
    磁矩在磁场方向上的分量为
    $$
    \mu =-m\beta _e
    $$
    **Zeeman效应：** $m$决定轨道角动量和轨道磁矩在磁场方向上的分量，所以存在外磁场时，氢原子$n, l$相同，$m$不同的状态，能量不同，即磁场中的能级分裂
    ## 2.2.4 自旋量子数***s***
    电子自旋运动：跃迁光谱出现双线现象
    自旋角动量大小:
    $$
    |M_s|=\sqrt{s(s+1)}\frac{h}{2\pi}，s=\frac{1}{2}
    $$
    在磁场方向分量:
    $$
    M_{sz}=m_s\frac{h}{2\pi}，m_s=±\frac{1}{2}
    $$
    自旋磁矩$\mu _s=g_e\sqrt{s(s+1)}\beta _e$，在磁场方向分量$\mu _{sz}=g_em_s\beta _e$，电子自旋因子$g_e=2.00232$。
    ## 2.2.4 总量子数***j***和总磁量子数$m_j$
    电子的总角动量
    $$
    \vec{M}_j=\vec{M}+\vec{M}_s，|M_j|=\sqrt{j(j+1)}\frac{h}{2\pi}，j=l+s,l+s−1,…,|l−s|
    $$
    沿磁场方向的分量
    $$
    M_{jz}=m_j\frac{h}{2\pi}，m_j=±\frac{1}{2},±\frac{3}{2},…,±j
    $$
    # 2.3 波函数和电子云的图形
    ## 2.3.1  $\psi -r$图和${\psi}^2-r$图
    $$\psi _{1s}=(\frac{z^3}{\pi {a_0}^3})^{1/2} e^{-\frac{Zr}{a_0}}$$
    $$\psi _{2s}=\frac{1}{4}(\frac{z^3}{2\pi {a_0}^3})^{1/2} (2-\frac{Zr}{a_0})e^{-\frac{Zr}{2a_0}}$$''')
    st.code('''
    import numpy as np
    import matplotlib.pyplot as plt
    Z = 1
    a_0 = 1  #Unit: a.u.
    xlim = st.slider('最远距离',0,20,5)
    r = np.linspace(0,xlim,100*xlim)
    fig, ax = plt.subplots()
    ax.set_title("Radial Wave function")
    plt.xlim(0, xlim)
    plt.xlabel('r')
    plt.ylabel(r'$\psi$')
    ax.plot(r, 1/np.sqrt(np.pi)*(Z/a_0)**1.5*np.exp(-Z/a_0*r), label = '1s')
    ax.plot(r, 1/(4*np.sqrt(2*np.pi))*(Z/a_0)**1.5*np.exp(-Z/a_0*r/2)*(2-Z/a_0*r), label = '2s')
    ax.plot(r, 1/(4*np.sqrt(2*np.pi))*(Z/a_0)**1.5*np.exp(-Z/a_0*r/2)*(Z/a_0*r), label = '2p')
    ax.plot(r, 1/(81*np.sqrt(3*np.pi))*(Z/a_0)**1.5*np.exp(-Z/a_0*r/3)*(27-18*Z/a_0*r+2*(Z/a_0*r)**2), label = '3s')
    ax.plot(r, 1*np.sqrt(2)/(81*np.sqrt(np.pi))*(Z/a_0)**1.5*np.exp(-Z/a_0*r/3)*(6-Z/a_0*r)*Z/a_0*r, label = '3p')
    ax.plot(r, 1*np.sqrt(2)/(81*np.sqrt(np.pi))*(Z/a_0)**1.5*np.exp(-Z/a_0*r/3)*(Z/a_0*r)**2, label = '3dxy')
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    ''')

    Z = 1
    a_0 = 1  #Unit: a.u.
    xlim = st.slider('最远距离',0,20,5)
    r = np.linspace(0,xlim,100*xlim)
    fig, ax = plt.subplots()
    ax.set_title("Radial Wave function")
    plt.xlim(0, xlim)
    plt.xlabel('r')
    plt.ylabel(r'$\psi$')
    ax.plot(r, 1/np.sqrt(np.pi)*(Z/a_0)**1.5*np.exp(-Z/a_0*r), label = '1s')
    ax.plot(r, 1/(4*np.sqrt(2*np.pi))*(Z/a_0)**1.5*np.exp(-Z/a_0*r/2)*(2-Z/a_0*r), label = '2s')
    ax.plot(r, 1/(4*np.sqrt(2*np.pi))*(Z/a_0)**1.5*np.exp(-Z/a_0*r/2)*(Z/a_0*r), label = '2p')
    ax.plot(r, 1/(81*np.sqrt(3*np.pi))*(Z/a_0)**1.5*np.exp(-Z/a_0*r/3)*(27-18*Z/a_0*r+2*(Z/a_0*r)**2), label = '3s')
    ax.plot(r, 1*np.sqrt(2)/(81*np.sqrt(np.pi))*(Z/a_0)**1.5*np.exp(-Z/a_0*r/3)*(6-Z/a_0*r)*Z/a_0*r, label = '3p')
    ax.plot(r, 1*np.sqrt(2)/(81*np.sqrt(np.pi))*(Z/a_0)**1.5*np.exp(-Z/a_0*r/3)*(Z/a_0*r)**2, label = '3dxy')
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    st.code('''#Unit: a.u.
    Z = 1
    a_0 = 0.529
    xlim = st.slider('最远距离 ',0,20,5)
    r = np.linspace(0,xlim,100*xlim)
    fig, ax = plt.subplots()
    ax.set_title("Radial Wave function of 1s")
    plt.xlim(0, xlim)
    plt.ylim(0, 0.6)
    plt.xlabel('r')
    plt.ylabel(r'$\psi$ or $\psi$$^2$')
    ax.plot(r, 0.56*np.exp(-r), label = r'$\psi$')
    ax.plot(r, (0.56*np.exp(-r))**2, label = r'$\psi$$^2$')
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    ''')

    #Unit: a.u.
    Z = 1
    a_0 = 0.529
    xlim = st.slider('最远距离 ',0,20,5)
    r = np.linspace(0,xlim,100*xlim)
    fig, ax = plt.subplots()
    ax.set_title("Radial Wave function of 1s")
    plt.xlim(0, xlim)
    plt.ylim(0, 0.6)
    plt.xlabel('r')
    plt.ylabel(r'$\psi$ or $\psi$$^2$')
    ax.plot(r, 0.56*np.exp(-r), label = r'$\psi$')
    ax.plot(r, (0.56*np.exp(-r))**2, label = r'$\psi$$^2$')
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    st.markdown(r'''
    ## 2.3.2 径向分布图 
    径向分布函数$D=r^2R^2=4\pi r^2\psi _s^2$
    >1. $（n−l）$个极大值峰 &$（n−l−1）$个节面（不算原点）
    2. $n$不变情况下，随$l$增加，主峰位置向核移动，峰数目减小，最内层峰离核越近
    3. $l$不变情况下，随$n$增加，电子沿$r$扩展得越远
    ''')
    st.code('''
    import numpy as np
    import matplotlib.pyplot as plt
    Z = 1
    a_0 = 1 #Unit: a.u.
    xlim = st.slider('最远距离  ',0,30,15)
    r = np.linspace(0,xlim,xlim*50)
    fig, ax = plt.subplots()
    plt.xlabel('r')
    plt.ylabel('r$^2$R$^2$')
    plt.xlim(0,xlim)
    ax.set_title("Radial Distribution Function")
    ax.plot(r, 4*np.pi*r**2*(1/np.pi*(Z/a_0)**3*np.exp(-2*Z/a_0*r)), label = '1s')
    ax.plot(r, 4*np.pi*r**2*(1/(4*np.sqrt(2*np.pi))*(Z/a_0)**1.5*np.exp(-Z/a_0*r/2)*(2-Z/a_0*r))**2, label = '2s')
    ax.plot(r, 4*np.pi*r**2*((1/(4*np.sqrt(2*np.pi))*(Z/a_0)**1.5*np.exp(-Z/a_0*r/2)*(Z/a_0*r))**2), label = '2p')
    ax.plot(r, 4*np.pi*r**2*((1/(81*np.sqrt(3*np.pi))*(Z/a_0)**1.5*np.exp(-Z/a_0*r/3)*(27-18*Z/a_0*r+2*(Z/a_0*r)**2))**2), label = '3s')
    ax.plot(r, 4*np.pi*r**2*((1*np.sqrt(2)/(81*np.sqrt(np.pi))*(Z/a_0)**1.5*np.exp(-Z/a_0*r/3)*(6-Z/a_0*r)*Z/a_0*r)**2), label = '3p')
    ax.plot(r, 4*np.pi*r**2*((1*np.sqrt(2)/(81*np.sqrt(np.pi))*(Z/a_0)**1.5*np.exp(-Z/a_0*r/3)*(Z/a_0*r)**2)**2), label = '3dxy')
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    ''')

    Z = 1
    a_0 = 1 #Unit: a.u.
    xlim = st.slider('最远距离  ',0,30,15)
    r = np.linspace(0,xlim,xlim*50)
    fig, ax = plt.subplots()
    plt.xlabel('r')
    plt.ylabel('r$^2$R$^2$')
    plt.xlim(0,xlim)
    ax.set_title("Radial Distribution Function")
    ax.plot(r, 4*np.pi*r**2*(1/np.pi*(Z/a_0)**3*np.exp(-2*Z/a_0*r)), label = '1s')
    ax.plot(r, 4*np.pi*r**2*(1/(4*np.sqrt(2*np.pi))*(Z/a_0)**1.5*np.exp(-Z/a_0*r/2)*(2-Z/a_0*r))**2, label = '2s')
    ax.plot(r, 4*np.pi*r**2*((1/(4*np.sqrt(2*np.pi))*(Z/a_0)**1.5*np.exp(-Z/a_0*r/2)*(Z/a_0*r))**2), label = '2p')
    ax.plot(r, 4*np.pi*r**2*((1/(81*np.sqrt(3*np.pi))*(Z/a_0)**1.5*np.exp(-Z/a_0*r/3)*(27-18*Z/a_0*r+2*(Z/a_0*r)**2))**2), label = '3s')
    ax.plot(r, 4*np.pi*r**2*((1*np.sqrt(2)/(81*np.sqrt(np.pi))*(Z/a_0)**1.5*np.exp(-Z/a_0*r/3)*(6-Z/a_0*r)*Z/a_0*r)**2), label = '3p')
    ax.plot(r, 4*np.pi*r**2*((1*np.sqrt(2)/(81*np.sqrt(np.pi))*(Z/a_0)**1.5*np.exp(-Z/a_0*r/3)*(Z/a_0*r)**2)**2), label = '3dxy')
    ax.legend()
    st.pyplot(fig, use_container_width=True)


    st.markdown(r'''
    ## 2.3.3 原子轨道等值线图
    根据求解薛定谔方程，我们可以得到氢原子波函数的通式：
    $$
    \begin{align}
    {\psi_{nlm}(r,\theta,\varphi)\\ =R_{nl}(r)\cdot Y_{lm}(\theta,\varphi)\\=\sqrt{(\frac{2}{na})^3\frac{(n-l-1)}{2n(n+l)!}}(\frac{2r}{na})^le^{-\frac{r}{na}}L_{n-l-1}^{2l+1}(\frac{2r}{na})\cdot Y_{lm}(\theta,\varphi)}
    \end{align}
    $$
    其中：$a=\frac{4\pi\varepsilon_0\hbar^2}{me^2}$。
    由此，以$2p_z$轨道为例，可以画出：
    ### 原子轨道二维/三维等值线图''')
    st.code('''
from scipy.special import factorial, genlaguerre, sph_harm
    from IPython.display import HTML
    plt.ion()
    X_limit = st.slider('box_size',10,100,30)
    def draw_orbit(n, l, m):
        #图像大小
        limit = X_limit

        #波函数定义
        a0 = 1 #图像归一化
        def hydrogen_wave_function(n, l, m):
            #径向部分R(r)
            def R(r):
                factor = np.sqrt((2./(n * a0))**3 * factorial(n-l-1) / (2 * n * factorial(n + l)))
                rho = 2 * r / (n * a0)  # 为简化计算，rho=2r/na
                return factor * (rho**l) * np.exp(-rho/2) * genlaguerre(n-l-1, 2*l+1)(rho)

            #角向部分Y(theta,phi)
            def Y(theta, phi):
                return sph_harm(m, l, phi, theta)

            #波函数值=径向部分*角向部分
            return lambda r, theta, phi: R(r) * Y(theta, phi)

        #线性组合为实数
        psi_c1 = hydrogen_wave_function(n, l, m)
        psi_c2 = hydrogen_wave_function(n, l, -m)
        if m > 0:
            psi_r = lambda r, theta, phi: ((-1)**m * psi_c1(r, theta, phi) + psi_c2(r, theta, phi)) / np.sqrt(2)
        elif m < 0:
            psi_r = lambda r, theta, phi: 1j * ((-1)**(m+1) * psi_c1(r, theta, phi) + psi_c2(r, theta, phi)) / np.sqrt(2)
        else:
            psi_r = psi_c1

        # 图像构筑
        step = limit / 100
        x = np.arange(-limit, limit, step)
        y = np.arange(-limit, limit, step)
        X, Y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(8,8)) 
        plt.ylabel('z')
        plt.xlabel('x')
        plt.grid(True)

        # 绘制等值线
        Z = np.real(psi_r(np.sqrt(X**2 + Y**2), np.arccos(Y / np.sqrt(X**2 + Y**2)), 1/6 * np.pi))

        C = plt.contour(X, Y, Z, 20, cmap='viridis')
        plt.clabel(C, inline=True, fontsize=5)
        st.pyplot(fig, use_container_width=True)

    #原子轨道参数
    n = st.number_input('Enter principal quantum number n for 2D Visualization:', min_value=1, max_value=10, value=1, step=1)
    l = st.number_input('Enter azimuthal quantum number l for 2D Visualization:', min_value=0, max_value=n-1, value=0, step=1)
    m = st.number_input('Enter magnetic quantum number m for 2D Visualization:', min_value=-l, max_value=l, value=0, step=1)

    if st.button('Show Orbit'):
        draw_orbit(n, l, m)

    st.markdown(r'### 原子轨道三维等值面图')
    from skimage.measure import marching_cubes
    X_limit = st.slider('box_size ',10,100,15)
    def new_fig_and_ax(plot_range=X_limit):
        fig = plt.figure(figsize=(8, 8),dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=30, azim=30)
    
        ax.set_xlabel("$x$", fontsize = 5)
        ax.set_ylabel("$y$", fontsize = 5)
        ax.set_zlabel("$z$", fontsize = 5)
        
        ax.set_xlim(-plot_range, plot_range)
        ax.set_ylim(-plot_range, plot_range)
        ax.set_zlim(-plot_range, plot_range)

        ticks = np.linspace(-plot_range, plot_range, 5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_zticks(ticks)
    
        return fig, ax
    def draw_orbit_3D(n, l, m):
        #等值面取值
        iso_value = 2e-10

        a0 = 1 #图像归一化
        #波函数定义
        def hydrogen_wave_function(n, l, m):
            #径向部分R(r)
            def R(r):
                factor = np.sqrt((2./(n * a0))**3*factorial(n-l-1)/(2*n*factorial(n + l)))
                rho = 2*r/(n*a0) # 为简化计算，rho=2r/na
                return factor*(rho**l)*np.exp(-rho/2)*genlaguerre(n-l-1, 2*l+1)(rho)
            #角向部分Y(theta,phi)
            def Y(theta, phi):
                return sph_harm(m, l, phi, theta)
            #波函数值=径向部分*角向部分
            return lambda r, theta, phi: R(r)*Y(theta, phi)

        #线性组合为实数
        psi_c1 = hydrogen_wave_function(n,l,m)
        psi_c2 = hydrogen_wave_function(n,l,-m)
        if m>0:
            psi_r = lambda r, theta, phi: ((-1)**m*psi_c1(r,theta,phi)+psi_c2(r,theta,phi))/np.sqrt(2)
        elif m<0:
            psi_r = lambda r, theta, phi: 1j*((-1)**(m+1)*psi_c1(r,theta,phi)+psi_c2(r,theta,phi))/np.sqrt(2)
        else:
            psi_r = psi_c1
        #图像构筑
        limit = X_limit #坐标范围，单位为a0
        n_points = 50
        vec = np.linspace(-limit, limit, n_points)
        X, Y, Z = np.meshgrid(vec, vec, vec)
        #球坐标变换
        R = np.sqrt(X**2+Y**2+Z**2)
        THETA = np.arccos(Z/R)
        PHI = np.arctan2(Y, X)
        psi_values = psi_r(R, THETA, PHI)
        prob_dens = np.abs(psi_values)**2

        #得到等值面坐标
        step =2*limit/(n_points-1)
        verts, faces, _, _ = marching_cubes(prob_dens,level=iso_value,spacing=(step, step, step),)
        verts -= limit
        verts[:, [0, 1]] = verts[:, [1, 0]]

        #图像调节
        m = np.max(verts)
        #绘图
        fig, ax = new_fig_and_ax()
        iso_surface = ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],lw=0,cmap="coolwarm_r")
        
        st.pyplot(fig, use_container_width=True)
    n_3D = st.number_input('Enter principal quantum number n for 3D Visualization:', min_value=1, max_value=10, value=1, step=1)
    l_3D = st.number_input('Enter azimuthal quantum number l for 3D Visualization:', min_value=0, max_value=n_3D-1, value=0, step=1)
    m_3D = st.number_input('Enter magnetic quantum number m for 3D Visualization:', min_value=-l_3D, max_value=l_3D, value=0, step=1)

    if st.button('Show 3D Orbit'):
        draw_orbit_3D(n_3D, l_3D, m_3D)
    ''')

    from scipy.special import factorial, genlaguerre, sph_harm
    from IPython.display import HTML
    plt.ion()
    X_limit = st.slider('box_size',10,100,30)
    def draw_orbit(n, l, m):
        #图像大小
        limit = X_limit

        #波函数定义
        a0 = 1 #图像归一化
        def hydrogen_wave_function(n, l, m):
            #径向部分R(r)
            def R(r):
                factor = np.sqrt((2./(n * a0))**3 * factorial(n-l-1) / (2 * n * factorial(n + l)))
                rho = 2 * r / (n * a0)  # 为简化计算，rho=2r/na
                return factor * (rho**l) * np.exp(-rho/2) * genlaguerre(n-l-1, 2*l+1)(rho)

            #角向部分Y(theta,phi)
            def Y(theta, phi):
                return sph_harm(m, l, phi, theta)

            #波函数值=径向部分*角向部分
            return lambda r, theta, phi: R(r) * Y(theta, phi)

        #线性组合为实数
        psi_c1 = hydrogen_wave_function(n, l, m)
        psi_c2 = hydrogen_wave_function(n, l, -m)
        if m > 0:
            psi_r = lambda r, theta, phi: ((-1)**m * psi_c1(r, theta, phi) + psi_c2(r, theta, phi)) / np.sqrt(2)
        elif m < 0:
            psi_r = lambda r, theta, phi: 1j * ((-1)**(m+1) * psi_c1(r, theta, phi) + psi_c2(r, theta, phi)) / np.sqrt(2)
        else:
            psi_r = psi_c1

        # 图像构筑
        step = limit / 100
        x = np.arange(-limit, limit, step)
        y = np.arange(-limit, limit, step)
        X, Y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(8,8)) 
        plt.ylabel('z')
        plt.xlabel('x')
        plt.grid(True)

        # 绘制等值线
        Z = np.real(psi_r(np.sqrt(X**2 + Y**2), np.arccos(Y / np.sqrt(X**2 + Y**2)), 1/6 * np.pi))

        C = plt.contour(X, Y, Z, 20, cmap='viridis')
        plt.clabel(C, inline=True, fontsize=5)
        st.pyplot(fig, use_container_width=True)

    #原子轨道参数
    n = st.number_input('Enter principal quantum number n for 2D Visualization:', min_value=1, max_value=10, value=1, step=1)
    l = st.number_input('Enter azimuthal quantum number l for 2D Visualization:', min_value=0, max_value=n-1, value=0, step=1)
    m = st.number_input('Enter magnetic quantum number m for 2D Visualization:', min_value=-l, max_value=l, value=0, step=1)

    if st.button('Show Orbit'):
        draw_orbit(n, l, m)

    st.markdown(r'''### 原子轨道三维等值面图''')
    from skimage.measure import marching_cubes
    X_limit = st.slider('box_size ',10,100,15)
    def new_fig_and_ax(plot_range=X_limit):
        fig = plt.figure(figsize=(8, 8),dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=30, azim=30)
    
        ax.set_xlabel("$x$", fontsize = 5)
        ax.set_ylabel("$y$", fontsize = 5)
        ax.set_zlabel("$z$", fontsize = 5)
        
        ax.set_xlim(-plot_range, plot_range)
        ax.set_ylim(-plot_range, plot_range)
        ax.set_zlim(-plot_range, plot_range)

        ticks = np.linspace(-plot_range, plot_range, 5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_zticks(ticks)
    
        return fig, ax
    def draw_orbit_3D(n, l, m):
        #等值面取值
        iso_value = 2e-10

        a0 = 1 #图像归一化
        #波函数定义
        def hydrogen_wave_function(n, l, m):
            #径向部分R(r)
            def R(r):
                factor = np.sqrt((2./(n * a0))**3*factorial(n-l-1)/(2*n*factorial(n + l)))
                rho = 2*r/(n*a0) # 为简化计算，rho=2r/na
                return factor*(rho**l)*np.exp(-rho/2)*genlaguerre(n-l-1, 2*l+1)(rho)
            #角向部分Y(theta,phi)
            def Y(theta, phi):
                return sph_harm(m, l, phi, theta)
            #波函数值=径向部分*角向部分
            return lambda r, theta, phi: R(r)*Y(theta, phi)

        #线性组合为实数
        psi_c1 = hydrogen_wave_function(n,l,m)
        psi_c2 = hydrogen_wave_function(n,l,-m)
        if m>0:
            psi_r = lambda r, theta, phi: ((-1)**m*psi_c1(r,theta,phi)+psi_c2(r,theta,phi))/np.sqrt(2)
        elif m<0:
            psi_r = lambda r, theta, phi: 1j*((-1)**(m+1)*psi_c1(r,theta,phi)+psi_c2(r,theta,phi))/np.sqrt(2)
        else:
            psi_r = psi_c1
        #图像构筑
        limit = X_limit #坐标范围，单位为a0
        n_points = 50
        vec = np.linspace(-limit, limit, n_points)
        X, Y, Z = np.meshgrid(vec, vec, vec)
        #球坐标变换
        R = np.sqrt(X**2+Y**2+Z**2)
        THETA = np.arccos(Z/R)
        PHI = np.arctan2(Y, X)
        psi_values = psi_r(R, THETA, PHI)
        prob_dens = np.abs(psi_values)**2

        #得到等值面坐标
        step =2*limit/(n_points-1)
        verts, faces, _, _ = marching_cubes(prob_dens,level=iso_value,spacing=(step, step, step),)
        verts -= limit
        verts[:, [0, 1]] = verts[:, [1, 0]]

        #图像调节
        m = np.max(verts)
        #绘图
        fig, ax = new_fig_and_ax()
        iso_surface = ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],lw=0,cmap="coolwarm_r")
        
        st.pyplot(fig, use_container_width=True)
    n_3D = st.number_input('Enter principal quantum number n for 3D Visualization:', min_value=1, max_value=10, value=1, step=1)
    l_3D = st.number_input('Enter azimuthal quantum number l for 3D Visualization:', min_value=0, max_value=n_3D-1, value=0, step=1)
    m_3D = st.number_input('Enter magnetic quantum number m for 3D Visualization:', min_value=-l_3D, max_value=l_3D, value=0, step=1)

    if st.button('Show 3D Orbit'):
        draw_orbit_3D(n_3D, l_3D, m_3D)

    st.markdown(r'''
    # 2.4 多电子原子的结构
    ## 2.4.1 Schrӧdinger方程及近似解
    对于原子序数为$Z$、含$n$个电子的原子（不考虑电子自旋及相互作用）：  
    $$(-\frac{h^2}{8\pi ^2 m}\sum_{i=1}^{N}{\bigtriangledown _i ^2}-\sum_{i=1}^{N}{\frac{Ze^2}{4\pi\epsilon _0r_i}}+\frac{1}{2}\sum_{i≠j}^{ }{\frac{Ze^2}{4\pi\epsilon _0r_{ij}}})\psi =E\psi$$  
    取原子单位$\frac{h}{2\pi}=1au$，$m=m_e=1$，$e=1au$，$4\pi \epsilon _0=1au$，化简获得：  
    $$(-\frac{1}{2}\sum_{i=1}^{N}{\bigtriangledown _i ^2}-\sum_{i=1}^{N}{\frac{Z}{r_i}}+\frac{1}{2}\sum_{i≠j}^{ }{\frac{1}{r_{ij}}})\psi =E\psi$$  
    其中$r_{ij}$涉及两个电子坐标，无法像之前进行分离变量。
    ### 方法1：电子独立运动模型（轨道近似模型）
    忽略电子间的相互作用（$\frac{1}{2}\sum_{i≠j}^{ }{\frac{1}{r_{ij}}}=0$），使之可分离变量。  
    获得体系近似波函数$\psi =\psi _1\psi _2…\psi _n$，体系总能量$E=E_1+E_2+…+E_n$。  
    ### 方法2：自洽场法（Hartree-Fock法）
    假定电子$i$处在原子核及其它$(n−1)$个电子的**平均势场**中运动，将其它电子可能出现位置平均，只与$r_i$有关的函数$V_{r_i}$，获得：        
    $$\widehat{H}_i=-\frac{1}{2}\bigtriangledown _i ^2 -\frac{Z}{r_i}+V(r_i)$$    
    利用其它$(n−1)$个电子波函数得$\widehat{H}_i$，获得$\psi _i^{(1)}$后计算$V^{(1)}{(r_i)}$。不断循环至$\psi _i^{(k+1)}$与$\psi _i^{(k)}$吻合为止。    
    获得整个体系波函数$\psi =\psi _1\psi _2…\psi _n$，体系总能量$=$获得单电子原子轨道能之和$−$多计算的电子间互斥能。                  
    ### 方法3：中心力场法
    将其他电子对第$i$个电子的排斥作用看成**球对称的、只与径向有关**的力场，于是第$i$个电子的势能函数：                
    $$V_i=-\frac{Z}{r_i}+\frac{\sigma _i}{r_i}=-\frac{Z-\sigma _i}{r_i}=-\frac{Z_i^*}{r_i}$$    
    式中$Z_i^*$为有效核电荷，$\sigma _i$为屏蔽常数。  
    获得多电子原子中第$i$个电子的**单电子薛定谔方程**：  
    $$(-\frac{1}{2}\bigtriangledown _i ^2-\frac{Z-\sigma _i}{r_i}\psi _i)=E_i\psi _i$$  
    和$\psi _i$对应的单电子原子轨道能为$E_i=-13.6\frac{(Z_i^*)^2}{n^2}$，$E_总$近似等于$\sum_{i=1}^{N}{E_i}$。  
    ## 2.4.2单电子原子轨道能和电子结合能
    #### **屏蔽效应**
    >将其它电子对第i个电子的排斥作用,归结为抵消一部分核电荷的作用,称为屏蔽效应。  
    #### **钻穿效应**            
    >1.主要是研究n相同I不同的轨道,哪个被屏蔽得多哪个被屏蔽得少,因而哪个轨道能量高哪个轨道能量低的问题。   
    2.当其它电子的数目与状态固定时,某个电子被屏蔽多少取决于该电子的状态。   
    3.对于该电子的状态来说,电子在原子核附近出现的几率大,受到的屏蔽效应小,轨道的能量低,但它却可能对其它电子起屏蔽作用,使其它电子的能量升高。    
    4.这种在核附近电子出现几率较大的轨道称为钻穿较深的轨道。   
    **注意：**  
    >1.屏蔽效应与钻穿效应都是影响轨道能量的重要因素,它们两者有联系又有区别,多电子原子中各电子的相互屏蔽,相互钻穿形成一个整体。  
    2.由于n不同,I不同的轨道钻穿效应引起的轨道能量差别,就可能使主量子数为n,角量子数I稍大的轨道能量**大于**主量子数为(n+1)，角量子数l较小的轨道能量。例如可能使nd的能量超过(n+1)s，而nf轨道钻穿得很少,被内部电子屏蔽得较完全,以致使nf的能量超过(n+2)s，这样就出现了能级交错的现象,即轨道次序颠倒的现象。   
    **最后要指出：**  
    >原子轨道的能量及次序不仅与屏蔽效应，钻穿效应有关，还与核电荷，主量子数，角量子数，电子自旋等等因素有关，也就是说决定轨道能量次序的因素是很复杂的，在各方面统筹考虑。  
    ## **2.5元素周期表与元素周期性质**
    ### **2.5.3电离能**
    >1.稀有气体的电离能总是处于最大值，而碱金属处于极小值。  
    2.除过渡金属元素外，同一周期元素的第一电离能基本上随原子序数增加而增加。  
    3.过渡金属元素的第一电离能不甚规则地随原子序数的增加而增加。  
    4.同一周期中，第一电离能的变化具有起伏性。 
    ### **2.5.4电子亲和能**
    - 定义：**气态**原子获得一个电子成为**一价负离子**时所放出的能量。  
    ### **2.5.5电负性**
    >1.电负性概念由Pauling提出，用以量度原子对成键电子吸引能力的相对大小。当A和B两种原子结合成双原子分子AB时，若A的电负性大，则生成分子的极性是A$\delta^-$B$\delta^+$，即A原子带有较多的负电荷，B原子带有较多的正电荷；反之，若B的电负性大，则生成分子的极性是A$\delta^+$B$\delta^-$。  
    2.Pauling的电负性标度**Xp**是用两元素形成化合物时的生成焓的数值来计算的。他认为，若A和B两个原子的电负性相同，A-B键的键能应为A-A键和B-B键键能的几何平均值。而大多数A-B键的键能均超过此平均值，此差值可用以测定A原子和B原子电负性的依据。  
    - 例如，H-F键的键能为565kJ/mol，而H-H和F-F键的键能分别为436kJ/mol和155kJ/mol。它们的几何平均值为260kJ/mol,差值Δ为305kJ/mol。根据一系列电负性数据拟合，可得方程  
    $$
    X_A-X_B=0.102Δ^{1/2}
    $$
    F的**Xp** 为4.0，这样H的电负性为
    $$
    4.0-0.102*(305)^{1/2}=2.2
    $$''')
    if 'show_code' not in st.session_state:
        st.session_state.show_code = False
        # 在侧边栏创建一个复选框来控制代码的显示
    st.session_state.show_code = st.sidebar.checkbox('显示代码', value=st.session_state.show_code)
    if st.session_state.show_code:
        with open(__file__, 'r') as file:
            code = file.read()
        st.markdown("```python\n" + code + "\n```", unsafe_allow_html=True)