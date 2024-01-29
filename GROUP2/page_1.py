'''
Author: cathayana populuscathayana@gmail.com
Date: 2024-01-25 13:16:21
LastEditors: cathayana populuscathayana@gmail.com
LastEditTime: 2024-01-27 21:44:45
FilePath: /website_structure/STRUCTURE/GROUP2/page_1.py
Description: 

Copyright (c) 2024 by Cathayana, All Rights Reserved. 
'''
import importlib
import streamlit as st

def show():
    import streamlit as st

    st.write(r'''
    # 1 量子力学基础知识
    ## 1.1 微观粒子的运动特征
    ### 1.1.1 黑体辐射与能量量子化
    **黑体:** 能全部吸收照射到它上面的各种波长辐射的物体
    **传统物理学在解释黑体辐射时的局限：**
    - Rayleigh-Jeans 公式不适用于长波区
    - Wien 公式不适用短波区''',unsafe_allow_html=True)
    st.markdown(r'''**Plank 的假设（1900年）**
    - 黑体中的原子或分子在辐射能量时作简谐振动
    - 只能发射或吸收频率为 $\nu$、能量为 $E=h\nu$ 的整数倍的电磁能  ''')
    st.markdown(r'''由此可以得出结论：
    - 频率为 $\nu$ 的振动的平均能量为$E=\frac{h\nu}{{e^\frac{h\nu}{kT}}-1}$
    - 单位时间、单位表面积上黑体辐射的能量为$E_\nu = \frac{2\pi h\nu^3}{c^2}({e^\frac{h\nu}{kT}-1})^{-1}$
    >由此可见，在定温下黑体辐射能量只与辐射频率有关。频率为 $\nu$ 的能量，其数值是不连续的，只能为 $h$ 的整数倍，称为能量量子化。''')
    st.code(
    '''    
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    h = 6.626*10**(-34)
    v = np.linspace(0.001,4,4000)
    k = 1.38*10**(-23)
    c = 3*10**8
    def E(v, T):
        return 10**(9)*2*np.pi*h*(v*10**14)**3/(np.exp(h*v*10**14/k/T)-1)/c**2
    fig, ax = plt.subplots()
    plt.xlabel('v/(10$^{14}$s$^{-1}$)')
    plt.ylabel('E$_v$/(10$^{-9}$J/m$^{-2}$)')
    plt.title("Energy distribution curve of black body at different temperatures", pad = 12, fontsize = 'large')

    ''', language='python')
    # Streamlit sliders for temperature selection
    temp1 = st.slider('Temperature 1 (K)', 500, 3000, 1000)
    temp2 = st.slider('Temperature 2 (K)', 500, 3000, 1500)
    temp3 = st.slider('Temperature 3 (K)', 500, 3000, 2000)
    st.code('''
    ax.plot(v, E(v, temp1), label=f'{temp1}K')
    ax.plot(v, E(v, temp2), label=f'{temp2}K')
    ax.plot(v, E(v, temp3), label=f'{temp3}K')
    plt.xlim((0, 4))
    plt.ylim((0, max(np.max(E(v,temp1)),np.max(E(v,temp2)),np.max(E(v,temp3)))*1.1))
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    ''', language='python')

    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    h = 6.626*10**(-34)
    v = np.linspace(0.001,4,4000)
    k = 1.38*10**(-23)
    c = 3*10**8
    def E(v, T):
        return 10**(9)*2*np.pi*h*(v*10**14)**3/(np.exp(h*v*10**14/k/T)-1)/c**2
    fig, ax = plt.subplots()
    plt.xlabel('v/(10$^{14}$s$^{-1}$)')
    plt.ylabel('E$_v$/(10$^{-9}$J/m$^{-2}$)')
    plt.title("Energy distribution curve of black body at different temperatures", pad = 12, fontsize = 'large')
    ax.plot(v, E(v, temp1), label=f'{temp1}K')
    ax.plot(v, E(v, temp2), label=f'{temp2}K')
    ax.plot(v, E(v, temp3), label=f'{temp3}K')
    plt.xlim((0, 4))
    plt.ylim((0, max(np.max(E(v,temp1)),np.max(E(v,temp2)),np.max(E(v,temp3)))*1.1))
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    st.markdown(r'''
    ### 1.1.2 光电效应和光子学说  
    **光电效应：** 光照射在金属表面上使金属发射出电子的现象。金属中的电子从照射光获得足够的能量而逸出金属，称为光电子。  1905 年，Einstein 提出光子学说：
    1. 光是一束光子流，每一种频率的光的能量都有一个最小单位，称为光子，光子的能量与光子的频率成正比，即  
    2. 光子不但有能量，而且有质量，但光子的静止质量为零。按相对论的质能联系定律，$E=mc^2$,光子的质量为  
    3. 光子具有一定的动量:  
    4. 光的强度取决于单位体积内光子的数目，即光子密度。
    将频率为 $\nu$ 的光照射到金属上，当金属中的电子受到光子撞击时，产生光电效应，光子消失，并把它的能量 $h$ 转移给电子。电子吸收的能量，一部分用于克服金属对它的束缚力，其余部分则变为光电子的动能。  
    $$E_k=h\nu-W$$
    ''')
    st.write(r'''
    ### 1.1.3 实物粒子的波粒二象性
    1924 年 de Broglie 受到光的波粒二象性理论的启发，提出实物微粒也有波性的假设。光的波粒二象性表明：光的波动性指光是电磁波，光的粒子性指光具有量子性。光在介质中以光速 $c$ 传播时，它的二象性通过下列公式联系着：<br>
    $$
    \begin{cases}
    E=h\nu\\
    m=\frac{h\nu}{c^2}\\
    p=\frac{h\nu}{c}\\
    \end{cases}
    $$
    <br>光的波性和粒性通过 Planck 常数 $h$ 联系起来。<br><br>静止质量为$m$的实物微粒也具有波粒二象性：<br>
    $$
    \begin{cases}
    E=h\nu\\
    p=\frac{h}{\lambda}=mv\\
    \end{cases}
    $$
    ''')
    st.image(
    'GROUP2/image/1.png'
    )
    st.write(r'''
    ### 1.1.4 不确定关系
    德国物理学家 W. K. Heisenberg于1927年提出：微观体系的共轭物理量不能同时被准确测定，若其中一种物理量被测定得越精确，则其共轭物理量被测定得越不精确，即不确定原理（uncertainty principle):<br>
    $$
    \begin{cases}
    \Delta x\Delta p \geq h\\
    \Delta E\Delta t \geq h/4\pi\\
    \end{cases}
    $$
    <br>比较微观粒子和宏观物体的特性，可见：
    >1. 宏观物体同时具有确定的坐标和动量，其运动规律可用经典物理学描述；而微观粒子没有同时确定的坐标和动量，其运动规律需用量子力学描述.
    2. 宏观物体有连续、可测的运动轨道，可追踪各个物体的运动轨迹来加以分辨；微观粒子具有概率分布的特性，不可能分辨出各个粒子的轨迹.
    3. 宏观物体可处于任意的能量状态，体系的能量可以为任意的、连续变化的数值;微观粒子只能处于某些确定的能量状态，能量的改变量不能取任意的、连续变化的数值，只能是分立的，即量子化的。
    7. 不确定原理对宏观物体无实际意义，在不确定度关系式中，Planck 常数 $h$ 可当作0;微观粒子遵循不确定度关系， $h$ 不能被看作0。所以，可以用不确定度关系式作为区分宏观物体与微观粒子的判别标准。''',unsafe_allow_html=True)
    st.markdown(r'''
    ## 1.2 量子力学的基本假设
    ### 1 . 2. 1 波函数和微观粒子的状态
    **假设一**  对于一个微观体系，它的状态和由该状态所决定的各种物理性质可用波函数$\Psi (x,y,z,t)$表示。
    >波函数的名称源于这一函数采用了经典物理学中波动的数学形式，该形式可由光波推演而得,故微观粒子的波函数可由平面单色光的波动公式推出。
    将波粒二象性关系$ E = h\nu , p=\frac{h}{ \lambda} $ 代入平面单色光的波动方程$ \Psi = Aexp[i2\pi (\frac{x}{\lambda}-\nu t] $，得单粒子一维运动的波函数：
    $$
    \Psi = Aexp[\frac{i2\pi}{h}(xp_x-E t]
    $$
    若去除时间项则上式被称为定态波函数，用$ \psi $表示,后面讨论的波函数基本都为定态波函数  
    - $\phi$—般是复数形式$\phi = f+ig$，
    $f, g$都是坐标的实函数，$i$为虚数单位，
    $\phi$的共轭复数为$\phi^* = f-ig$，
    即把$\phi$中所有含$i$的地方用$-i$替换即可。
    由于波函数$\phi$描述的波是概率波，因而它必须满足下列**3个条件：**
    >1. 波函数必须是单值的，即在空间每一点$\phi$只能有一个值。
    2. 波函数必须是连续的，即$\phi$的值不出现突跃;$\phi$对x, y, z的一阶微商也是连续函数。
    5. 波函数必须是平方可积的，即$\phi$在整个空间的积分
    $ \int \phi^*\phi\,d\tau$为一个有限数。
    通常要求波函数归一化，即
    $$
    \int \phi^* \phi\,d\tau = 1
    $$
    符合这 3 个条件的波函数称为**合格波函数或品优波函数**。
    ### 1.2.2 物理量和算符
    **假设二**  对一个微观体系的每个可观测的物理量，都对应着一个线性自轭算符。
    >- **算符：** 即对某一函数进行运算操作，规定运算操作性质的符号，例如$+, -, \frac{d}{dx}, lg$等等.
    >- 设物理量A相应的算符为$\widehat{A}$, 若满足下一条件：
    $$
    \widehat{A}(\psi_1 + \psi_2) = \widehat{A}\psi_1 + \widehat{A}\psi_2
    $$
    则称$\widehat{A}$为线性算符,，若$\widehat{A}$能满足：
    $$
    \int\psi_1^*\widehat{A}\psi_1\,d\tau = \int\psi_1(\widehat{A}\psi_1)^*\,d\tau
    $$
    or
    $$
    \int\psi_1^*\widehat{A}\psi_2\,d\tau = \int\psi_2(\widehat{A}\psi_1)^*\,d\tau
    $$
    则称$\widehat{A}$为自轭算符或厄米算符。
    量子力学中若干物理量和对应的算符如下表
    | 物理量 || 算符 |
    | ---: | :--- |:---:|
    | 位置|$x$ | $\widehat{x}=x$ |
    | 动量的$x$轴分量|$p_x$ | $ \widehat{p_x} = -\frac{ih}{2\pi}\frac{\partial}{\partial x} $|
    | 角动量的$z$轴分量|$ M_z = xp_y - yp_x $ | $\widehat{M_z} = -\frac{ih}{2\pi}(x\frac{\partial}{\partial y}-y\frac{\partial}{\partial x})$ |
    | 动能 | $T = p^2/2m$ | $$ \widehat{T} = - \frac{h^2}{8\pi^2m}(\frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2}+\frac{\partial^2}{\partial z^2}=- \frac{h^2}{8\pi^2m}\nabla^2$$ |
    | 势能 | $ V $ | $\widehat{V} = V$ |
    | 总能 | $E=T+V$ | $\widehat{H} = - \frac{h^2}{8\pi^2m}\nabla^2 + V $ |''')
    st.markdown(r'''
    ### 1.2.3 本征态、本征值和 $Schr\ddot{o}dinger$ 方程
    **假设三** 若某一物理量$A$的算符$\widehat{A}$作用于某一状态函数$\psi$等于某一常数$a$乘以$\psi$，即
    $$
    \widehat{A} \psi = a \psi
    $$
    那么对$\psi$所描述的微观状态，物理量$A$具有确定的数值$a$。$a$称为物理量算符$\widehat{A}$的本征值，$\psi$称为$\widehat{A}$的本征态或本征波函数，$
    \widehat{A} \psi = a \psi
    $称为 A 的本征方程。
    >- **自轭算符的第一项重要性质** 
    <br>
    自轭算符的本征值一定为实数，这和本征值的物理意义是相适应的
    <br>
    >- **自轭算符的第二项重要性质**
    <br>
    对一个微观体系，自轭算符$A$给出的本征函数$\psi_1$，$\psi_2$ ,$\psi_3$，…形成一个正交性和归一性的函数组。
    
    (1)归一性是指粒子在整个空间出现的概率为 1，即
    $$
    \int \psi^* \psi\, d\tau = 0
    $$
    (2)正交性是指
    $$
    \int \psi_i^* \psi_j\, d\tau = 0, (i \neq j)
    $$
    ### 1.2.4 态叠加原理
    **假设四** 若$\psi_1, \psi_2, …, \psi_n $为某一微观体系的可能状态，则由它们线性组合所得的$\psi$也是该体系可能存在的状态。
    $$
    \psi = c_1\psi_1 + c_2\psi_2 + … +c_n\psi_n = \sum_i c_i\psi_i
    $$
    式中$c_1, c_2, …, c_n$为任意常数，称为线性组合系数。
    > - 本征态的物理量的平均值
    <br>
    设与$\psi_1, \psi_2, …, \psi_n $对应的本征值分别为$a_1, a_2, …, a_n$，当体系处于状态$\psi$并且$\psi$已经归一化时，物理量$A$的平均值
    $$
    <a> = \int \psi^* \widehat{A} \psi \, d\tau = \int (\sum_i c_i^*\psi_i^*) \widehat{A} (\sum_i c^*\psi^*) \, d\tau = \sum_i |c_i|^2 a_i
    $$
    > - 非本征态的物理量的平均值
    <br>
    若状态函数$\psi$不是物理量$A$的算符$\widehat{A}$的本征态，当体系处于该状态时，可用积分计算其平均值：
    $$
    <a> = \int \psi^* \widehat{A} \psi \, d\tau
    $$
    ### 1.2.5 Pauli（泡利）原理
    **假设五** 在同一原子轨道或分子轨道上，最多只能容纳两个电子，这两个电子的自旋状态必须相反。或者说，两个自旋相同的电子不能占据同一轨道。
    ## 1.3 箱中粒子的$Schr\ddot{o}dinger$方程及其解
    ### 1.3.1 箱中粒子
    一维势箱中粒子是指一个质量为$m$、在一维方向上运动的粒子，它受到如图所示的势能的限制，图中横坐标表示粒子的运动范围，纵坐标为势能。
    ''')
    st.image(
    'GROUP2/image/2.png'
    )
    st.write(r'''
    $$
    V =
    \begin{cases}
    0, 0<x<l \\
    \infty, x \leq 0 or x\geq 0 \\
    \end{cases}
    $$
    故粒子只能在$0~l$的范围内自由运动，其$ Schr\ddot{o}dinger $方程为：
    $$
    -\frac{h^2}{8\pi^2m} \frac{d^2 \psi}{dx^2} = E\psi
    $$
    解出该方程的通解为：
    $$
    \psi (l) = c_1 \cos(\frac{8\pi^2mE}{h^2})^{1/2} x + c_2 \sin(\frac{8\pi^2mE}{h^2})^{1/2}x 
    $$
    在$x=0和x=l$处，由于势能无穷大哦，所以粒子运动到该处的几率为0，即：
    $$
    \begin{cases}
    \psi(0)=c_1\cos(0)+c_2\sin(0)=0 \Rightarrow c_1=0 \\
    \psi(l)=c_2\sin(\frac{8\pi^2mE}{h^2})^{1/2}l=0 \Rightarrow (\frac{8\pi^2mE}{h^2})^{1/2}l = n\pi\\
    \end{cases}
    $$
    因为$n\neq =0 $，所以可得
    $$
    E = \frac{n^2h^2}{8ml^2}
    $$
    将$E$的表达式回代入通解得
    $$ \psi(x) = c_2 \sin(n\pi x/l)$$
    $\psi (x)$满足归一化条件，解得$c_2 = (l/2)^{1/2}$
    所以势箱中的波函数为：
    $$
    \psi_n(x) = (2/l)^{1/2} \sin(n\pi x/l)
    $$
    能级为
    $$
    E_n = \frac{n^2 h^2}{8nl^2}  
    (n = 1, 2, 3…)
    $$ 
    从上可以得出**一维势箱中粒子的特性：**  
    > - 粒子可以存在多种运动状态，它们可由$\psi_1, \psi_2, …, \psi_n $等描述
    > - 能量量子化
    > - 存在零点能
    > - 没有经典运动轨道，只有概率分布
    > - 存在节点，节点多，能量高
    ''')
    
    
    
    
    st.code('''n_list = [2**i for i in [0, 1, 2, 3, 4, 5, 6, 7]]
    x = np.linspace(0,1,400)
    fig = plt.figure(figsize=(6,6))
    plt.axis('off')
    st.write(r'$$\large\textbf{Particle\ in\ Box}$$', unsafe_allow_html=True)
    l = 1
    for i in range(1, 3):
        for j in range(1, 5):
            ax = fig.add_subplot(2, 4, (i-1)*4+j)
            n = n_list[(i-1)*4+(j-1)]
            ax.set_title('n='+str(n), pad = 1.0)
            ax.set_ylim((-2, 2))
            ax.plot(x, (2/l)**0.5*np.sin(n*np.pi*x/l), 'y')
    plt.subplots_adjust(top=0.8)
    plt.tight_layout() 
    st.pyplot(fig, use_container_width=True)''')
    n_list = [2**i for i in [0, 1, 2, 3, 4, 5, 6, 7]]
    x = np.linspace(0,1,400)
    fig = plt.figure(figsize=(6,6))
    plt.axis('off')
    st.write(r'''
             $$
             \large\textbf{Particle\ in\ Box}
             $$''', unsafe_allow_html=True)
    l = 1
    for i in range(1, 3):
        for j in range(1, 5):
            ax = fig.add_subplot(2, 4, (i-1)*4+j)
            n = n_list[(i-1)*4+(j-1)]
            ax.set_title('n='+str(n), pad = 1.0)
            ax.set_ylim((-2, 2))
            ax.plot(x, (2/l)**0.5*np.sin(n*np.pi*x/l), 'y')
    plt.subplots_adjust(top=0.8)
    plt.tight_layout() 
    st.pyplot(fig, use_container_width=True)
    
    
    
    
    
    st.code(
    '''
    n_list = [2**i for i in [0, 1, 2, 3, 4, 5, 6, 7]]
    x = np.linspace(0,1,400)
    fig = plt.figure(figsize=(6,6))
    plt.axis('off')
    st.write(r'$$\large\textbf{Particle\ in\ Box}$$', unsafe_allow_html=True)
    l = 1
    for i in range(1, 3):
        for j in range(1, 5):
            ax = fig.add_subplot(2, 4, (i-1)*4+j)
            n = n_list[(i-1)*4+(j-1)]
            ax.set_title('n='+str(n), pad = 1.0)
            ax.set_ylim((-2, 2))
            ax.plot(x, (2/l)**0.5*np.sin(n*np.pi*x/l)**2, 'y')
    plt.subplots_adjust(top=0.8)
    plt.tight_layout() 
    st.pyplot(fig, use_container_width=True)
    ''')
    n_list = [2**i for i in [0, 1, 2, 3, 4, 5, 6, 7]]
    x = np.linspace(0,1,400)
    fig = plt.figure(figsize=(6,6))
    plt.axis('off')
    st.write(r'''
             $$
             \large\textbf{Particle\ in\ Box}
             $$''', unsafe_allow_html=True)
    l = 1
    for i in range(1, 3):
        for j in range(1, 5):
            ax = fig.add_subplot(2, 4, (i-1)*4+j)
            n = n_list[(i-1)*4+(j-1)]
            ax.set_title('n='+str(n), pad = 1.0)
            ax.set_ylim((-2, 2))
            ax.plot(x, (2/l)**0.5*np.sin(n*np.pi*x/l)**2, 'y')
    plt.subplots_adjust(top=0.8)
    plt.tight_layout() 
    st.pyplot(fig, use_container_width=True)

    st.markdown(r'''
    **用量子力学处理微观体系的一般步骤：**  
    >1. 根据体系的物理条件，写出它的势能函数，进一步写出台算符及$ Schr\ddot{o}dinger $方程。
    >3. 解$ Schr\ddot{o}dinger $方程，根据合格条件求得$\psi$和$E$。
    >3. 描绘$\psi,\psi^2$等的图形，讨论它们的分布特点。
    >2. 由所得的$\psi$，求各个对应状态的各种物理量的数值，了解体系的性质。
    >2. 联系实际问题，对所得结果加以应用。
    ''')
    if 'show_code' not in st.session_state:
        st.session_state.show_code = False
        # 在侧边栏创建一个复选框来控制代码的显示
    st.session_state.show_code = st.sidebar.checkbox('显示代码', value=st.session_state.show_code)
    if st.session_state.show_code:
        with open(__file__, 'r') as file:
            code = file.read()
        st.markdown("```python\n" + code + "\n```", unsafe_allow_html=True)