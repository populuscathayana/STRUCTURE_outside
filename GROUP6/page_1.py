'''
Author: cathayana populuscathayana@gmail.com
Date: 2024-01-25 13:16:21
LastEditors: cathayana populuscathayana@gmail.com
LastEditTime: 2024-01-27 21:33:12
FilePath: /website_structure/STRUCTURE/GROUP6/page_1.py
Description: 

Copyright (c) 2024 by Cathayana, All Rights Reserved. 
'''
import importlib
import streamlit as st

def show():
    st.write('''# 第二章：化学键理论''')
    st.markdown('内容：\n- 化学键概述\n- 分子轨道理论对氢分子离子的处理\n- 分子轨道理论与双原子分子的结构\n- 价键理论\n- 价电子对互斥理论（VSEPR）\n- 杂化轨道理论')

    st.markdown("""
    # 化学键概述
    ## 化学键的概念
    分子是保持化合物特性的最小微粒，也是参与化学反应的最基本单元之一。分子是由两个或两个以上的原子组成的，在相邻的原子之间存在着某种强烈的相互作用力，一般将这种原子间的强相互作用称为化学键。
    ## 化学键的种类
    通常所说的化学键主要指共价键、离子键和金属键，此外的其他各种化学键统称为次级键，它们使分子进一步结合成超分子、分子组合体等。
    ## 共价键、离子键和金属键的对比
    |性质|共价键|离子键|金属键|
    |:---:|:---:|:---:|:---:|
    |A和B的电负性|A电负性B电负性|A电正性B电负性|A电正性B电正性|
    |结合力性质|核间电荷密度增加|离子间静电吸引|自由电子与金属正离子间吸引|
    |结合的几何形式|由轨道叠加和价电子数控制|正负离子的半径比与极化性质|金属原子密堆积|
    |键强度性质|由净成键电子数和成键轨道类型决定|由离子的电价和半径大小决定|6个价电子最高，大于或小于6个都逐渐减小|
    |电学性质|固态和熔态均为绝缘体和半导体|固态为绝缘体，熔态为导体|导体|
    ## 对化学键的理解
    量子力学诞生之前, 化学键被视为一种特殊的化学力。量子力学进一步揭示了化学键（尤其是共价键）的本质。对化学键的理解主要来自MO理论和VB理论。
    # 分子轨道理论对氢分子离子的处理
    ## 分子轨道（MO）理论
    R.S.Mulliken由于建立和发展MO理论荣获1966年诺贝尔化学奖。分子轨道（MO）是原子轨道在分子体系中的推广。
    ## 分子轨道理论处理氢分子离子
    $H_2^+$的能量如下式：""")

    st.latex(r'E_I = \frac{H_{11}+H_{12}}{1+S_{12}}\quad E_{II}= \frac{H_{11}-H_{12}}{1-S_{12}}')

    st.write(r"""
    $E_I$是基态能量，对应基态波函数。$E_{II}$是激发态能量，对应激发态波函数。

    $E_I$和$E_{II}$均是核间距R的函数，

    实验测得R = 106 pm时，EI有最小值-269.0 kJ/mol。


    电子云密度与共价键本质：

    在$H_2^+$的基态，核间的电荷密度增大，电子同时受到两个核的吸引，由于这种吸引使得体系能量降低，使具有一定稳定性的共价键形成。在分子轨道理论中称为成键轨道。

    在$H_2^+$的激发态，核间电荷密度变小，体系能量升高，所以称为反键轨道。
    # 分子轨道理论与双原子分子的结构
    ## 分子轨道理论基本要点
    ### 1 单电子近似的基本思想——单电子波函数
    多核多电子体系的薛定鄂方程无法精确求解。

    近似忽略分子中电子间的瞬时相互作用，将分子中的每一个电子的运动看作是在所有核和其余（n-1）个电子所形成的平均势场中运动，整个分子的运动状态可近似地用单电子波函数的乘积来表示。
    ### 2 分子轨道是原子轨道的线性组合（LCAO）
    从原子轨道过渡到分子轨道，这两者之间存在着一定的联系。电子也保留了原来原子轨道的某些成份。考虑这两点，把分子轨道近似地描述为原子轨道的线性组合。
    ### 3 有效组成分子轨道的条件（成键三原则）
    #### （1）能量相近条件
    原子轨道要有效组成分子轨道，这些原子轨道的能量要相近。
    #### （2）最大重叠条件
    两原子轨道要有效组成分子轨道，必须尽可能重迭，这就是轨道最大的重迭条件。

    两原子轨道要能最大重迭，那么两原子的核间距要小，并且两原子必须按一定的方向接近。

    这一点也说明了共价键为什么具有方向性。
    #### （3）对称性匹配条件
    对称性匹配条件是指参加组成分子轨道的各原子轨道的对称性必须一致。

    在有效组成分子轨道的三个条件中，首先应考虑对称性匹配条件，如果原子轨道的对称性不一致，则不可能组成分子轨道。对称性匹配原则是形成分子轨道的前提，其余两条原则只是组合效率的问题。
    ### 4 分子轨道的类型、符号和能级顺序
    #### （1）类型和符号
    一般分子轨道可以分为σ、π、ō 三种类型。

    σ：对键轴具有圆柱形对称的分子轨道为σ轨道。

    π：任何分子轨道，如果具有一个含键轴的节面并相对于这一节面是反对称的轨道称为π轨道。

    ō：以通过键轴有两个节面的分子轨道称为ō型分子轨道。
    #### （2）轨道的能级顺序
    O、F、Cl、Br、I、He形成的分子或离子的轨道能级顺序：

    $σ_{1s}，σ_{1s}^{*}，σ_{2s}，σ_{2s}^*，σ_{2pz}，π_{2py}=π_{2px}，π_{2py}^*=π_{2px}^*，σ_{2pz}^* ……$

    或写作：$1σ_g，1σ_u， 2σ_g， 2σ_u，3σ_g，1π_u，1π_g，3σ_u……$

    异核：1σ，2σ，3σ，4σ，5σ，1π，2π，6σ，……

    Li，Be，B，C，N形成的分子或离子的轨道能级顺序：

    $σ_{1s}，σ_{1s}^*，σ_{2s}，σ_{2s}^*，$<font color=red>$π_{2py}=π_{2px}，σ_{2pz}$</font>$，π_{2py}^{*}=π_{2px}^*，σ_{2pz}^* ……$

    或写作：$1σ_g，1σ_u， 2σ_g， 2σ_u，$<font color=red>$1π_u，3σ_g$</font>$，1π_g，3σ_u……$

    异核：1σ，2σ，3σ，4σ，<font color=red>1π，5σ</font>，2π，6σ，……
    ### 5、电子填充原则
    遵循：能量最低原理，保理原理和洪特规则。
    ## 双原子分子的电子组态与性质
    分子的电子组态：电子在分子轨道上的排布。

    如F2分子的电子组态：

    $(σ_{1s})^2(σ_{1s}^{*})^2(σ_{2s})^2(σ_{2s}^*)^2(σ_{2pz})^2(π_{2py})^2(π_{2px})^2(π_{2py}^*)^2(π_{2px}^*)^2$

    内层电子不参与成键，可用KK代表：

    KK$(σ_{2s})^2(σ_{2s}^*)^2(σ_{2pz})^2(π_{2py})^2(π_{2px})^2(π_{2py}^*)^2(π_{2px}^*)^2$

    双原子分子的键级：""",unsafe_allow_html=True)

    st.latex(r'键级=\frac{成键电子总数-反键电子总数}{2}')

    st.markdown("""
                键级越大，化学键越强。
    #### 等电子原理
    如果两个分子的电子数目相等（这两个分子称为等电子分子），那它们的分子电子组态往往相似。
    若轨道保持原来原子轨道的能级，称为非键轨道。如HF分子中的1σ，2σ，1π轨道。
    # 价键理论
    VB理论是近似求解分子Schrödinger方程的另一种方法。1927年, W.H.Heitler和F.W.London首次用它处理H2分子中的电子对键, 奠定了近代价键理论的基础。
    VB法将分子中的共价键视为电子配对形成的定域键, 所以也称为电子配对法。其处理结果与的MO处理相似。
    ## 电子配对法要点
    两个原子的价轨道上各有一个电子且自旋相反，可以配对形成电子对键;

    原子价层轨道中未成对电子数即为原子价, 两个电子配对后就不再与第三个电子配对，所以共价键有饱和性;原子轨道沿重叠最大的方向成键, 所以，共价键有方向性。

    ## VB理论的局限性
    无法解释有些分子的键角明显偏离原子轨道之间的夹角, 有些原子的成键数目大于价层轨道中未成对电子数。

    为解决这些矛盾，Pauling提出了杂化轨道（HAO）的概念：

    为在化合过程中更有效地形成化学键，同一原子中几个轨道经过线性组合形成新的轨道杂化轨道。杂化轨道仍是原子轨道, 但在特定方向上的值增大了，超过了纯粹的原子轨道，在此方向上成键更为有效。

    ## 杂化遵循的规则
    (1) 参与杂化的原子轨道能量相近；

    (2) 轨道数守恒，即杂化前后轨道数目不变，只有轨道的空间分布发生改变；

    (3) 形成的HAO也应正交归一化；

    (4) HAO一般均与其他原子形成σ键或容纳孤对电子，而不以空轨道形式存在。

    ## MO理论和VB理论的关系和区别
    ### 1 选择变分函数的思想不同
    MO：以MO为基函数， VB：以AO为基函数
    ### 2 变分结果不同
    基态波函数：MO过分强调了离子项，VB过分忽略了离子项。

    电荷密度：MO过分强调电子在键区的集中，使电子的排斥过度。
    ### 3 校正
    在VB波函数中增加一定比例的离子项，在MO波函数中混入一定比例的激发态，二者结果就趋于一致。
    # 价电子对互斥理论（VSEPR)
    价 层 电 子 对 互 斥 规 则 （ VSEPR） 起源于 1 9 4 0 年Sidgwick和Powell的工作。Gillespie和Nyholm作了进一步推广和普及。

    ## 一、基本思想
    在ABn型分子或基团中（几个B可以是不同的原子），若中心原子A的价层不含d电子或d电子分布为球对称的d5或d10，则几何构型主要由价电子对——σ电子对和孤电子对——的数目决定。

    价电子对之间应尽可能相互远离，使体系的能量最低，这就是价电子对互斥理论的基本思想。所以，2、3、4、5、6个价电子对的空间分布分别为直线形、平面三角形、正四面体形、三角双锥形和正八面体形。
    ## 二、基本规律
    1、孤对电子与孤对电子＞孤对电子与成键电子＞成键电子与成键电子。

    2、多重键亦作为单键看，总体斥力: 叁键＞双键＞单键（重键含电子多）。

    3、价电子对之间的夹角大于90°，斥力为零。
    ## 三、应用价电子对互斥理论，确定分子的几何构型
    （1）对分子离子，正、负电荷均归到中心原子A上；

    （2）根据中心原子A的价电子数和成键情况，确定中心原子的成键电子对BP数目及孤电子对LP数目的总和；

    （3）有重键时，重键被看做“超级电子对”，它只用去更多电子但不产生新的排布方向；

    （4）对于4BP+1LP、3BP+2LP、2BP+3LP，LP均处于赤道位置而不在极轴位置；

    （5）为分子几何构型命名时，只看BP的排布方式，不再包括LP；

    （6）由于LP处于中心原子A上, 两个LP之间产生的斥力较大（LP-LP > LP-BP > BP-BP），所以，有时可能需要对分子几何构型作一些轻微调整。单个孤电子也按LP对待，但产生的斥力较小。
    # 杂化轨道理论
    ## 一、由来
    价键理论无法解释CH4的四面体结构，为解决这些矛盾，Pauling提出了杂化轨道（HAO）的概念。
    ##  二、杂化轨道理论
    ### 1、基本设想
    所谓“杂化”就是单中心原子轨道的线性组合，即在形成分子过程中，原子中能级相近的几个原子轨道可以相互混合，从而产生新的原子轨道。

    在杂化过程中轨道数目不变。即有n个参加杂化的原子轨道，可以组合成n个新的杂化轨道。
    ### 2、某些能量不同的原子轨道为什么可以杂化？
    按照量子力学中态的叠加原理，简并状态的任何线性组合，也一定是允许的状态。（满足合格条件的波函数线性组合后也一定是薛定谔方程的解）。

    因此三个p轨道的线性组合，得到的新的原子轨道仍然是p轨道，只是方向发生变化。

    对于能量不同的轨道，只要这些轨道的能量相差不大，它们之间可以杂化。
    ### 3、杂化轨道必须是正交归一性的。
    ### 4、单位轨道贡献
    对应于每一个参加杂化的原子轨道，在所有新的杂化轨道中该轨道的成份之和必须为一个单位。（成份就是原子轨道系数的平方）


    $\sum_{k=1}^n$ $C_{ ki}^2 $= $C_{ 1i}^2 $+$C_{ 2i}^2 $+……+$C_{ ni}^2 $=1


    若杂化中，$C_{ 1i}^2 = C_{ 2i}^2 = …… C_{ni}^2 ＝1/n$，则称为等性杂化。

    若杂化中，$C_{ 1i}^2 = C_{ 2i}^2 = …… C_{ni}^2 ≠1/n$，则称为不等性杂化。

    ### 5、原子轨道为什么需要杂化？
    原子轨道杂化以后可使成键能力增加因而使生成的分子更加稳定。
    杂化是量子力学处理分子结构的一种数学手段，是讨论化学键的一种理论方法。对于杂化应理解为在原子相互结合形成分子的过程中，原子中的价电子的运动状态发生变化，使电子云聚集和延伸在某个方向，以便与其它原子形成稳定的化学键，使体系的能量降低。
    ## 三、s-p杂化轨道及有关分子结构
    | 杂化类型 | 杂化轨道 | 形状 | 点群 | 例子 |
    |  :---: |  :---: | :---: | :---: | :---: |
    |sp|$s,px$|直线|$D_{∞h}$|$CO_2,N^{3-}$|
    |$sp^2$|$s, p_x,p_y$|三角形|$D_{ 3h}$|$BF_3, SO_3$|
    |$sp^3$|$s,p,p_y,p_z$|四面体|$T_d$|$CH_4$|
    |$dsp^2$|$d_{x²-y²},s,p_x,p_y$|平面四方|$D_{4h}$|$Ni(CN)_4^{2-}$|
    |$dsp^3$|$d_{z²},s,p_x,p_y,p_z$|三角双锥|$D_{3h}$|$PF_5$|
    |$dsp^3$|$d_{x²-y²},s,p_x,p_y,p_z$|四方锥|$C_{4v}$|$IF_5$|
    |$d^2sp^3$|$d_{z²},d_{x²-y²},s,p_x,p_y,p_z$|八面体|$O_h$|$SF_6$|
    """,unsafe_allow_html=True)
    if 'show_code' not in st.session_state:
        st.session_state.show_code = False
        # 在侧边栏创建一个复选框来控制代码的显示
    st.session_state.show_code = st.sidebar.checkbox('显示代码', value=st.session_state.show_code)
    if st.session_state.show_code:
        with open(__file__, 'r') as file:
            code = file.read()
        st.markdown("```python\n" + code + "\n```", unsafe_allow_html=True)