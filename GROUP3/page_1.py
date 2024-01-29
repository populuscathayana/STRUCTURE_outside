'''
Author: cathayana populuscathayana@gmail.com
Date: 2024-01-25 13:16:21
LastEditors: cathayana populuscathayana@gmail.com
LastEditTime: 2024-01-27 21:32:40
FilePath: /website_structure/STRUCTURE/GROUP3/page_1.py
Description: 

Copyright (c) 2024 by Cathayana, All Rights Reserved. 
'''
import importlib
import streamlit as st

def show():
    st.title('配位化学')
    st.write(r"""
            ## 概述
            ### 定义
            配合物一词很难确切定义，在本章讨论的配合物是指由中央的金属原子（或离子）与周围若干的分子或离子所组成的化合物；
            - 中央的金属原子（或离子）称为**中心离子（原子）**；可有单核、多核（杂多核）、金属原子簇（M-M）
            - 中心离子周围的分子基团或离子称为**配位体**；可有单齿(NH<sub>3</sub>)、非鳌合多齿(CO<sub>3</sub><sup>2-</sup>)、鳌合([Co(EDTA)]<sup>-</sup>)、( [Cu(En)<sub>2</sub>]<sup>2+</sup>)、π键配位体(烯烃)
            - 中心离子周围配位点的数目称为**配位数**，一般配位数在2-12之间，例如配位数为2的[Hg(CN)<sub>2</sub>]，配位数为9的[Nd(H<sub>2</sub>O)<sub>9</sub>]<sup>3+</sup>。
            - 对于π键配位体按其提供的电子数目计算。
            """,unsafe_allow_html=True)
    st.image('GROUP3/image/1.png',width=600)
    st.write(r"""
            配合物中心离子（原子）和配位体之间的键称为**配位键**。
            ## 配合物中的化学键理论
            ### 价键理论
            L.Pauling：按杂化轨道理论用共价配键与电价配键来解释配合物中心离子与配位体之间的配位键。
            #### 共价配键与共价配合物
            共价配键就是由配位体L提供孤对电子与中心离子M的价电子层空轨道形成σ配键；过渡元素中心离子的价电子层空轨道不是单纯的原子轨道，而是由能量相近的未充满的d，s，p组成的杂化轨道，组成杂化轨道的原因在于提高成键能力。
            共价配合物中，中心离子的未成对电子都减少了，也就是这些配合物的顺磁性要比未络合的自由离子减弱，所以共价配合物是低自旋配合物。
            #### 电价配键与电价配合物
            电价配键就是带正电的中心离子与带负电或有偶极矩的配位体靠静电吸引力结合在一起，称为电价配键，相应的配合物称为电价配合物。
            中心离子或原子与配位体电负性相差较大时，容易形成电价配键，卤素离子的电负性较大，H<sub>2</sub>O分子的偶极矩较大（氧带负电），这两种离子或分子作为配位体时，一般形成电价配键，例如[FeF<sub>6</sub>]<sup>3-</sup>，[Fe(H<sub>2</sub>O)<sub>6</sub>]<sup>3+</sup>，[CoF<sub>6</sub>]<sup>3-</sup>都是电价络离子。
            按这样的分类仍不能解释d电子数小于3大于8的配合物，例如$d^9$离子配合物$[Cu(NH_3)_4]^{2+}$，配合物的磁性与自由离子一样；

            提出内轨配合物与外轨配合物(杂化）

            - (n-1)d，ns，np 内轨配合物（共价配合物）

            - ns，np，nd 外轨配合物（电价配合物）

            外轨配合物没有影响中心离子的电子层结构，采取高自旋形式。
            ### 晶体场理论（CFT）（Crystal Field Theory）
            #### 基本要点
            - 配合物中心离子与视为点电荷或点偶极的配位体之间的相互作用是纯静电作用
            - 当受到配体的静电作用时，中心离子原来五重简并的d轨道要发生分裂
            #### d轨道的能级分裂
            将配位体近似视为点电荷与中心原子/离子发生作用
            - 中心离子价电子层有5个d轨道，它们的空间取向不同；
            - 从配位体角度看，不同配位数对中心离子产生的静电作用不同，即不同配位数组成的配位体静电场具有不同的对称性；
            - 中心离子的5个d轨道在不同对称性的配位体静体场作用下发生变化。
            ##### 八面体场(Oh)
            （高无结上有关于f轨道的算算？）
            """,unsafe_allow_html=True)
    st.code('''import numpy as np
import scipy.special as ss
from itertools import product
import matplotlib.pyplot as plt
from numpy.random import rand

def hydrogen(n,l,m,r,th,phi):
    na = 0.53*n
    tmp = (2/na)**3
    tmp *= ss.factorial(n-l-1)
    tmp /= 2*n*ss.factorial(n+l)**3
    tmp2 = np.exp(-r/na) * (2*r/na)**l
    laguerre = ss.assoc_laguerre(2*r/na, n-l-1, 2*l+1)
    R_nl = np.sqrt(tmp)*tmp2*laguerre
    return R_nl*ss.sph_harm(m,l,th,phi)

def pts2xyz(pts,xd,yd,zd):
    r, th, phi = zip(*pts)
    x = r*np.sin(phi)*np.cos(th)+xd
    y = r*np.sin(phi)*np.sin(th)+yd
    z = r*np.cos(phi)+zd
    return x,y,z

def getPts(n, l, m, N, rMax, randMax):
    r = rand(N)*rMax
    th = rand(N)*np.pi*2
    phi = rand(N)*np.pi
    pts = []
    for axis in zip(r,th,phi):
        p = hydrogen(n, l, m,*axis)**2
        if p*axis[0]**2 > np.random.rand()*randMax:
            pts.append(axis)
    #print(len(pts))
    return pts

xyzd = [[8,0,0],
        [0,8,0],
        [0,0,8],
        [-8,0,0],
        [0,-8,0],
        [0,0,-8],
        ]

pts = getPts(3, 2, 0, 200000, 5, 1/20000)
x,y,z = pts2xyz(pts,0,0,0)
ax = plt.subplot(projection='3d')
ax.scatter(x, y, z, marker='.')
pts = getPts(1, 0, 0, 4000, 5, 1/150)
for a in xyzd:
    x,y,z = pts2xyz(pts,a[0],a[1],a[2])
    ax = plt.subplot(projection='3d')
    ax.scatter(x, y, z, marker='.')
ax.view_init(45, 45) #第一个是平着转的角度，第二个是竖着转的角度
plt.show()

''')
    import numpy as np
    import scipy.special as ss
    from itertools import product
    import matplotlib.pyplot as plt
    from numpy.random import rand

    def hydrogen(n,l,m,r,th,phi):
        na = 0.53*n
        tmp = (2/na)**3
        tmp *= ss.factorial(n-l-1)
        tmp /= 2*n*ss.factorial(n+l)**3
        tmp2 = np.exp(-r/na) * (2*r/na)**l
        laguerre = ss.assoc_laguerre(2*r/na, n-l-1, 2*l+1)
        R_nl = np.sqrt(tmp)*tmp2*laguerre
        return R_nl*ss.sph_harm(m,l,th,phi)

    def pts2xyz(pts,xd,yd,zd):
        r, th, phi = zip(*pts)
        x = r*np.sin(phi)*np.cos(th)+xd
        y = r*np.sin(phi)*np.sin(th)+yd
        z = r*np.cos(phi)+zd
        return x,y,z

    def getPts(n, l, m, N, rMax, randMax):
        r = rand(N)*rMax
        th = rand(N)*np.pi*2
        phi = rand(N)*np.pi
        pts = []
        for axis in zip(r,th,phi):
            p = hydrogen(n, l, m,*axis)**2
            if p*axis[0]**2 > np.random.rand()*randMax:
                pts.append(axis)
        #print(len(pts))
        return pts

    xyzd = [[8,0,0],
            [0,8,0],
            [0,0,8],
            [-8,0,0],
            [0,-8,0],
            [0,0,-8],
            ]

    pts = getPts(3, 2, 0, 200000, 5, 1/20000)
    x,y,z = pts2xyz(pts,0,0,0)
    ax = plt.subplot(projection='3d')
    ax.scatter(x, y, z, marker='.')
    pts = getPts(1, 0, 0, 4000, 5, 1/150)
    for a in xyzd:
        x,y,z = pts2xyz(pts,a[0],a[1],a[2])
        ax = plt.subplot(projection='3d')
        ax.scatter(x, y, z, marker='.')
    ax.view_init(45, 45) #第一个是平着转的角度，第二个是竖着转的角度
    st.pyplot(plt)
    #plt.show()
    st.write(r"""
            在一个八面体配合物中,$d_{z^2}$和$d_{x^2-y^2}$,这一对轨道结合在一起称为$e_g$轨道,而$d_{xy}$,$d_{yz}$和$d_{xz}$轨道合称$t_{2g}$轨道。这两组轨道间的能级差称为$Δ_o$，下标“o”是表示八面体。为了保持分裂前后d轨道的重心不变,即一组轨道(在这里是$t_{2g}$轨道)能量降低,而由另一组轨道(这里是$e_g$轨道)能量升高来补偿。$t_{2g}$轨道比分裂前低$2/5Δ_o$而$e_g$轨道比分裂前高$3/5Δ_o$
            ##### 四面体场(Td)
            同理也有分裂能，一般四面体场分裂能$Δ_t$为八面体场分裂能$Δ_o$的一半
            ##### 平面正方形
            """,unsafe_allow_html=True)
    st.code('''
            import numpy as np
            import scipy.special as ss
            from itertools import product
            import matplotlib.pyplot as plt
            from numpy.random import rand

            def hydrogen(n,l,m,r,th,phi):
                na = 0.53*n
                tmp = (2/na)**3
                tmp *= ss.factorial(n-l-1)
                tmp /= 2*n*ss.factorial(n+l)**3
                tmp2 = np.exp(-r/na) * (2*r/na)**l
                laguerre = ss.assoc_laguerre(2*r/na, n-l-1, 2*l+1)
                R_nl = np.sqrt(tmp)*tmp2*laguerre
                return R_nl*ss.sph_harm(m,l,th,phi)

            def pts2xyz(pts,xd,yd,zd):
                r, th, phi = zip(*pts)
                x = r*np.sin(phi)*np.cos(th)+xd
                y = r*np.sin(phi)*np.sin(th)+yd
                z = r*np.cos(phi)+zd
                return x,y,z

            def getPts(n, l, m, N, rMax, randMax):
                r = rand(N)*rMax
                th = rand(N)*np.pi*2
                phi = rand(N)*np.pi
                pts = []
                for axis in zip(r,th,phi):
                    p = hydrogen(n, l, m,*axis)**2
                    if p*axis[0]**2 > np.random.rand()*randMax:
                        pts.append(axis)
                print(len(pts))
                return pts

            xyzd = [[8,0,0],
                    [0,8,0],
                    [-8,0,0],
                    [0,-8,0],
                    ]

            pts = getPts(3, 2, 0, 200000, 5, 1/20000)
            x,y,z = pts2xyz(pts,0,0,0)
            ax = plt.subplot(projection='3d')
            ax.scatter(x, y, z, marker='.')
            pts = getPts(1, 0, 0, 4000, 5, 1/150)
            for a in xyzd:
                x,y,z = pts2xyz(pts,a[0],a[1],a[2])
                ax = plt.subplot(projection='3d')
                ax.scatter(x, y, z, marker='.')
            ax.view_init(45, 45) #第一个是平着转的角度，第二个是竖着转的角度
            plt.show()
    ''')
    xyzd = [[8,0,0],
            [0,8,0],
            [-8,0,0],
            [0,-8,0],
            ]
    plt.clf()
    pts = getPts(3, 2, 0, 200000, 5, 1/20000)
    x,y,z = pts2xyz(pts,0,0,0)
    ax = plt.subplot(projection='3d')
    ax.scatter(x, y, z, marker='.')
    pts = getPts(1, 0, 0, 4000, 5, 1/150)
    for a in xyzd:
        x,y,z = pts2xyz(pts,a[0],a[1],a[2])
        ax = plt.subplot(projection='3d')
        ax.scatter(x, y, z, marker='.')
    ax.view_init(45, 45) #第一个是平着转的角度，第二个是竖着转的角度
    st.pyplot(plt)
    st.write('''
            #### 构效关系(以Oh为例)
            ##### 分裂能大小
            在八面体场作用下，中心离子的d轨道发生能级分裂，Δ值的大小可以通过紫外可见光谱得到，产生的吸收光谱一般在紫外可见区域。（称为d-d跃迁），（$10000﹤Δ﹤30000 cm^{-1}，即在1~4 eV）。

            分裂能的大小由中心离子与配位体两个方面的情况决定。
            - 中心离子不变，不同配位体产生的Δ顺序如下（配体光谱化学序列）：
            $I^-﹤Br^-﹤Cl^-﹤-SCN^-﹤F^-≈OH^-≈HCOO^-﹤C_2O_4^{2-}﹤H_2O﹤-NCS^-﹤NH_2CH_2COO^-﹤EDTA﹤吡啶≈NH_3﹤乙二胺﹤SO_3^{2-}﹤邻二氮杂菲﹤ -NO_2^-﹤CN^-(CO)$
            - 配位体不变，Δ随中心离子而变：
            - 同一金属离子，高价离子产生的Δ大于低价离子
            - 同价态同族元素的中心离子，周期数大的离子产生的Δ大于周期数小的离子，第二系列过渡元素比第一系列Δ值增大40-50%，第三系列比第二系列Δ增大20-25%
            ##### d电子排布方式

            借助于电子层构建原理和Hund规则,可以根据八面体场中能级的分裂给出某个八面体配合物基态时的电子组态,对于$d^1-d^3$和$d^8-d^{10}$配合物,基态时只有一种电子组态.它们分别为:

            $t_{2g}^1,t_{2g}^2,t_{2g}^3,t_{2g}^6e_g^2,t_{2g}^6e_g^3,t_{2g}^6e_g^4$

            但是对于$d^4-d^7$的配合物,则有高自旋和低自旋两种电子组态。在高自旋配合物中分裂能$Δ_o$小于成对能,这时电子多占$t_{2g}$和$e_g$轨道,而避免电子成对。对低自旋配合物,分裂能大于成对能,这时电子成对地优先占据$t_{2g}$轨道,直至填满$t_{2g}$轨道后再占据高能级的$e_g$轨道。
            - 强场低自旋/弱场高自旋

            对第四周期过渡元素八面体配合物,究竟是属于高自旋还是属于低自旋主要决定于配位体的性质。另一方面，对第五、六周期过渡元素八面体配合物。由于$Δ_o$较大，成对能较小，低自旋较为有利。对于较大的4d和5d轨道，影响成对能的电子的推斥作用要比较小的3d轨道小。

            对于四面体配合物，由于$Δ_t$通常为$Δ_o$一半，一般$Δ_t<P$，极有利于高自旋组态，实际上，低自旋的四面体配合物极少见。
            - 关于d-d跃迁

            d-d跃迁或晶体场跃迁是指电子从受到推斥干扰的d轨道的能级跃迁到相同类型的能级。换言之,电子原来处在中心金属离子,跃迁后依然处在该离子的激发态上。当配合物具有$Oh$对称性,对这个点群的特征标表的研究说明一个电偶极允许的跃迁必须是“g"态和“u"态间的跃迁,即u——g(Laporte选律)。由于这里讨论到的晶体场的状态都是中心对称$g$态，跃迁是明显地不可能。即所有的d-d跃迁都是对称禁阻的，因而强度很低。实际观察到的d-d跃迁是由于电子运动和分子的振动间的相互作用。
            #### 配位（晶体）场稳定化能 LFSE（CFSE）
            -以球对称场中未分裂d轨道的能量为零，将d电子从未分裂的d轨道进入分裂的d轨道所产生的总能量下降值，称为配位场稳定化能。（不考虑成对能）

            $LFSE = 0-(-0.4m+0.6n)Δ$

            - Δ—分裂能, $m—t_{2g}上电子数, n—e_g上电子数$

            LFSE越大，表明能量下降越多，配合物越稳定，故可用LFSE的大小来衡量配合物的稳定性。

            通过Gaussian计算了$Fe[Cl]_6^{3-}$(弱场配合物，理论LFSE为0)以及$Fe[NH_3]_6^{3+}$(强场配合物，理论LFSE为2.0)的稳定化能，有下列数据可以看出，$Fe[NH_3]_6^{3+}$较$Fe[Cl]_6^{3-}$有更强的稳定性
            |Fe(Ш)   |$Cl^-$   |$NH_3$   |Pro1   |Pro1 LFSE   |Pro2   |Pro2 LFSE   |
            |:--------| :---------:|--------:|:--------| :---------:|--------:| ---------|
            |-1260.14	 |-459.84   |-56.17   |-1598.37   |-1.21   |-4019.07   |-0.49   |


            以下LFSE的大小对配合物性质有着重要影响，以下是几方面：
            ##### 水化热与点阵能
            水为弱场配体，第四周期二价金属离子由Ca到Zn,由于3d电子层受核吸引增大,离子水化热理应循序增加,但实际上由于受LFSE的影响,出现下图的形状。
            ''',unsafe_allow_html=True)
    st.image('GROUP3/image/2.png',width=600)
    st.write('''
    第四周期金属元素的卤化物,从$CaX_2到ZnX_2$,(X=Cl,Br,I),其点阵能随d电子数变化也有相似的双突起的情况。
    ##### 配合物中心离子半径
    若将第四周期金属六配位的二价离子的离子半径对3d电子数作图，得到下图。由于随核电荷增加，d电子数目也增加，但d电子不能将增加的核电荷完全屏蔽。单从这个因素考虑，离子半径应单调下降，实际上由于LFSE的影响，d轨道分裂后的$t_{2g}$与$e_g$轨道，$t_{2g}$轨道能级较低，在此轨道上的电子对配位体的斥力较小，$e_g$轨道能级较高，此轨道上的电子对配位体的斥力较大。对配位体斥力大，中心离子半径大，反之，离子半径小，这是影响离子半径的一个因素。当d电子填入$t_{2g}$轨道时，斥力作用较小，有效核电荷增加使离子半径减小为主，所以从$d^0-d^3$是离子半径减小的过程；电子填入能量较高、配位体斥力较大的$e_g$轨道时，这种斥力使半径增大为主，所以从$d^4-d^5$是离子半径增大过程。同理可解释$d^6-d^8$离子半径下降，$d^9-d^{10}$离子半径增加。故HS型出现向下双蜂，LS型出现向下单峰，这是LFSE的能量效应对微观结构的影响。对八面体配合物，HS态的半径（空心点）比LS态的半径(实心点)大。
    ''',unsafe_allow_html=True)
    st.image('GROUP3/image/3.png',width=600)
    st.write('''
    ##### Jahn-Teller效应
    - 任何具有简并性电子状态，即具有简并轨道的非线性分子将会产生变形除去简并性。而且若原来的体系具有对称中心对称性，在变形后的体系中依然保持对称中心。
    - 即：d电子云分布的不对称性引起配位多面体畸变
    由电子组态可以看出，只有$d^0,d^3,d^5(HS),d^6(LS),d^8和d^{10}$有着非简并性的基态。因此这6种不会出现Jahn-Teller变形。
    在八面体配合物中,由$t_{2g}$轨道发生的Jahn-Teller变形较小，而由奇数电子处在$e_g$轨道发生的变形则较大。即$d^9、d^4(HS)、d^7(LS)$Jahn-Teller效应较强，其它的Jahn-Teller效应较弱。以$d^9(t_{2g}^6e_g^3)$八面体配合物为例说明。这时轨道的简并性出现在e轨道。为了保持体系的对称中心对称性，配合物出现四方变形：处在z轴上的两个配位体或者受压缩，或者受拉长，而处在xy平面上的4个配位体保持它们原来的位置不变，如下图所示。
    ''',unsafe_allow_html=True)
    st.image('GROUP3/image/4.png',width=600)
    st.write('''
    一个八面体配合物发生四方变形时,金属离子d轨道产生的两种晶体场分裂情况归纳在下图中。由此图可清楚看到不论电子组态为$e_g^4b_{2g}^2a_{1g}^2b_{1g}^1或是b_{2g}^2e_g^4b_{1g}^2a_{1g}^1$，在能量上都将比原来的八面体配含物的电子组态$t_{2g}^6e_g^3$有利。正是这种能量上的稳定性导致Jahn-Teller变形。沿z轴拉长的熟识的例子是Cu(Ⅱ)的卤化物$CuX_2$。在这些体系中,每个$Cu^{2+}$离子被6个卤素离子包围，4个较短，2个较长。它们的结构数据示于下。
    ''',unsafe_allow_html=True)
    st.image('GROUP3/image/5.png',width=600)
    st.image('GROUP3/image/6.png',width=600)
    st.write('''
    ### 分子轨道理论
    用分子轨道理论处理配合物，仍用分子轨道理论的基本要点，将中心离子的价轨道与配位体的价轨道按对称性匹配条件进行线性组合，得到了配合物的分子轨道，以八面体配合物为例，简单介绍分子轨道理论在配合物中的应用。
    中心离子的价轨道共有9个原子轨道:
    - σ型 $d_{x^2-y^2}，d_{z^2}，s，p_x，p_y，p_z$，
    - π型 $d_{xy}，d_{xz}，d_{yz}$，
    每个配位体可以有σ型分子轨道和π型分子轨道
    #### 形成σ键的配合物
    中心离子的$d_{x^2-y^2}，d_{z^2}，s，p_x，p_y，p_z$，6个价轨道的极大值方向都是沿着±x，±y，±z三个坐标轴指向配位体，可以与配位体孤对电子的σ型轨道沿键轴重叠，每个配位体有一个σ型孤对电子轨道，6个配位体有6个σ型轨道，中心离子6个原子轨道，则总共有12个轨道参与各种线性组合，所以可得12个σ型离域分子轨道。
    - 6个成键轨道，
    - 6个反键轨道，
    - 成键分子轨道中电子的性质主要具有配位体电子的性质；
    - 反键分子轨道中电子的性质主要是中心离子电子的性质；
    - $e_g^*$轨道主要是中心离子的d轨道，而$t_{2g}$非键轨道本身就是中心离子的d轨道
    - $E_{e_g^*}-E_{t_{2g}}=Δ_o=10Dq$

    前面利用配位场分裂Δ解释配合物的结构与性能，同样可以利用分子轨道理论得到的分裂能Δ进行解释。
    #### 含π键的配合物
    - 配位体有π型轨道，与$t_{2g}$轨道对称性匹配，则它们之间可以形成π键；
    - 配位体提供的π型轨道可是配位原子的P轨道，d轨道，也可是配位基团的$π^*$轨道，例如中心离子的$d_{xy}$轨道可以与配位体的4个P轨道线性组合。

    由$t_{2g}$轨道与配位体π型轨道组合成离域π轨道，因此$t_{2g}$轨道的能级发生变化，而这种能级变化又由配位体π型轨道的能级高低决定：

    如果配位体π型轨道能级高于$t_{2g}$轨道，则组合后使$t_{2g}$轨道能级降低；

    如果配位体π型轨道能级低于$t_{2g}$轨道，则组合后使$t_{2g}$轨道能级增加。

    - 配位体π型轨道是高能级的空轨道（高而空）

    π型轨道能级高于$t_{2g}$轨道，并且π型轨道是空轨道。
    从图中可以看出，由于$t_{2g}$轨道能级下降，使得$e_g^*$与$t_{2g}$间的能量差Δ增大，因此配位体有这样的高能级π空轨道，则产生的Δ大，是强场配位体，生成的是低自旋配合物，例如$CN^-$，$-NO_2^-$等配位体存在能量较高空的$π^*$轨道，是强场配位体。
    - 配位体π型轨道是低能级的占据轨道（低而满） 
    π型轨道能级低于$t_{2g}$，并且π型轨道已被电子占据，由于$t_{2g}$能级升高，使$e_g^*$与$t_{2g}$间的能量差减小，因此配位体有此类型的低能占有π轨道，则产生的Δ较小，是弱场配位体，生成的是高自旋配合物，例如，卤素离子的P轨道，$H_2O$分子中配位原子氧的P轨道，因此它们是弱场配位体。
    ### 配位场理论
    晶体场理论把中央金属和配位体之间的相互作用，看作是不同对称性的配位体静电场对中心离子的作用，这是一种类似于离子晶体中正负离子间的静电作用，这种基于点电荷模型的化学键类似于电价键，没有共价键的性质。

    分子轨道理论主要将所有的电子离域化，突出了共价键的作用，而忽视静电作用，综合考虑这两种理论的结果，一般称为配位场理论。

    ### 特殊形式的配合物
    #### σ-π配键
    配合物$[Co(CO)_4]^-$，中心离子Co是负一价的；
    - 提出中心离子或原子与配体间存在另一种键；
    - 中心离子或原子一方面接受配体给出的电子，另一方面将电子送回到配位体上去，这样就避免中心离子（原子）上负电荷过分聚集，使配合物稳定存在，这种键就是σ—π配键。
    ''',unsafe_allow_html=True)
    st.image('GROUP3/image/7.png',width=600)
    st.write('''
    以$Cr(CO)_6$为例说明σ-π配键的形成：
    Cr(0)，$3d^54s^1$，采取$d^2sp^3$杂化，每个杂化轨道接受CO中碳上的孤对电子，形成6个σ配键;
    这两种键合成一起称为σ-π配键或σ-π电子授受键，这种电子的授受作用是同时进行的，σ配键的形式增加了中心原子或离子的反馈作用，而反馈作用使π键形成，减少中心原子或离子的负电荷，又有利于σ键的形成，这种互相配合，加强的作用称为“协同作用”，这种协同作用的结果是：
    **加强中心离子或原子与配位体的结合，削弱配位体内部的结合。**
    #### 羰基配合物和某些小分子配位化合物
    -CO与过渡金属形成的稳定配合物称为羰基配合物；

    除了锆和铪的羰基配合物尚未制备出来，其余全部过渡金属都能形成羰基配合物；

    羰基配合物有单核配合物与多核配合物，例如：$Fe(CO)_5，Co_2(CO)_8$；

    羰基配合物的分子式特点：金属原子提供的价电子数与CO提供的价电子数加起来一般满足18电子层结构，用此特点可帮助判断羰基配合物的配位数

    等电子体：$N_2、NO^+、CN^-$与CO等电子分子，结构相似，可与过渡金属形成配合物。

    NO比CO多一个电子，且处在$π^*$轨道上。NO配位到过渡金属原子一般有三种键合方式：
    - 直线式端基配位，N以sp杂化，向金属原子提供一对σ电子形成 M<-NO 配键，还提供一个π电子和两个$π^*$轨道形成 M->NO配键，表现为三电子配体。
    - 弯式端基配位，N以$sp^2$杂化，向金属原子提供一个电子形成σ键，键角约120°；
    - 桥键配位，N以$sp^2$杂化连接两个金属原子，或N以sp杂化连接三个金属原子，表现为三电子配体，形成两个或三个σ配键。
    #### 不饱和烃配合物
    不饱和烃分子与中心离子（原子）一般也是形成σ-π配键，因为不饱和烃分子有成键π轨道和$π^*$反键轨道，前者将电子给予中心原子，后者从中心原子接受电子，形成电子授受键。
    ''',unsafe_allow_html=True)
    st.image('GROUP3/image/8.png',width=600)
    st.write('''
    从以上讨论形成σ-π配键情况看，形成σ-π配键并不限于CO配位体和烯烃配位体，只要配位体具有提供电子对的能力及有π型空轨道，均能与中心离了或原子形成σ-π配键；
    例如：$CN^-，R_3P（P的d空轨道），N_2$等分子作为配位体均能与中心离子（原子）形成σ-π配键，所以在配合物中形成σ-π配键是较普遍的；
    从分子轨道理论来看，存在高能空轨道的配位体均能形成σ-π配键；
    在σ-π配键中，有的σ键为主（d电子较少），有的以π键为主（d电子较多），有的两者相当，各种情况视具体配合物而定。
    ### 金属-金属键
    $K_2[Re_2Cl_8]·2H_2O$的晶体结构中最有意义的内容是二价负离子$[Re_2Cl_8]^{2-}$的$D_{4h}$对称构型，它具有很短的Re-Re距离224pm，比金属铼中Re-Re间的平均距离275pm短得多。另一不平常的特色是它的Cl原子间采用重叠式构型，按照Cl原子的范德华半径和为360pm，理应期望它为交错式构型。这两个特色都是由于它形成了Re-Re四重键所致。
    $[Re_2Cl_8]^{2-}$离子的骨干和成键情况可表述如下：每个Re原子利用它的一组平面四方形的$dsp^2(d_{z^2-y^2},s,p_x,p_y)$杂化轨道和配位Cl原子的p轨道形成Re-Cl键。Re原子的$p_z$轨道不用来成键。每个Re原子还剩下$d_{z^2},d_{xz},d_{yz}和d_{xz}$4个原子轨道，它们互相和另一Re原子的相同原子轨道叠加产生分子轨道。
    ## 金属簇合物
    ### 定义
    3个或3个以上金属原子，相互通过金属-金属键连接，形成以金属原子为顶点的多面体或缺顶多面体的核心骨干，周围连接配位体的配合物。
    ### 金属键计算
    根据18电子规则的基本要点，金属-金属间形成键，相互提供电子，但每个金属原子仍有满足18价电子的趋势，以形成稳定的结构。金属-金属间形成键的总数(b)用键数表示：

    $b=0.5*(18n-g)$

    g: 分子中与Mn有关的价电子数(以下三部分）：
    - 簇合物中n个M原子的所有价电子；
    - 配体提供给M原子的电子数；
    - 簇合物的电荷数。

    一些常见的配合物中心离子骨干形态
    ''',unsafe_allow_html=True)
    st.write('''三核：''',unsafe_allow_html=True)
    st.image('GROUP3/image/9.png',width=600)
    st.write('''四核：''',unsafe_allow_html=True)
    st.image('GROUP3/image/10.png',width=600)
    st.write('''五核：''',unsafe_allow_html=True)
    st.image('GROUP3/image/11.png',width=600)
    st.write('''六核：''',unsafe_allow_html=True)
    st.image('GROUP3/image/12.png',width=600)
    st.write('''六核以上：''',unsafe_allow_html=True)
    st.image('GROUP3/image/13.png',width=600)
    if 'show_code' not in st.session_state:
        st.session_state.show_code = False
        # 在侧边栏创建一个复选框来控制代码的显示
    st.session_state.show_code = st.sidebar.checkbox('显示代码', value=st.session_state.show_code)
    if st.session_state.show_code:
        with open(__file__, 'r') as file:
            code = file.read()
        st.markdown("```python\n" + code + "\n```", unsafe_allow_html=True)