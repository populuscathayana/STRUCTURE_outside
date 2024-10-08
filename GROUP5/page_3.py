import importlib
import streamlit as st

def show():
    pages = st.sidebar.radio('choose a page',
                             ('离子键和晶格能',
                              '离子半径',
                              '离子化合物的晶体结构',
                              '硅酸盐的结构化学 '
                              ))
    if pages == '离子键和晶格能':
        st.markdown("""
        ## 离子键
离子键由正负离子间电子库伦作用力形成的化学键，无方向性和饱和性，它的强度与离子电价成正比，与键长成反比。
[补充]:正负离子可看作不等径圆球的密堆积；堆积过程中正负离子尽可能多地与异号离子接触从而降低体系能量，这也是离子化合物配位数高的原因。 
        """)
        st.markdown("""
        ## 晶格能
晶格能是指在0K时,lmol离子化合物中的正负离子，由相互远离的气态离子，结合成离子晶体时所释放出的能量，晶格能也被叫做点阵能。即
$$M^+(g) + X^-(g) \\rightarrow MX(s) + U$$  

离子晶体的形成是由两方面的作用力即正负离子间的吸引力和排斥力处于相对平衡的结果。两个电荷为($Z_+$)e和($Z_-$)e、距离为r的球形离子的吸引势能为

$$\\epsilon _C=-\\cfrac{Z_+Z_-e^2}{4\\pi \\epsilon _0 r}$$  

当这两个离子近距离接触时，电子云间的排斥能为

$$\\epsilon _R=\\cfrac{B}{{4\\pi \\epsilon _0 r^m}}$$  

其中m称为玻恩指数，m大小与离子的电子层结构有关。  
""")
        st.image("GROUP5/img/9-1.png")
        st.markdown("""

总库伦势能为

$$\\epsilon=-\\cfrac{Z_+Z_-e^2}{4\\pi \\epsilon _0 r}+\\cfrac{B}{{4\\pi \\epsilon _0 r^m}}$$  

对$\\epsilon$求导，当导数为0时，势能最低，此时的r为平衡距离$r_0$，代入总势能式得到  

$$\\epsilon=-\\cfrac{Z_+Z_-e^2}{4\\pi \\epsilon _0 r_0}(1-\\cfrac{1}{m})$$    

在晶体中，正负离子按照一定的规律排列，每个离子的周围都有许多正负离子和它相互作用。现以NaCl型晶体为例，了解晶体中离子间作用能的情况。设$Na^+$和$Cl^-$最近的距离r时，每个$Na^+$周围有6个距离为$r$的$Cl^-$，12个距离为$\\sqrt{2}r$的$Na^+$，8个距离为$\\sqrt{3}r$的$Cl^-$，6个距离为$\\sqrt{4}r$的$Na^+$。

其库伦势能为

""")
        st.image("GROUP5/img/9-x.png")
        st.markdown("""

式中$A$为马德隆常数,例如在$NaCl$中A就是与晶体结构有关的常数计算结果，约为1.748
总势能可改写为

$$\\epsilon=-\\cfrac{Z_+Z_-e^2N_A}{4\\pi \\epsilon _0 r_0}(1-\\cfrac{1}{m})A$$  

A与结构形式有关

""")
        st.image("GROUP5/img/9-2.png")
    if pages == '离子半径':
        st.markdown("""
## 性质
离子键没有方向性和饱和性，所以可以把离子晶体看作不等径圆球的堆积问题。离子半径之和即决定了离子键的键长。正负离子的半径比决定了正负离子的配位数，从而影响了一个离子晶体的立体构型。

### 计算  

①哥希密特半径  

假如能决定一种元素的离子半径，那么其它元素的离子半径就可以从有关的离子的接触距离推算出来。  

正负离子间排列可有三种情况：

> i负离子互相接触，正、负离子不接触；  
>
> ii正负离子，负-负离子间都正好接触；  
>
> iii正离子较大，正、负离子间接触，负-负离子间不接触。
""")
        st.image("GROUP5/img/9-3.png")
        st.markdown("""

从x衍射结果得到哪种离子化合物是哪一型的晶体，如图中（1）中情况，就能算出负离子半径，再反推（2）（3），就可以得出相同负离子情况下正离子的离子半径。，哥希密特运用这个方法确定了八十多种离子半径的数值。  

②鲍林半径  

鲍林认为

I.离子半径与离子核外电子排布情况和核对电子的作用大小有关，离子的大小主要由外层电子分布决定；

Ⅱ.同电子构型的离子的半径
$$r_1=\\cfrac{C_n}{z-\\sigma}$$  

其中r_1为离子的单价半径，C_n为比例常数，σ为屏蔽常数。一般而言对于正负离子同主量子数n，他们的C_n一样，用正离子半径比上负离子半径只需计算屏蔽常数，又由x衍射实验得到正负离子与晶体常数关系，那么就能算出正负离子半径。

将正负离子半径带回又能得到C_n，但是对于高价(非±1）离子，他们的实际离子半径
$$r=r_1Z^\\cfrac{-2}{m-1}$$  

其中m为波恩指数，Z为电荷量。  
""")
        st.image("GROUP5/img/9-4.png")
        st.markdown("""

## 配位数对离子半径的影响  

$$R_0=\\cfrac{mB}{Z^2e^2N_AA}^{\\cfrac{1}{m-1}}$$  

对NaCl和CsCl，假设$m_{NaCl} \\approx m_{CsCl}$，则有  

$$\\cfrac{R_{CsCl}}{R_{NaCl}}=(\\cfrac{B_{CsCl}}{B_{NaCl}}\\cdot \\cfrac{A_{NaCl}}{A_{CsCl}})^{\\cfrac{1}{m-1}}$$  

B是排斥常数，在1mol离子晶体中，每个离子周围的异号离子多少，即每个离子周围的配位数多少应与B有关，配位数越多，B越大，故B与配位数成正比。每个正离子，在CsCl型结构与8个负离子相接触，而在NaCl型结构中，只与6个负离子相接触，所以可认为$\\cfrac{B_{CsCl}}{B_{NaCl}}=\\cfrac{8}{6}$，再根据$A_{NaCl}和A_{CsCl}$即可求出CsCl和NaCl中正负离子的距离比。  

## 离子半径的变化规律  
①同一周期的正离子的半径随着正离子价数增加而减小；  
②同族元素离子半径自上而下增加；  
③周期表中左上方和右下方的正离子对角线中，离子半径近似相等；  
④同一元素的各种价态，离子的电子数越多，半径越大；  
⑤负离子的半径较大，正离子的半径较小。  


### 离子配位多面体   

正负离子间排列可有三种情况，当正负离子不相接触，而负离子自己相互接触时，静电排斥力大，而吸引力小，晶体不很稳定；当正负离子相接触，而负离子自己不接触时， 静电吸引力大，排斥力小，晶体比较稳定。当正负离子，负-负离子间都正好接触时，正离子的有效配位数最大，势能最低。所以晶体稳定的最优条件是正负离子相互接触，配位数尽可能的高。按这个最优条件可以求得如下表所示的配位多面体的半径比的极限(最小)值。
""")
        st.image("GROUP5/img/9-5.png")
        st.markdown("""

正负离子半径比只是影响晶体结构的一个因素，在复杂多样的离子晶体中，还有其他因素影响晶体的结构：  

①M—X间共价键的成分增大，中心原子按一定数目的共价键的方向性与周围原子形成一定几何形状的配位体，并使M -X键缩短。  

②某些过渡金属常形成M -M键，使配位多面体扭歪，例如$VO_2、MnO_2、ReO_2$形成扭歪的金红石型结构。  

③$M^{n+}$周围$X^{-}$的配位场效应使离子配位多面体变形。  

为了使晶体保持电中性，必然要求正负离子等当，即正离子数×正离子电价=负离子数×负离子电价，正负离子数由化合物组成决定。如$CaF_2$，一个正离子两个负离子，他们的比为1：2；在晶体中配位数一定是2：1。

[补充]用结晶化学语言来描述简单离子晶体的结构时，主要说明负离子的堆积方式和正离子所占据空隙的种类和分数。

## 结晶定律  

(1)哥希密特结晶化学定律  

晶体的结构型式，取决于其组成者的数量关系，大小关系与极化性能，组成者系指原子、离子或原子团； 

组成者的数量关系——指晶体的化学组成类型，对无机化合物晶体，一般按AB型，AB$_2$型等类型来讨论结构型式，化学组成类型不同，晶体结构型式也不同。

组成者的大小关系——指组成者之间的半径比，说明相对大小不同，晶体结构型式受到影响。  

组成者的极化性能——指由极化作用而对结构型式的影响。  

(2)类质同晶与同质多晶现象  

类质同晶现象是指一些化学式相似的物质，具有相似的晶体外形，具有同晶现象的物质称为同晶体,其原因一方面是具有相同的化学组成类型，另一方面是相应离子的半径相近或离子半径比相近。  

同质多晶现象是指同一种化学组成的物质，可以形成多种晶体结构类型的变体，即相同化学成份，但晶体形态、构造和性质不同的晶体，不同变体在一定的物理条件下可以发生晶型转变。
其原因较复杂，根据我们已学的知识，有两个原因。

第一，由于化学组成类型和离子半径比不变，正负离子各自的配位数不变，但离子的堆积方式不同，正离子占据空隙的种类，分数不同，这样就产生不同的变体。 
第二，同一物质在不同温度条件下，由于极化作用的影响而产生同质多晶现象。  

(3)鲍林规则  

①第一规则：离子配位多面体规则

正离子与负离子之间的距离取决于正负离子半径之和，而正离子配位多面体的型式和配位数取决于半径之比。  

②第二规则：电价规则

在一个稳定的离子化合物结构中，每一负离子的电价等于或近乎等于从邻近的正离子至该负离子的各静电键的强度的总和。

电价规则主要用在多元离子晶体中，因在多元离子晶体中一个负离子可与几个种类不同的正离子相连。
例如$（Si_2O_7)^{6-}$中有一个氧与两个Si以Si-O-Si的σ键相连，几何构型为公用一个顶点的两个正四面体，计算公共顶点氧的电价

$Si^{4+}：Z_+=4,CN_+=4;Z_-=2×Z_+//CN_+=2$,故公共氧的电价为2，符合预期。

电价规则说明的是一个负离子与几个正离子相连的问题，也可以说讨论的是公用同一顶点的配位多位体数目。

③第三规则：共用顶点、棱边和面的规则

在一个配位结构中，公用的棱，特别是公用的面的存在会降低这个结构的稳定性。  

离子晶体结构稳定性降低主要是由正离子间的库仑斥力所引起。当2个四面体共边连接时，将使位于四面体中心的正离子之间的距离缩短。共棱时的距离只有共顶点时的116/200 = 0.58，共面时的距离则只有共顶点时的0.33。所以，共边和共面连接，相应的库仑斥力项增大，晶体的稳定性降低。 

""")
    if pages == '离子化合物的晶体结构':
        st.markdown("""
## 1.NaCl型

在NaCl晶体结构中，正负离子的配位情况相同，配位数均为6，都是八面体配位。$Cl^-$采取立方最密堆积，$Na^+$占据所有八面体空隙。许多二元离子化合物的晶体结构属NaCl型，键型可以从典型的离子键过渡到共价键或金属键，其电性、磁性和力学性能亦随键型变化产生很大差异，下图为NaCl型晶体的结构。  
""")
        st.image("GROUP5/img/9-6.png")
        st.markdown(""" 
### 2.NiAs  
在NiAs晶体结构中，As作六方最密堆积，Ni处在As的六方最密堆积的配位八面体中，而As处在由Ni形成的配位三方柱体中。从NiAs的晶胞参数(a = 360.2 pm, c=500.9pm)可以看出，Ni-Ni间的距离只有250pm，与金属镍中的距离一致。NiAs晶体有明显的金属性。  

""")
        st.image("GROUP5/img/9-7.png")
        st.markdown("""  

## 3.ZnS  

ZnS的晶体结构可看作$S^{2-}$作最密堆积，$Zn^{2+}$填在一半四面体空隙之中，填隙时互相间隔开，使填隙四面体不会出现共面连接或共边连接。立方ZnS结构是$S^{2-}$作立方最密堆积，六方ZnS结构$S^{2-}$作六方最密堆积。  
""")
        st.image("GROUP5/img/9-8.png")
        st.markdown(""" 

## 4.CaF$_2$  
CaF$_2$的结构可看作是$Ca^{2+}$作立方最密堆积，$F^-$填在所有四面体空隙中。  
""")
        st.image("GROUP5/img/9-9.png")
        st.markdown("""  


## 5.TiO$_2$  
在$TiO_2$中,$O^{2-}$近似地堆积成六方密堆积结构，密置层垂直晶胞a轴延伸，$Ti^{4+}$填入其中一半的八面体空隙，而$O^{2-}$周围 有3个近于正三角形配位的$Ti^{4+}$。从结构的配位多面体连接来 看，每个$[TiO_6]$八面体和相邻2个八面体共边连接成长链，链平行于四重轴，链和链沿垂直方向共用顶点连成三维骨架。  
""")
        st.image("GROUP5/img/9-10.png")
        st.markdown("""   

## 6.CdI$_2$、CdCl$_2$  
层型晶体结构的特点：正负离子都按等径球紧密堆积排列成为层型，每层由负离子作密置双层排列，而正离子填充其八面体空隙（近似看为一层），层内A—B键已带有相当共价成份的离子键，层与层之间的结合力以分子间作用力为主。  

在$CdI_2$ 晶体中，层型分子沿垂直于层的方向堆积,$I^-$作六方最密堆积，$Cd^{2+}$填入其中部分八面体空隙中。CdCl$_2$ 晶体中$Cl^-$作立方最密堆积，$Cd^{2+}$交替地一层填满一层空缺地填入八面体空隙中。  

""")
        st.image("GROUP5/img/9-11.png")
        st.markdown("""  

### 7.CsCl  
CsCl 结构可看作$Cl^-$作简单立方堆积，$Cs^+$填入立方体空隙中形成的。  

""")
        st.image("GROUP5/img/9-12.png")
        st.markdown("""  

### 8.CaTiO$_3$  
钙钛矿($CaTiO_3$ )结构可看作$O^{2-}$和$Ca^{2+}$—起有序地作立方最密堆积，可划分出面心立方晶胞，晶胞的顶点为$Ca^{2+}$，面心为$O^{2-}$。$Ti^{4+}$占据晶胞体心位置，即由6个$O^{2-}$组成的八面体空隙中心(晶胞中共计4个八面体空隙，但只有处在晶胞中心的八面体空隙是由6个$O^{2-}$组成,其他处在棱边中心的3个八面体空隙都是由4个$O^{2-}$和2个$Ca^{2+}$组成)。  

""")
        st.image("GROUP5/img/9-13.png")
        st.markdown("""   

### 9.尖晶石型 (XY$_2$O$_4$型)  
基本结构是$O^{2-}$按ABC顺序在垂直于(111)方向堆积。四面体与八面体层相间，四面体与八面体数之比为2:1。尖晶石结构可看作氧离子形成立方最紧密堆积，再由X离子占据64个四面体空隙的1/8，即8个A位，Y离子占据32个八面体空隙的1/2，即16个B位。  

尖晶石的晶体包括正常尖晶石型结构、反尖晶石型结构。

正常尖晶石型结构：$O^{2-}$成立方紧密堆积，三价阳离子占据八面体空隙，二价阳离子占据四面体空隙。正尖晶石结构，结构通式$XY_2O_4$，X为二价阳离子，Y为三价阳离子。其中X占据四面体位置，Y占据八面体位置。  

反尖晶石型结构：二价阳离子和半数三价阳离子占据八面体空隙，另半数三价阳离子占据四面体空隙。反尖晶石型结构又称倒置尖晶石型结构。若结构中所有的X阳离子和一半的Y阳离子占据八面体位置，另一半Y阳离子占据四面体位置，则称反尖晶石结构。结构通式$Y[XY]O_4$。  

""")
        st.image("GROUP5/img/9-14.png")
    if pages == '硅酸盐的结构化学 ':
        st.markdown("""
        ## 基本介绍
### (1)主要成分及作用
硅酸盐的主要成分是硅和氧，它是数量极大的一类无机物，约占地壳重量的80%。地壳中的岩石、砂子、黏土、土壤，建筑材料中的砖瓦、水泥、陶瓷、玻璃，大都由硅酸盐组成。在各种矿床中，硅酸盐起着重要的作用，它几乎是所有金属矿物的伴生矿物，而有的硅酸盐本身就是金属矿物（如Be,Li,Zn, Ni等金属矿）或非金属矿物（如云母、滑石、石棉、高岭石等）。  
### (2)基本结构
在硅酸盐中，结构的基本单位是$[SiO_4]$四面体，四面体互相公用顶点连接成各种各样的结构型式。四面体的连接方式决定硅氧骨干的结构型式，是了解硅酸盐结构化学的基础。  
### (3)Al在硅酸盐化学中的特殊作用
在硅酸盐化学中，铝具有特殊的作用。由于$Al^{3+}$的大小和$Si^{4+}$相近，$Al^{3+}$可以无序地或有序地置换$Si^{4+}$，置换数量有多有少，这时A1处在四面体配位中和Si一起组成硅铝氧骨干， 形成硅铝酸盐。为了保持电中性,每当骨干中有$Al^{3+}$置换$Si^{4+}$时，必然伴随着引入其他正离子补偿其电荷。$Al^{3+}$的大小又适合于处在配位数为6的配位八面体中，这时$Al^{3+}$又可以作为硅氧骨干外的正离子，起平衡电荷的作用。
### (4)硅酸盐结构特征
①除少数例外，硅酸盐中Si处在配位数为4的$[SiO_4]$四面体中，其键长、键角的平均值为：$d(Si—O) = 162pm$, $\\angle OSiO= 109. 5°$，$\\angle SiOSi = 140°$。  
②在天然硅酸盐中置换作用非常广泛而重要。A1置换Si形成硅铝酸盐就很普遍。Al也可占据配位八面体(这时常称为硅氧骨干外的离子)。A1置换Si伴随有正离子进入，以平衡其电荷。  
③$[(Si,Al)O_4]$只共顶点连接，而不共边和共面，而且2个Si—O—A1的能量比1个A1—O一A1和1个Si—O—Si的能量低。  
④$[SiO_4]$四面体的每个顶点上的$O^{2-}$最多只能公用于2个这样的四面体之间。  
⑤在硅铝酸盐中，硅铝氧骨干外的金属离子容易被其他金属离子置换,置换不同的离子，对骨干的结构影响较小，但对它的性能影响很大。
⑥硅氧基团是以共价结合的，而硅氧负离子与金属正离子是以离子键结合的：如果硅氧基团是有限的结构单元，如正硅酸根，$（Si_2O_7)^{6-}$，等基团为结构单元，那么此时硅酸盐晶体为离子型晶体；如果是无限的链状或层状结构，那么硅酸盐晶体是混合键型的；如果硅氧基团发展为立体结构，此时晶体为共价型的$SiO_2$晶体。  

## 硅酸盐晶体结构
链型硅酸盐可分为单链和双链两类。单链的特点是每个$[SiO_4]$四面体共用两个顶点，连成一维无限长链。如硅灰石（$CaSiO_3$）.透辉石（$CaMg(SiO_3)_2$），下图示出几种周期较短的单链连接方式。  
""")
        st.image("GROUP5/img/9-15.png")
        st.markdown("""

### (1)分立型硅酸盐

这类硅酸盐晶体又可分为：

①具有单独硅氧四面体$[SiO_4^{4-}]$的正硅酸盐，如橄榄石（$Mg_2SiO_4$）、错英石（$ZrSiO_4$）、石榴石（$Mg_3Al_2(SiO_4)_3$）等。在这类晶体中氧硅的原子比等于或大于4（大于4指存在有$SiO_4^{4-}$以外的氧），且硅氧基团之间被一些金属离子（ $Mg^{2+}、Ca^{2+}、Fe^{3+}、Al^{3+}$等）隔开。   
②含有非环状有限硅氧基团的硅酸晶体，在这类晶体中，氧硅原子比小于4；此种晶体的硅氧基团为$Si_2O_7^{6-}$；硅氧基团之间由$Mg^{2+}、Ca^{2+}、Zn^{2+}$等金属离子来联系。  
③含有环状有限硅氧基团的硅酸盐晶体  
在这类晶体中，氧硅原子比小于4；此类硅氧基团有$Si_3O_9^{6-}、Si_4O_{12}^{8-}、Si_5O_{16}^{12-}、Si_6O_{18}^{12-}$等；同样硅氧基团之间由各种金属离子来联接。

### (2)链型硅酸盐 

链型硅酸盐可分为单链和双链两类。单链的特点是每个$[SiO_4]$四面体共用两个顶点，连成一维无限长链。如硅灰石（$CaSiO_3$）.透辉石（$CaMg(SiO_3)_2$），下图示出几种周期较短的单链连接方式。  
""")
        st.image("GROUP5/img/9-16.png")
        st.markdown("""

双链结构中有一部分$[SiO_4]$四面体公用3个顶点而互相连接。下图示出几种双链的结构。  
""")
        st.image("GROUP5/img/9-17.png")
        st.markdown("""

链型硅酸盐中，链内Si—O键属共价型键，结合力较牢，链之间的金属离子与O键相对较弱，因此此类晶体容易群裂成为柱体或纤维，例如石棉为重要的工业纤维，可做成石棉绳、石棉布等。

### (3)层型硅酸盐

层型硅酸盐就是含有层状硅氧基团的硅酸盐，$[SiO_4]$四面体共用3个顶点，由于连接方式的不同，可形成多种类型。层状硅氧基团结构都具有无限的网状结构，化学式为$(Si_2O_5)_n^{2n-}$等，网与网之间主要由金属离子连接。  

在层型硅酸盐中，层的结构可以存在多种型式；层内离子可以互相置换，化学成分可在很大的范围内变化；层间的水分子和金属离子有多有少，可有可无；层间的堆积型式可以有序，也可以无序，其结构和组成随着外界条件(水分的多少、盐的浓度、机械作用力等)改变而变化。所以由层型硅酸盐组成的黏土和土壤，其结构和性质是非常复杂多样的，但它们都是层型结构，沿层方向容易解理，晶粒较小，具有柔软、易水合、容易进行离子交换等共性。
""")
        st.image("GROUP5/img/9-18.png")
        st.markdown("""


### (4)骨架型硅酸盐 

在硅石、长石、沸石等类骨架型硅酸盐中，$[SiO_4]$四面体的4个顶点都相互连接形成三维的骨架。除硅石外，各种骨架型硅酸盐均有$Al^{3+}$置换$Si^{4+}$，使骨架带有一定的负电荷,需在骨架外引入若干正离子，但无论引入多少$Al^{3+}$，铝、硅总和与氧的原子比均为1:2。  

## 沸石分子筛  

多孔化合物与多孔材料的共同特征是具有规则而均匀的孔道结构，其中包括孔道与窗口的大小尺寸和形状；孔道维数、走向；孔壁的组成和性质。其中孔道大小又分为微孔（＜2nm）；介孔（2-50nm）；大孔（＞50nm)。

分子筛是一种架型硅铝酸盐。在硅铝酸根中，$Al^{3+}$和$Si^{4+}$均处在$O^{2-}$所组成的四面体空隙中，但一般$Al^{3+}$取代$Si^{4+}$的数目不超过总数的一半，因为铝氧静键强度比硅氧的静电键强度小，这样会在结构中出现铝氧四面体共用一条棱的情况，根据鲍林第三规则，结构中引入不稳定因素从而使骨架强度削弱。  

分子筛的结构单元有硅氧四面体和铝氧四面体 (初级结构单元)、环（次级结构单元）和笼。
### (1)硅氧四面体和铝氧四面体
Si和O都以正四面体方式连接，Si位于氧负离子形成的正四面体空隙中；铝氧四面体同样。但是铝氧四面体中Al-O键；O-O键键长比硅氧四面体稍大。
### (2) 环
硅氧四面体或铝氧四面体通过公用四面体顶点的氧原子连接形成各种形状大小的环。
""")
        st.image("GROUP5/img/9-y.jpeg")
        st.markdown("""
### (3)笼
又分为
①正方体笼，由6个四元环构成，笼体积很小，一般分子都进不去。

②六棱柱笼，由六个四元环和两个六元环构成，体积也很小，只可容纳一个离子或小分子。  

③β笼（又称方钠石笼），β笼是由24个硅（铝）氧四面体连接而成的孔穴，它是一个十四面体，由立方体和八面体围聚而成，如下图。  
""")
        st.image("GROUP5/img/9-19.png")
        st.markdown("""

④α笼，由6个八元环，8个六元环和12个四元环构成。其外形可以看作是削去了正方体的8个顶点和12条棱，原来的6个面变成了6个八元环，8个顶点变成了8个六元环，12条棱变成了12个四元环。 笼共有26个面，48个顶点，孔穴平均直径达1140pm，最大窗孔为八元环，孔径410pm。  

⑤八面沸石笼，由18个四元环，4个六元环和4个十二元环构成。每个十二元环的12条边中有3个与六元环共用，9个与四元环共用，每两个六元环之间被三个四元环相隔。共有26个面，48个顶点。八面沸石笼可以近似看作是正四面体先削去4个角，再将每条棱削成三个正方形，这样，原来的4个顶点变成了4个六边形，原来的4个面变成了4个十二边形，原来的每一条棱变成了3个正方形。需要注意的是，八面沸石笼的十二元环不在同一平面上。  
""")
        st.image("GROUP5/img/9-20.jpg")
        st.markdown("""

不同结构的笼再通过氧桥互相联结形成各种不同结构的分子筛，主要有A-型、X型和Y型。

这些分子筛具有吸附量大，选择性高的特点，在吸附材料，催化，离子交换等领域有重要作用。
""")
    if 'show_code' not in st.session_state:
        st.session_state.show_code = False
        # 在侧边栏创建一个复选框来控制代码的显示
    st.session_state.show_code = st.sidebar.checkbox('显示代码', value=st.session_state.show_code)
    if st.session_state.show_code:
        with open(__file__, 'r') as file:
            code = file.read()
        st.markdown("```python\n" + code + "\n```", unsafe_allow_html=True)