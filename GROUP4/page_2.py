'''
Author: cathayana populuscathayana@gmail.com
Date: 2024-01-25 13:16:25
LastEditors: cathayana populuscathayana@gmail.com
LastEditTime: 2024-01-25 13:47:16
FilePath: /website_structure/STRUCTURE_WEBSITE/GROUP4/page_2.py
Description: 

Copyright (c) 2024 by Cathayana, All Rights Reserved. 
'''
import importlib
import streamlit as st

def show():
    st.markdown("""
        # 二、 群的基本概念&对称操作的乘积
        ## 群在数学中的相关定义、概念
        ### 群的定义
        一个集合G含有A、B、C...等元素，在这些元素之间定义一种二元运算（通常称为乘法【这里“乘法”的含义和通常意义上的乘法不一样】），如果满足下述四种性质，则称集合G为“群”。
        ### 定义群的四种性质
        1. 封闭性

        集合中任意两个元素的积，必为集合中的某一元素。即：
        $$A\in G, B\in G$$
        则
        $$AB = C \in G$$

        2. 缔合性（乘法结合律）

        即$$(AB)C = A(BC)$$

        3. 存在单位元素

        单位元素：即集合中任意元素A与单位元素的积都仍为A
        $$EA = AE = A$$
        4. 集合中的任一元素都存在逆元素

        即$$AA^{-1} = A^{-1}A = E$$

        **满足以上四种性质的集合，就叫作*群***

        ### 几个群的例子
        数学中有很多群的例子，比如整数对于加法的运算构成一个群，非零实数集对于乘法的运算构成一个群等

        ## 群在结构化学中
        ### Example
        $H_2O$所具有的对称操作的集合构成一个群，即
        $$G = \{E, C_2^1, \sigma_v, \sigma_v'\}$$
        几个对称因素的模型图如下（对称轴和对称面）
        """)
        
    st.image('GROUP4/232946402fb85ff255838891ffce586.png', width= 300)
        
    st.markdown(
            """
            以上这四种对称操作构成的群称为$C_{2v}$点群。这种由对称操作构成的群，其运算法则是群元素之间的**乘积**（即各种对称操作之间的乘积）。

        **验证以上提到的集合G的四个性质：**

        1. 封闭性
        $$C_2\sigma_v = \sigma_v'$$
        将以上公式译为通常语言，即先关于$C_2$轴旋转一次，再按照xz面进行对称操作，等价于按照yz面进行对称操作。【$\sigma_v$和$\sigma_v'$是包含主轴的两个镜面（$\sigma_h$是垂直主轴的镜面）（主轴即$C_2$旋转轴即z轴）】

        同理还有
        $$\sigma_v\sigma_v' = C_2$$
        $$C_2 \sigma_v'= \sigma_v$$
        2. 结合律
        $$C_2\sigma_v\sigma_v'$$
        3. 单位元素
        存在单位元素E

        4. 逆元素
        每个操作的逆元素都存在于G中，如
        $\sigma_v$、$C_2$、$\sigma_v'$和E的逆元素都是它们自身

        ### 对于群的理解

        - 构成群的对象是及其广泛的，群的元素不仅可以是数学对象，还可以是各种各样的物理操作。
        - 同样，群的运算“*乘法*”不仅可以是数学运算，也可以是物理操作。
        - 群元素的乘积如果满足交换律，则称其为“对易群”或“阿贝尔群”。如上面的$C_{2v}$点群。反之称非对易群。

        ### 阶
        群中元素的数目，就称为“*群的阶*”

        如前述整数群，是无限阶群；水分子对称操作构成的$C_{2v}$群，为四阶群。

        ## 对称操作的乘积
        **定义**：如果一个操作C产生的结果和两个或多个其它操作连续作用的结果相同，则将操作C作为其它操作的乘积，即
        $$C = ABDEF...$$

        如：
        $$C_4^3 = C_2^1C_4^1$$

        **几点注意**

        - 几个相同的对称操作的乘积记为乘方形式$AA = A^2$

        - 多个对称操作连续作用时，所施行操作的次序是重要的，不是所有的对称操作连用都满足交换律，即AB可能 $\\neq$ BA

        - 一般按照从右往左的顺序进行操作

        **应用对称操作的乘积律可以构造乘法表**（即对称操作的等价关系）

        如下

        |$$\pmb{C_{2v}}$$|**E**|$\pmb{C_2^1}$|$\pmb{\sigma_{yz}}$|$\pmb{\sigma_{xz}}$|
        |---|---|---|---|---|
        |**E**|E|$$C_2^1$$|$$\sigma_{yz}$$|$$\sigma_{xz}$$|
        |$$\pmb{C_2^1}$$|$$C_2^1$$|E|$$\sigma_{xz}$$|$$\sigma_{yz}$$|
        |$$\pmb{\sigma_{yz}}$$|$$\sigma_{yz}$$|$$\sigma_{xz}$$|E|$$C_2^1$$|
        |$\pmb{\sigma_{xz}}$|$$\sigma_{xz}$$|$$\sigma_{yz}$$|$$C_2^1$$|E|
    
            """
        )
    if 'show_code' not in st.session_state:
        st.session_state.show_code = False
        # 在侧边栏创建一个复选框来控制代码的显示
    st.session_state.show_code = st.sidebar.checkbox('显示代码', value=st.session_state.show_code)
    if st.session_state.show_code:
        with open(__file__, 'r') as file:
            code = file.read()
        st.markdown("```python\n" + code + "\n```", unsafe_allow_html=True)

        