'''
Author: cathayana populuscathayana@gmail.com
Date: 2024-01-25 13:27:43
LastEditors: cathayana populuscathayana@gmail.com
LastEditTime: 2024-01-25 13:47:41
FilePath: /website_structure/STRUCTURE_WEBSITE/GROUP4/page_5.py
Description: 

Copyright (c) 2024 by Cathayana, All Rights Reserved. 
'''
import importlib
import streamlit as st

def show():
    st.markdown("""
        # 五、 点群应用
        ### ①
        点群是数学的一个重要分支，其重要性体现在多个领域。
        
        首先，在数学领域，群论的概念在数学的许多分支都有出现，而且群论的研究方法也对抽象代数的其它分支有重要影响。
        
        其次，在物理学和化学的研究中，许多不同的物理结构，如晶体结构和氢原子结构可以用群论方法来进行建模。
        群论在物理学和化学中有大量的应用。如以下几个方面：
        
        1. 晶体结构分析：
        在材料科学和固体物理学中，晶体结构的分析是至关重要的。通过使用点群理论，科学家可以确定晶体对称性，这对于理解材料的物理和化学性质以及设计新材料至关重要。
        
        2. 化学反应：
        在化学中，分子可以通过点群理论进行分类和描述。例如，有机化学的分子反应可以通过对称性原理来理解和预测。
        
        3. 图像处理：
        在图像处理中，点群理论可以用于分析和识别图像中的对称性。例如，在计算机视觉和机器学习中，使用点群理论可以自动检测图像中的形状和模式。
        
        4. 音乐理论：
        在音乐理论中，音乐中的对称性和变换可以通过点群理论来描述和分析。例如，调性变换和音乐对称性的研究可以通过群论的方法来进行。
        ### ②
        对称性在基础科学中得到了广泛而深入的研究，迄今已取得许多成熟的成果。然而，对称性在应用科学中的广泛存在与它所受到的关注并不成正比。通过大量的算例研究，发现几乎所有的机械结构都具有对称性，并且大多数都具有点群对称性的特征。

        因此，可以将晶体学中的点群对称性概念扩展到机械领域，并根据机械结构进行调整。
        
        浙江大学计算机辅助设计与计算机图形学国家重点实验室的陈秀明等人提出了机械点群对称性的分类方法，并通过实例说明了点群对称性在机械中的应用。然后，他们对对称性的要求进行了分析和比较，并利用数据挖掘软件RapidMiner挖掘需求与对称性之间的关联规则。基于挖掘结果，他们总结了点群对称的4个选择原则，为结构设计提供思路。最后，他们综合运用挖掘结果和选型原理，发明了一种新型径向力平衡齿轮泵。

        参考文献：
        https://doi.org/10.3390/sym12091507


    """)
    if 'show_code' not in st.session_state:
        st.session_state.show_code = False
        # 在侧边栏创建一个复选框来控制代码的显示
    st.session_state.show_code = st.sidebar.checkbox('显示代码', value=st.session_state.show_code)
    if st.session_state.show_code:
        with open(__file__, 'r') as file:
            code = file.read()
        st.markdown("```python\n" + code + "\n```", unsafe_allow_html=True)