'''
Author: cathayana populuscathayana@gmail.com
Date: 2024-01-25 13:27:21
LastEditors: cathayana populuscathayana@gmail.com
LastEditTime: 2024-01-29 12:54:29
FilePath: /website_structure/STRUCTURE_outside/GROUP4/page_0.py
Description: 

Copyright (c) 2024 by Cathayana, All Rights Reserved. 
'''
import streamlit as st

def show():
    st.title("Introduction")
    st.write("")
    st.markdown("""
        ### 群论和对称性
        #### 一、 对称元素与对称操作
        #### 二、 群的基本概念&对称操作的乘积
        #### 三、 几种点群整理
        #### 四、 例题分析
        #### 五、 点群应用
        #### 作者与分工
        GROUP4:

        组长：jxy（例题分析整理）

        组员：dyx（整合）；cwy（点群整理）；xmy（点群整理，点群应用）；
        lph（对称元素整理，点群应用）；wcy（群的基本概念整理，网页部署）
    """)
    if 'show_code' not in st.session_state:
        st.session_state.show_code = False
        # 在侧边栏创建一个复选框来控制代码的显示
    st.session_state.show_code = st.sidebar.checkbox('显示代码', value=st.session_state.show_code)
    if st.session_state.show_code:
        with open(__file__, 'r') as file:
            code = file.read()
        st.markdown("```python\n" + code + "\n```", unsafe_allow_html=True)
