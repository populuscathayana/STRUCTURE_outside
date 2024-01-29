'''
Author: cathayana populuscathayana@gmail.com
Date: 2024-01-25 13:42:45
LastEditors: cathayana populuscathayana@gmail.com
LastEditTime: 2024-01-29 12:52:45
FilePath: /website_structure/STRUCTURE_outside/GROUP1/page_0.py
Description: 

Copyright (c) 2024 by Cathayana, All Rights Reserved. 
'''
import importlib
import streamlit as st

def show():
    st.title('Introduction')
    st.write(''' 
                ### 第四章：多原子分子的结构和性质
                页面1：主要内容
            ''')
    st.write("页面2：Hückel Molecular Orbital Theory Calculator")
    st.write("页面3：一些Cathayana个人编写的未在前文出现的/与本章节无关的程序，")
    st.write("作者：cathayana(组长),cyh,jjk,ccz,czh,ghr")
    if 'show_code' not in st.session_state:
        st.session_state.show_code = False
        # 在侧边栏创建一个复选框来控制代码的显示
    st.session_state.show_code = st.sidebar.checkbox('显示代码', value=st.session_state.show_code)
    if st.session_state.show_code:
        with open(__file__, 'r') as file:
            code = file.read()
        st.markdown("```python\n" + code + "\n```", unsafe_allow_html=True)