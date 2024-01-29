'''
Author: cathayana populuscathayana@gmail.com
Date: 2024-01-27 15:54:02
LastEditors: cathayana populuscathayana@gmail.com
LastEditTime: 2024-01-29 12:53:45
FilePath: /website_structure/STRUCTURE_outside/GROUP2/page_0.py
Description: 

Copyright (c) 2024 by Cathayana, All Rights Reserved. 
'''
import importlib
import streamlit as st

def show():
    st.title('Introduction')
    st.write('''# 第一章：量子力学基础和原子结构''')
    st.write('''GROUP2：cjr（组长）、wtb、lsc、ljn、lx、zlj''')
    if 'show_code' not in st.session_state:
        st.session_state.show_code = False
        # 在侧边栏创建一个复选框来控制代码的显示
    st.session_state.show_code = st.sidebar.checkbox('显示代码', value=st.session_state.show_code)
    if st.session_state.show_code:
        with open(__file__, 'r') as file:
            code = file.read()
        st.markdown("```python\n" + code + "\n```", unsafe_allow_html=True)