'''
Author: cathayana populuscathayana@gmail.com
Date: 2024-01-25 13:57:28
LastEditors: cathayana populuscathayana@gmail.com
LastEditTime: 2024-01-29 13:05:03
FilePath: /website_structure/STRUCTURE_outside/GROUP5/page_0.py
Description: 

Copyright (c) 2024 by Cathayana, All Rights Reserved. 
'''
import urllib.error
import streamlit as st


def show():
    st.title('Introduction')
    st.markdown(
        """
        本网站包括了GROUP5：whs、zzh、gjj、lya、wyh、yyf制作的结构化学固体化学章节的笔记，对应课本第7、8、9章。

1、第七章 wyh
2、第八章 yyf
3、第九章 whs
4、整合 lya
5、整合及网站构建 zzh
6、展示 gjj

源文件可见于https://github.com/Observer299792458/structural-chemistry-work
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