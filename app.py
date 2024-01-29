'''
Author: cathayana populuscathayana@gmail.com
Date: 2024-01-17 14:37:39
LastEditors: cathayana populuscathayana@gmail.com
LastEditTime: 2024-01-29 13:11:38
FilePath: /website_structure/STR/app.py
Description:

Copyright (c) 2024 by Cathayana, All Rights Reserved.
'''

# app.py
import importlib
import streamlit as st


# 创建侧边栏选择器
page_group = st.sidebar.selectbox('选择页面组', ['网站说明','第一章：量子力学基础和原子结构', '第二章：化学键理论', '第三章：群论和对称性', '第四章：多原子分子的量子化学', '第五章：配位化合物和超分子', '第六章：固体化学','组件测试'])


# 根据选择的页面组显示不同的页面
if page_group == '第一章：量子力学基础和原子结构':
    if 'page1_selected' not in st.session_state:
        st.session_state.page1_selected = '0'
    # 定义按钮的标签和对应的页面编号
    buttons = {
        "Introduction": "0",
        "量子力学基础知识": "1",
        "原子的结构和性质": "2"
    }

    for label, page1_number in buttons.items():
        # 创建自定义文字的按钮
        if st.sidebar.button(label):
            st.session_state.page1_selected = page1_number

    # 根据选中的页面动态加载模块
    if st.session_state.page1_selected is not None:
        module = importlib.import_module(f"GROUP2.page_{st.session_state.page1_selected}")
        module.show()
    else:
        st.sidebar.write("请选择一个页面。")

elif page_group == '第二章：化学键理论':
    if 'page2_selected' not in st.session_state:
        st.session_state.page2_selected = '0'
    # 定义按钮的标签和对应的页面编号
    buttons = {
        "Introduction": "0",
        "化学键理论": "1",
    }

    for label, page2_number in buttons.items():
        # 创建自定义文字的按钮
        if st.sidebar.button(label):
            st.session_state.page2_selected = page2_number

    # 根据选中的页面动态加载模块
    if st.session_state.page2_selected is not None:
        module = importlib.import_module(f"GROUP6.page_{st.session_state.page2_selected}")
        module.show()
    else:
        st.sidebar.write("请选择一个页面。")

elif page_group == '第三章：群论和对称性':
    if 'page3_selected' not in st.session_state:
        st.session_state.page3_selected = '0'
    # 定义按钮的标签和对应的页面编号
    buttons = {
        "Introduction": "0",
        "对称元素与对称操作": "1",
        "群的基本概念&对称操作的乘积": "2",
        "几种点群整理": "3",
        "例题分析": "4",
        "点群应用": "5"
    }
    for label, page3_number in buttons.items():
        # 创建自定义文字的按钮
        if st.sidebar.button(label):
            st.session_state.page3_selected = page3_number

    # 根据选中的页面动态加载模块
    if st.session_state.page3_selected is not None:
        module = importlib.import_module(f"GROUP4.page_{st.session_state.page3_selected}")
        module.show()
    else:
        st.sidebar.write("请选择一个页面。")


elif page_group == '第四章：多原子分子的量子化学':
    if 'page4_selected' not in st.session_state:
        st.session_state.page4_selected = '0'
    # 定义按钮的标签和对应的页面编号
    buttons = {
        "Introduction": "0",
        "多原子分子的结构和性质": "1",
        "Hückel Molecular Orbital Theory Calculator": "2",
        "Cathayana个人脚本": "3",
    }

    for label, page4_number in buttons.items():
        # 创建自定义文字的按钮
        if st.sidebar.button(label):
            st.session_state.page4_selected = page4_number

    # 根据选中的页面动态加载模块
    if st.session_state.page4_selected is not None:
        module = importlib.import_module(f"GROUP1.page_{st.session_state.page4_selected}")
        module.show()
    else:
        st.sidebar.write("请选择一个页面。")


elif page_group == '第五章：配位化合物和超分子':
    if 'page5_selected' not in st.session_state:
        st.session_state.page5_selected = '0'
    # 定义按钮的标签和对应的页面编号
    buttons = {
        "Introduction": "0",
        "配位化学": "1",
        "超分子化学": "2"
    }
    for label, page5_number in buttons.items():
        # 创建自定义文字的按钮
        if st.sidebar.button(label):
            st.session_state.page5_selected = page5_number

    # 根据选中的页面动态加载模块
    if st.session_state.page5_selected is not None:
        module = importlib.import_module(f"GROUP3.page_{st.session_state.page5_selected}")
        module.show()
    else:
        st.sidebar.write("请选择一个页面。")

elif page_group == '第六章：固体化学':
    if 'page6_selected' not in st.session_state:
        st.session_state.page6_selected = '0'
    # 定义按钮的标签和对应的页面编号
    buttons = {
        "Introduction": "0",
        "第一部分": "1",
        "第二部分": "2",
        "第三部分": "3"
    }
    for label, page6_number in buttons.items():
        # 创建自定义文字的按钮
        if st.sidebar.button(label):
            st.session_state.page6_selected = page6_number

    # 根据选中的页面动态加载模块
    if st.session_state.page6_selected is not None:
        module = importlib.import_module(f"GROUP5.page_{st.session_state.page6_selected}")
        module.show()
    else:
        st.sidebar.write("请选择一个页面。")

elif page_group == '网站说明':
    st.title('网站说明')
    st.write('''
                本网站是2023年秋冬学期吴韬老师开设的《结构化学》课程的笔记，由全班同学协力完成。具体每个章节的分工在章节内部可见。
                #### 在左侧边栏选择章节和页面，即可查看对应的笔记。
                #### 全文正文分为六个章节：
            ''')
    st.table({'章节': ['网站构建整合','第一章：量子力学基础和原子结构', '第二章：化学键理论', '第三章：群论和对称性' ,'第四章：多原子分子的量子化学', '第五章：配位化合物和超分子', '第六章：固体化学','组件测试'],
            '负责小组': ['Cathayana','GROUP2', 'GROUP6', 'GROUP4', 'GROUP1', 'GROUP3', 'GROUP5','Cathayana']})
    st.write('本站所有代码开源，点击边栏处的“显示代码”即可查看。项目源码见：https://github.com/populuscathayana/STR')
    st.write('''如果网站出现任何问题，欢迎联系Cathayana \\
        邮箱：populuscathayana@gmail.com \\
        QQ: 1278889459\\
        Wechat：''')
    st.image('qrcode.jpg')
    if 'show_code' not in st.session_state:
        st.session_state.show_code = False
        # 在侧边栏创建一个复选框来控制代码的显示
    st.session_state.show_code = st.sidebar.checkbox('显示代码', value=st.session_state.show_code)
    if st.session_state.show_code:
        with open(__file__, 'r') as file:
            code = file.read()
        st.markdown("```python\n" + code + "\n```", unsafe_allow_html=True)


elif page_group == '组件测试':
    from 组件测试示例 import show
    show()
    if 'show_code' not in st.session_state:
        st.session_state.show_code = False
        # 在侧边栏创建一个复选框来控制代码的显示
    st.session_state.show_code = st.sidebar.checkbox('显示代码', value=st.session_state.show_code)
    if st.session_state.show_code:
        with open(__file__, 'r') as file:
            code = file.read()
        st.markdown("```python\n" + code + "\n```", unsafe_allow_html=True)

