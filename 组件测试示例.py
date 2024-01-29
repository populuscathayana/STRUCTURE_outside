


def show():
    import streamlit as st
    import numpy as np
    import pandas as pd
    import time
    import altair as alt
    import numpy as np
    import matplotlib.pyplot as plt
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title('This is a title')

    plt.style.use('default')
    h = 6.626*10**(-34)
    v = np.linspace(0.001,4,4000)
    k = 1.38*10**(-23)
    c = 3*10**8
    def E(v, T):
        return 10**(9)*2*np.pi*h*(v*10**14)**3/(np.exp(h*v*10**14/k/T)-1)/c**2
    fig, ax = plt.subplots()
    plt.xlim((0, 4))
    plt.ylim((0, 5))
    plt.xlabel('v/(10$^{14}$s$^{-1}$)')
    plt.ylabel('E$_v$/(10$^{-9}$J/m$^{-2}$)')
    ax.set_title("Energy distribution curve of black body at different temperatures")
    ax.plot(v, E(v, 1000), label = '1000K')
    ax.plot(v, E(v, 1500), label = '1500K')
    ax.plot(v, E(v, 2000), label = '2000K')
    ax.legend()
    #plt.show()

    st.markdown('使用matplotlib，支持 tex 表达式')
    st.pyplot(fig)



    import streamlit as st
    import numpy as np
    import plotly.graph_objects as go
    # Define constants
    h = 6.626 * 10 ** (-34)
    k = 1.38 * 10 ** (-23)
    c = 3 * 10 ** 8
    # Define the function for energy distribution
    def E(v, T):
        return 10 ** 9 * 2 * np.pi * h * (v * 10 ** 14) ** 3 / (np.exp(h * v * 10 ** 14 / k / T) - 1) / c ** 2
    # Create the plot
    v = np.linspace(0.001, 4, 4000)
    fig = go.Figure()
    # Update plot layout
    fig.update_layout(
        height=600,  # Adjust the height of the plot
        width=800,   # Adjust the width of the plot
    )


    fig.add_trace(go.Scatter(x=v, y=E(v, 1000), mode='lines', name='1000K', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=v, y=E(v, 1500), mode='lines', name='1500K', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=v, y=E(v, 2000), mode='lines', name='2000K', line=dict(color='green')))

    # Update plot layout
    fig.update_layout(
        title="Energy distribution curve of black body at different temperatures",
        xaxis_title='$v/(10^{14}s^{-1})$',
        yaxis_title='$E_v/(10^{-9}J/m^{-2})$',
        xaxis=dict(range=[0, 4]),
        yaxis=dict(range=[0, 5]),
        legend=dict(
            title='Temperature',
            orientation='v',
            yanchor="bottom",
            y=0.8,
            xanchor="right",
            x=0.9
        )
    )

    st.markdown('使用html展示方式plotly 可以使用 tex 表达式')
    st.components.v1.html(fig.to_html(include_plotlyjs='cdn',include_mathjax='cdn'),height=600)
    st.write('用st.write(fig)也无法保留 tex 表达式:')
    st.write(fig)


    st.markdown('st.plotly_chart(fig) 也同样无法使用 tex 表达式')
    # Display the plot in Streamlit
    st.plotly_chart(fig)










    # 主副标题指令


    # 展示源代码指令
    with st.echo():
        st.header('This is a header')
        st.subheader('This is a subheader')

    ########################################### 普通write指令的不同形式
    # 标题指令
    st.write("尝试一个表格")
    st.write('Hello, *World!*')
    st.write(1234)
    st.write(pd.DataFrame({
        'first column': [1, 2, 3, 4],
        'second column': [10, 20, 30, 40],
    }))
    st.write('1 + 1 = ', 2)
    #write表格
    df1 = pd.DataFrame({
        '第一列':[1,2,3,4],
        '第二列':['a','b','c','d']})
    st.write('Below is a DataFrame:', df1, 'Above is a dataframe.')
    #write图表
    
    df3 = pd.DataFrame(
        np.random.randn(200, 3),
        columns=['a', 'b', 'c'])
    c = alt.Chart(df3).mark_circle().encode(
        x='a', y='b', size='c', color='c')
    st.write('alt.chart()',c)

    ###########################################

    # 普通md指令
    st.markdown('Streamlit is **_really_ cool**.')

    # 普通代码指令
    code = '''def hello():
    ...     print("Hello, Streamlit!")'''
    st.write('st.code()')
    st.code(code, language='python')

    #用table功能制作表格
    st.write("st.table(pd.dataframe(...))表格")
    df2 = pd.DataFrame({
        '第一列':[1,2,3,4],
        '第二列':['a','b','c','d']})
    st.table(df2)
    
    st.write('st.line_chart(pd.DataFrame(...))')
    #用chart功能构建一个可交互的图表
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['a', 'b', 'c'])
    st.line_chart(chart_data)

    st.write('st.map(map_data)')
    #用map功能来制作地图相关的内容
    map_data = pd.DataFrame(
        np.random.randn(1, 1) / [200, 200] + [30.30270122354777, 120.08210320964814],  #N,W
        columns=['lat', 'lon'])
    st.map(map_data)

    st.write('st.dataframe(df)')
    # dataframe 可交互数据帧
    df = pd.DataFrame(
        np.random.randn(50, 20),
        columns=('col %d' % i for i in range(20)))
    st.dataframe(df)  # Same as st.write(df)

    # 漂亮的plotly
    st.write('plotly绘图及筛选')
    import plotly.express as px
    # 创建一个简单的散点图
    df = px.data.iris()
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
    st.write(df)
    st.plotly_chart(fig)
    colors = {
    1: 'red',  # 物种ID为0的颜色
    2: 'green', # 物种ID为1的颜色
    3: 'blue'  # 物种ID为2的颜色
    }
    st.write('修改图的各个参数')
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Scatter(x=df['sepal_width'], y=df['sepal_length'], mode='markers', marker_color=[colors[id] for id in df['species_id']]))
    fig.update_layout(title='Sepal Dimensions', xaxis_title='Width', yaxis_title='Length')
    st.plotly_chart(fig)
    st.write('进行筛选')
    species = st.selectbox("Choose a species", df['species'].unique())
    filtered_df = df[df['species'] == species]
    fig = px.scatter(filtered_df, x="sepal_width", y="sepal_length",color_discrete_sequence=[colors[id] for id in filtered_df['species_id']])
    st.plotly_chart(fig)

    st.write('3维作图')
    fig = px.scatter_3d(df, x="sepal_width", y="sepal_length", z="petal_length", color="species")
    st.plotly_chart(fig)

    # 用于绘制地图的pydeck

    地图定位展示 = '''
    import pydeck as pdk
    import pandas as pd
    #由于服务器网络难连接到URL地址故注释掉了
    st.write('定位个海纳苑')
    # 从 URL 中加载数据集
    DATA_URL = "https://raw.githubusercontent.com/ajduberstein/geo_datasets/master/housing.csv"
    data = pd.read_csv(DATA_URL)
    # 绘制地图
    layer = pdk.Layer(
        "HexagonLayer",
        data=data,
        get_position=["longitude", "latitude"],
        auto_highlight=True,
        elevation_scale=50,
        pickable=True,
        elevation_range=[0, 3000],
        extruded=True,
        coverage=1,
    )

    # 设置视图状态
    view_state = pdk.ViewState(
        longitude=120.07013,
        latitude=30.301668,
        zoom=16,
        min_zoom=5,
        max_zoom=16,
        pitch=40.5,
        bearing=-27.36,
    )

    # 绘制
    r = pdk.Deck(layers=[layer], initial_view_state=view_state)
    st.pydeck_chart(r)    
    '''
    



    # 用于绘制图表的altair
    import altair as alt
    import pandas as pd
    import numpy as np
    st.write('altair绘图（好像前面有过了，懒得删了）')
    # 用于绘制图表的数据
    df = pd.DataFrame(
        np.random.randn(200, 3),
        columns=['a', 'b', 'c'])

    c = alt.Chart(df).mark_circle().encode(
        x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])

    st.write(c)

    # 用于绘制图表的matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    plt.clf()
    st.write('matplotlib绘图')
    arr = np.random.normal(1, 1, size=100)
    plt.hist(arr, bins=20)

    st.pyplot()

    st.write('seaborn+matplotlib绘图')
    # 用于绘制图表的seaborn
    import seaborn as sns
    import pandas as pd
    import numpy as np
    plt.clf()
    df = pd.DataFrame(
        np.random.randn(200, 3),
        columns=['a', 'b', 'c'])

    sns.violinplot(data=df)

    st.pyplot()

    # 用于绘制图表的plotly
    import plotly.figure_factory as ff
    import numpy as np
    st.write('plotly.figure_factory 绘图')
    # 用于绘制图表的数据
    x1 = np.random.randn(200) - 2
    x2 = np.random.randn(200)
    x3 = np.random.randn(200) + 2
    x4 = np.random.randn(200) + 4

    # 创建直方图
    hist_data = [x1, x2, x3, x4]
    group_labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4']

    # 创建直方图
    fig = ff.create_distplot(
        hist_data, group_labels, bin_size=[.1, .25, .5, 1])

    # 绘制
    st.plotly_chart(fig, use_container_width=True)
    
    
    
    # 用于绘制图表的bokeh
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource
    from bokeh.palettes import Spectral6
    from bokeh.transform import factor_cmap
    st.write('bokeh绘图')
    # 用于绘制图表的数据
    fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
    counts = [5, 3, 4, 2, 4, 6]

    source = ColumnDataSource(data=dict(fruits=fruits, counts=counts))

    p = figure(x_range=fruits, toolbar_location=None, title="Fruit Counts")

    p.vbar(x='fruits', top='counts', width=0.9, source=source, legend_field="fruits",
            line_color='white', fill_color=factor_cmap('fruits', palette=Spectral6, factors=fruits))

    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.y_range.end = 9
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"

    # 绘制
    st.bokeh_chart(p, use_container_width=True)



    # 用于绘制图表的vega_lite
    # 用于绘制图表的数据
    import pandas as pd
    import numpy as np
    st.write('vega_lite绘制散点图')
    df = pd.DataFrame(
        np.random.randn(200, 3),
        columns=['a', 'b', 'c'])

    # 创建散点图
    st.vega_lite_chart(df, {
        'mark': {'type': 'circle', 'tooltip': True},
        'encoding': {
            'x': {'field': 'a', 'type': 'quantitative'},
            'y': {'field': 'b', 'type': 'quantitative'},
            'size': {'field': 'c', 'type': 'quantitative'},
            'color': {'field': 'c', 'type': 'quantitative'},
        },
    })
    

    
    # 加入额外页面
    st.sidebar.write('联系我')
    st.sidebar.image('qrcode.jpg')
    st.sidebar.markdown('## This is a sidebar')
    st.sidebar.markdown('### Another sidebar')
    #st.markdown('## This is a sidebar')
    #st.markdown('### Another sidebar')

    # 在sidebar中加入交互式小部件
    # Add a selectbox to the sidebar:
    st.sidebar.write('加入意义不明的selectbox')
    
    add_selectbox = st.sidebar.selectbox(
        'How to contact me?',
        ('Email', 'Home phone', 'Mobile phone')
    )
    if add_selectbox == 'Email' :
        st.sidebar.write(add_selectbox)

    # Add a slider to the sidebar:
    add_slider = st.sidebar.slider(
        'Select a range of values',
        0.0, 100.0, (25.0, 75.0)
    )
    st.sidebar.write(add_slider)

    # 记录访客的ip地址
    import streamlit as st
    import requests

    st.write('获取客户端地址')
    def get_client_ip():
        # 获取客户端的IP地址
        ip = requests.get('https://api.ipify.org').text
        return ip

    # 在Streamlit应用中显示访客的IP地址
    ip_address = get_client_ip()
    st.write("服务器的IP地址：", ip_address)