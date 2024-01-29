# hydrogen_molecule_visualization.py
import numpy as np
import plotly.graph_objects as go
from pyscf import gto, scf
import streamlit as st

# 创建氢分子的分子对象
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='6-311g')
mol.build()

# 进行SCF计算
mf = scf.RHF(mol)
mf.kernel()

# 获取轨道能级和系数
orbital_energies = mf.mo_energy
orbital_coefficients = mf.mo_coeff

# Streamlit 应用
def page():
    st.title("Hydrogen Molecule Molecular Orbitals")

    # 显示轨道能级
    st.write("Orbital Energies:")
    st.write(orbital_energies)

    # 绘制轨道能级图
    fig = go.Figure(data=go.Scatter(y=orbital_energies, mode='markers'))
    fig.update_layout(title='Molecular Orbital Energies', xaxis_title='Orbital Index', yaxis_title='Energy (Hartree)')
    st.plotly_chart(fig)

    # 选择要可视化的轨道
    orbital_index = st.slider("Select Orbital Index for Visualization", 0, len(orbital_energies)-1, 0)
    st.write(f"You selected orbital index: {orbital_index}")

    # 计算和绘制电子密度
    # 注意：这里仅为示例，需要根据实际轨道系数进行计算
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X**2 + Y**2) * np.abs(orbital_energies[orbital_index])))

    # 绘制轨道电子密度
    fig = go.Figure(data=go.Contour(z=Z, x=x, y=y, colorscale='Viridis'))
    fig.update_layout(title='Electron Density of Selected Orbital', xaxis_title='X', yaxis_title='Y')
    st.plotly_chart(fig)

if __name__ == "__main__":
    page()
