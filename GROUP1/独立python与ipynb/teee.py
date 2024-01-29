import numpy as np
import plotly.graph_objects as go

# By Cathayana

import plotly.graph_objects as go

# Creating the tree diagram with centered labels
fig = go.Figure(go.Treemap(
    labels=[
        "Molecular Geometry",
        "Other Situation",
        "Electron Pairs = 1","直线型",
        "Electron Pairs = 2"," 直线型",
        "Electron Pairs = 3", "Atom Count = 3<br>(折线型)", "Atom Count = 4<br>(平面三角型)",
        "Electron Pairs = 4", "Atom Count = 3<br> (折线型)", "Atom Count = 4<br>(三角锥型)", "Atom Count = 5<br> (四面体型)",
        "Electron Pairs = 5", "Atom Count = 3<br>(直线型)", "Atom Count = 4<br>(T型)", "Atom Count = 5<br> (变形四面体型)", "Atom Count = 6<br> (三角双锥型)",
        "Electron Pairs = 6", "Atom Count = 3<br> (直线型)", "Atom Count = 4<br> (T型)", "Atom Count = 5<br> (平面正方型)", "Atom Count = 6<br> (四方锥型)", "Atom Count = 7<br> (八面体型)"
    ],
    parents=[
        "",
        "Molecular Geometry",
        "Molecular Geometry", "Electron Pairs = 1",
        "Molecular Geometry", "Electron Pairs = 2",
        "Molecular Geometry", "Electron Pairs = 3", "Electron Pairs = 3",
        "Molecular Geometry", "Electron Pairs = 4", "Electron Pairs = 4", "Electron Pairs = 4",
        "Molecular Geometry", "Electron Pairs = 5", "Electron Pairs = 5", "Electron Pairs = 5", "Electron Pairs = 5",
        "Molecular Geometry", "Electron Pairs = 6", "Electron Pairs = 6", "Electron Pairs = 6", "Electron Pairs = 6", "Electron Pairs = 6"
    ],
    textinfo="label",
    textposition="middle center"
))

fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))

fig.show()

# 由于 plotly 的 Treemap 暂时不支持图片插入，所以没有加入对应分子的演示图片


a_0 = 1


def psi_1s(x, y, z, Z):
    r = np.sqrt(x**2 + y**2 + z**2)
    normalization_factor = (Z**3 / (np.pi * a_0**3))**0.5
    return normalization_factor * np.exp(-Z * r / a_0)

def psi_2s(x, y, z, Z):
    r = np.sqrt(x**2 + y**2 + z**2)
    normalization_factor = (Z**3 / (8 * np.pi * a_0**3))**0.5
    return normalization_factor * (1 - Z * r / (2 * a_0)) * np.exp(-Z * r / (2 * a_0))

def psi_2px(x, y, z, Z):
    r = np.sqrt(x**2 + y**2 + z**2)
    normalization_factor = (Z**5 / (32 * np.pi * a_0**5))**0.5
    return normalization_factor * (x / r) * np.exp(-Z * r / (2 * a_0))

def psi_2py(x, y, z, Z):
    r = np.sqrt(x**2 + y**2 + z**2)
    normalization_factor = (Z**5 / (32 * np.pi * a_0**5))**0.5
    return normalization_factor * (y / r) * np.exp(-Z * r / (2 * a_0))

def psi_2pz(x, y, z, Z):
    r = np.sqrt(x**2 + y**2 + z**2)
    normalization_factor = (Z**5 / (32 * np.pi * a_0**5))**0.5
    return normalization_factor * (z / r) * np.exp(-Z * r / (2 * a_0))





X, Y, Z = np.mgrid[-5:5:100j, -5:5:100j, -5:5:100j]


# Wavefunction combinations for CH4 MOs
psi_s =         psi_2s(X, Y, Z, Z=6) + 0.5 * (psi_1s(X+1, Y+1, Z+1, Z=1) + psi_1s(X+1, Y-1, Z-1, Z=1) + psi_1s(X-1, Y+1, Z-1, Z=1) + psi_1s(X-1, Y-1, Z+1, Z=1))
psi_s_star =    psi_2s(X, Y, Z, Z=6) - 0.5 * (psi_1s(X+1, Y+1, Z+1, Z=1) + psi_1s(X+1, Y-1, Z-1, Z=1) + psi_1s(X-1, Y+1, Z-1, Z=1) + psi_1s(X-1, Y-1, Z+1, Z=1))
psi_px =        psi_2px(X, Y, Z, Z=6) + 0 * (psi_1s(X+1, Y+1, Z+1, Z=1) + psi_1s(X+1, Y-1, Z-1, Z=1) - psi_1s(X-1, Y+1, Z-1, Z=1) - psi_1s(X-1, Y-1, Z+1, Z=1))
psi_px_star =   psi_2px(X, Y, Z, Z=6) - 0.5 * (psi_1s(X+1, Y+1, Z+1, Z=1) + psi_1s(X+1, Y-1, Z-1, Z=1) - psi_1s(X-1, Y+1, Z-1, Z=1) - psi_1s(X-1, Y-1, Z+1, Z=1))
psi_py =        psi_2py(X, Y, Z, Z=6) + 0.5 * (psi_1s(X+1, Y+1, Z+1, Z=1) - psi_1s(X+1, Y-1, Z-1, Z=1) - psi_1s(X-1, Y+1, Z-1, Z=1) + psi_1s(X-1, Y-1, Z+1, Z=1))
psi_py_star =   psi_2py(X, Y, Z, Z=6) - 0.5 * (psi_1s(X+1, Y+1, Z+1, Z=1) - psi_1s(X+1, Y-1, Z-1, Z=1) - psi_1s(X-1, Y+1, Z-1, Z=1) + psi_1s(X-1, Y-1, Z+1, Z=1))
psi_pz =        psi_2pz(X, Y, Z, Z=6) + 0.5 * (psi_1s(X+1, Y+1, Z+1, Z=1) - psi_1s(X+1, Y-1, Z-1, Z=1) + psi_1s(X-1, Y+1, Z-1, Z=1) - psi_1s(X-1, Y-1, Z+1, Z=1))
psi_pz_star =   psi_2pz(X, Y, Z, Z=6) - 0.5 * (psi_1s(X+1, Y+1, Z+1, Z=1) - psi_1s(X+1, Y-1, Z-1, Z=1) + psi_1s(X-1, Y+1, Z-1, Z=1) - psi_1s(X-1, Y-1, Z+1, Z=1))


orbitals= [psi_s, psi_s_star, psi_px, psi_px_star, psi_py, psi_py_star, psi_pz, psi_pz_star]
colorscale = [[0, 'blue'], [0.5, 'white'], [1, 'red']]
for orbital in orbitals:
    fig = go.Figure()

    fig.add_trace(go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=orbital.flatten(),
        isomin=-0.1,
        isomax=0.1,
        opacity=0.1,
        surface_count=20,
        colorscale=colorscale
    ))

    fig.update_layout(
        title="CH4 Molecular Orbitals Visualization",
        margin=dict(t=0, l=0, b=0, r=0)
    )

    fig.show()

