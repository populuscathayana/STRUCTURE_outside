# By Cathayana
# 介于它很卡，建议单独运行以免卡掉 notebook

import numpy as np
import plotly.graph_objects as go


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







X, Y, Z = np.mgrid[-5:5:40j, -5:5:40j, -5:5:40j]


# Wavefunction combinations for CH4 MOs
psi_s =         psi_2s(X, Y, Z, Z=6) + 0.5 * (psi_1s(X+1, Y+1, Z+1, Z=1) + psi_1s(X+1, Y-1, Z-1, Z=1) + psi_1s(X-1, Y+1, Z-1, Z=1) + psi_1s(X-1, Y-1, Z+1, Z=1))
psi_s_star =    psi_2s(X, Y, Z, Z=6) - 0.5 * (psi_1s(X+1, Y+1, Z+1, Z=1) + psi_1s(X+1, Y-1, Z-1, Z=1) + psi_1s(X-1, Y+1, Z-1, Z=1) + psi_1s(X-1, Y-1, Z+1, Z=1))
psi_px =        psi_2px(X, Y, Z, Z=6) - 0.5 * (psi_1s(X+1, Y+1, Z+1, Z=1) + psi_1s(X+1, Y-1, Z-1, Z=1) - psi_1s(X-1, Y+1, Z-1, Z=1) - psi_1s(X-1, Y-1, Z+1, Z=1))
psi_px_star =   psi_2px(X, Y, Z, Z=6) + 0.5 * (psi_1s(X+1, Y+1, Z+1, Z=1) + psi_1s(X+1, Y-1, Z-1, Z=1) - psi_1s(X-1, Y+1, Z-1, Z=1) - psi_1s(X-1, Y-1, Z+1, Z=1))


# Orbitals list
orbitals = {
    "psi_s": psi_s,
    "psi_s_star": psi_s_star,
    "psi_px": psi_px,
    "psi_px_star": psi_px_star,
}

colorscale = [[0, 'blue'], [0.5, 'white'], [1, 'red']]

# Generating figures for each orbital
for name, orbital in orbitals.items():
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
        title=f"CH4 Molecular Orbital Visualization: {name}",
        margin=dict(t=0, l=0, b=0, r=0)
    )

    fig.show()

