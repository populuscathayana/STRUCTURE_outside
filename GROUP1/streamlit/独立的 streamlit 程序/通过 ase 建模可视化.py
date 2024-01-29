from ase.spacegroup import crystal
from ase.build import make_supercell
import plotly.graph_objects as go
import numpy as np
import streamlit as st

# Define the crystal structure of NaCl
a = 5.64  # Lattice parameter for NaCl in Angstroms
nacl = crystal(['Na', 'Cl'], basis=[(0, 0, 0), (0.5, 0.5, 0.5)], spacegroup=225,
                cellpar=[a, a, a, 90, 90, 90])

# Define a supercell to extend the structure
supercell_matrix = np.identity(3) * 2  # Adjust the size as needed
nacl_supercell = make_supercell(nacl, supercell_matrix)

# Extract positions for visualization
positions = nacl_supercell.get_positions()
symbols = nacl_supercell.get_chemical_symbols()

# Create Plotly figure
fig = go.Figure(data=[go.Scatter3d(
    x=positions[:, 0],
    y=positions[:, 1],
    z=positions[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=['blue' if symbol == 'Na' else 'green' for symbol in symbols],  # Color Na blue, Cl green
        opacity=0.8
    )
)])

# Update layout
fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(
        xaxis_title='X (Å)',
        yaxis_title='Y (Å)',
        zaxis_title='Z (Å)'
    )
)

# Streamlit integration (optional)
st.title("NaCl Crystal Structure Visualization")
st.plotly_chart(fig)
