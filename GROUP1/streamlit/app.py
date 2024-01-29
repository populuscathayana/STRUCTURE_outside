
# app.py


import streamlit as st
import hydrogen_atom as hydrogen_atom, hydrogen_molecule, others, hydrogen_atom_2d

PAGES = {
    "Hydrogen Atom": hydrogen_atom,
    "Hydrogen Atom_2d": hydrogen_atom_2d,
    "Hydrogen Molecule": hydrogen_molecule,
    "Others": others
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Choose Visualization", list(PAGES.keys()))
page = PAGES[selection]
page.page()
