import streamlit as st
import numpy as np
from rdkit import Chem
from scipy.linalg import det, eig,eigh
from streamlit_ketcher import st_ketcher
from rdkit.Chem import Draw, AllChem,rdmolops
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem import rdChemReactions
from rdkit.Chem import rdchem
import pandas as pd
import sympy as sp
from scipy.optimize import fsolve

'''
    By Cathayana

    本项目的主要重点在正交归一化矩阵，在很多同学的作业计算中得到的矩阵是没有正交化的。

    其余的主要是符号非精确求解，多重根的识别取舍，以及录入 Ketcher 的 MOl 文件的问题。

    由于进行后续轨道系数的绘图和本部分实际上是独立的，将矩阵内容和图中对应的原子通过某种方式链接需要写一套额外的算法。且进行 3d 绘图和标注不一定有系数直观，故仅仅给出轨道系数。

    原子和对应位置可通过得到的 Huckel 矩阵自行推得。

'''

def parse_molfile(molfile_content):
    """
    解析 MOL 文件，构建休克尔矩阵。
    """
    molecule = Chem.MolFromMolBlock(molfile_content)
    atoms = molecule.GetAtoms()
    bonds = molecule.GetBonds()

    num_carbon_atoms = sum(1 for atom in atoms if atom.GetSymbol() == 'C')
    huckel_matrix = sp.zeros(num_carbon_atoms, num_carbon_atoms)

    for bond in bonds:
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()

        if begin_atom.GetSymbol() == 'C' and end_atom.GetSymbol() == 'C':
            begin_idx, end_idx = begin_atom.GetIdx(), end_atom.GetIdx()
            huckel_matrix[begin_idx, end_idx] = huckel_matrix[end_idx, begin_idx] = 1

    for i in range(num_carbon_atoms):
        huckel_matrix[i, i] = sp.symbols('x')

    return huckel_matrix

def solve_for_x(huckel_matrix, x_range=(-10, 10), threshold=1e-6, xtol=1e-10, merge_tol=1e-3):
    """
    使用数值方法求解行列式为零时的 x 值，并合并接近的解。
    :param huckel_matrix: 赫克尔矩阵
    :param x_range: x 的数值搜索范围，例如 (-10, 10)
    :param threshold: 检查数值解的有效性阈值
    :param xtol: 数值方法的求解精度
    :param merge_tol: 合并接近解的阈值
    :return: 行列式为零时的 x 值列表
    """
    x = sp.symbols('x')
    det_huckel = huckel_matrix.det()

    # 将符号表达式转换为数值函数
    det_func = sp.lambdify(x, det_huckel, 'numpy')

    # 在指定范围内搜索零点
    roots = []
    for val in np.linspace(x_range[0], x_range[1], 5000):
        root, = fsolve(det_func, val, xtol=xtol)
        if np.isclose(det_func(root), 0, atol=threshold):
            # 合并接近的解
            if not any(np.isclose(root, existing_root, atol=merge_tol) for existing_root in roots):
                roots.append(root)

    # 对找到的根进行排序和合并
    roots.sort()
    merged_roots = []
    current_group = [roots[0]]

    for root in roots[1:]:
        if np.isclose(root, current_group[-1], atol=merge_tol):
            current_group.append(root)
        else:
            merged_roots.append(np.mean(current_group))
            current_group = [root]

    merged_roots.append(np.mean(current_group))  # 添加最后一组的平均值

    return merged_roots

def is_vector_close(v1, v2, tol=1e-5):
    """
    检查两个向量是否足够接近。
    :param v1, v2: 待比较的两个向量
    :param tol: 接近的阈值
    :return: 如果接近则为True，否则为False
    """
    return np.linalg.norm(v1 - v2) < tol or np.linalg.norm(v1 + v2) < tol

def calculate_orbital_coefficients(huckel_matrix, x_values, merge_tol=1e-3):
    """
    对于每个解x，计算并归一化对应的轨道系数，并合并接近的特征向量。
    :param huckel_matrix: 赫克尔矩阵（包含符号变量x）
    :param x_values: 一系列求解得到的x值
    :param merge_tol: 合并接近特征向量的阈值
    :return: 对于每个x值的归一化轨道系数列表
    """
    orbital_coefficients = []

    for x_val in x_values:
        # 代入x值得到数值矩阵
        num_matrix = np.array(huckel_matrix.subs(sp.symbols('x'), x_val).evalf()).astype(np.float64)

        # 计算特征值和特征向量
        eigenvalues, eigenvectors = eigh(num_matrix)

        # 归一化处理特征向量并合并接近的向量
        for eigenvector in eigenvectors.T:
            normalized_vect = eigenvector / np.linalg.norm(eigenvector)
            if not any(is_vector_close(normalized_vect, existing_vect, tol=merge_tol) for existing_vect in orbital_coefficients):
                orbital_coefficients.append(normalized_vect)

    return orbital_coefficients

def match_orbitals_to_x_values(orbital_coefficients, huckel_matrix, x_values, threshold=1e-5):
    """
    反推轨道系数对应的 x 值。
    """
    x = sp.symbols('x')
    orbital_x_matches = {}

    for coeff in orbital_coefficients:
        test_vect = sp.Matrix(coeff)
        for x_val in x_values:
            substituted_matrix = huckel_matrix.subs(x, x_val)
            res = substituted_matrix * test_vect
            if all(sp.Abs(elem) < threshold for elem in res):
                orbital_x_matches[tuple(coeff)] = x_val
                break

    return orbital_x_matches


def format_vector(vect, tol=1e-4, precision=4):
    """
    格式化向量，将小于tol的元素设为0，其余元素保留指定小数位数。
    """
    formatted_vect = np.where(np.abs(vect) < tol, 0, np.round(vect, precision))
    return formatted_vect


def main():
    st.title("Hückel Molecular Orbital Theory Calculator")
    molecule = st_ketcher(key="molecule", molecule_format="MOLFILE")
    if molecule:
        huckel_matrix = parse_molfile(molecule)
        #st.text(huckel_matrix)
        x_values = solve_for_x(huckel_matrix)
        alpha = sp.symbols('α')
        beta = sp.symbols('β')
        energies = [sp.simplify(alpha - beta * round(float(x), 4) if isinstance(x, (float, int)) else x) for x in x_values]
        st.markdown(f"<h4 style='text-align: center;'>Hückel Matrix:</h4>", unsafe_allow_html=True)
        st.text(sp.pretty(huckel_matrix))
        st.markdown(f"<h4 style='text-align: center;'>x Values and Corresponding Energies (E):</h4>", unsafe_allow_html=True)
        for x_val, energy in zip(x_values, energies):
            # 格式化 x 值为保留四位小数
            formatted_x_val = round(float(x_val), 4) if isinstance(x_val, (float, int)) else x_val
            formatted_energy=round(float(energy), 4) if isinstance(energy, (float, int)) else energy
            st.write(f"x = {formatted_x_val}, E = {formatted_energy}")
        # 展示轨道系数结果
        orbital_coefficients = calculate_orbital_coefficients(huckel_matrix,format_vector(x_values))
        orbital_x_matches = match_orbitals_to_x_values(orbital_coefficients, huckel_matrix, x_values)
        st.markdown(f"<h4 style='text-align: center;'>Orbital Coefficients</h4>", unsafe_allow_html=True)
        for coeff, x_val in orbital_x_matches.items():
            st.write(f"Orbital Coefficients for x = {round(float(x_val), 4)}:")
            formatted_coeff = format_vector(coeff)
            st.text(np.array2string(formatted_coeff))
    if st.button('Show Code'):
        with open(__file__, 'r') as file:
            code = file.read()
        st.markdown("```python\n" + code + "\n```", unsafe_allow_html=True)


if __name__ == "__main__":
    main()