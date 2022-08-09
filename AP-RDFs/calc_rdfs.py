#!/usr/bin/env python3

import numpy as np
from glob import glob
from CifFile import ReadCif
from itertools import product, combinations, combinations_with_replacement
import math
from atomic_property_dict import apd
from datetime import datetime
import os
import argparse

# By default, save the csv in the current working directory
script_path, script_name = os.path.split(os.path.realpath(__file__))
cwd = os.getcwd()
csv_path = '{}/RDFs.csv'.format(cwd)

# Desired distance bins (in this case with linear increase in bin size from 2 to 30 A)
bins = np.arange(113, dtype=np.float64)
bins[0] = 2.0
step = 0.004425

for i in range(1, 113):
    bins[i] = bins[i - 1] + step
    step += 0.004425

n_bins = len(bins)


def main(name, prop_names, smooth, factor):
    mof = ReadCif(name)
    mof = mof[mof.visible_keys[0]]

    elements = mof["_atom_site_type_symbol"]
    n_atoms = len(elements)

    super_cell = np.array(list(product([-1, 0, 1], repeat=3)), dtype=float)

    # Make sure the property exists for the particular element
    prop_list = [apd[name] for name in prop_names]
    for atom in set(elements):
        for count, prop in enumerate(prop_list):
            if atom not in prop:
                print( "\n***WARNING***",
                       "Property '{}' does not exist for atom {}!".format(prop_names[count], atom),
                       "RDFs for this property will be skipped.")
                prop_list.remove(prop)
                prop_names.remove(prop_names[count])

    n_props = len(prop_names)
    prop_dict = {}


    for a1, a2 in combinations_with_replacement(set(elements), 2):
        prop_arr = [prop[a1] * prop[a2] for prop in prop_list]
        prop_dict[(a1, a2)] = prop_arr
        if a1 != a2:
            prop_dict[(a2, a1)] = prop_arr

    la = float(mof["_cell_length_a"])
    lb = float(mof["_cell_length_b"])
    lc = float(mof["_cell_length_c"])
    aa = np.deg2rad(float(mof["_cell_angle_alpha"]))
    ab = np.deg2rad(float(mof["_cell_angle_beta"]))
    ag = np.deg2rad(float(mof["_cell_angle_gamma"]))
    # If volume is missing from .cif, calculate it.
    try:
        cv = float(mof["_cell_volume"])
    except KeyError:
        cv = la * lb * lc * math.sqrt(1 - (math.cos(aa)) ** 2 -
                (math.cos(ab)) ** 2 - (math.cos(ag)) ** 2 +
                (2 * math.cos(aa) * math.cos(ab) * math.cos(ag)))


    frac2cart = np.zeros([3, 3], dtype=float)
    frac2cart[0, 0] = la
    frac2cart[0, 1] = lb * np.cos(ag)
    frac2cart[0, 2] = lc * np.cos(ab)
    frac2cart[1, 1] = lb * np.sin(ag)
    frac2cart[1, 2] = lc * (np.cos(aa) - np.cos(ab)*np.cos(ag)) / np.sin(ag)
    frac2cart[2, 2] = cv / (la * lb * np.sin(ag))

    frac = np.array([
        mof["_atom_site_fract_x"],
        mof["_atom_site_fract_y"],
        mof["_atom_site_fract_z"],
    ], dtype=float).T

    apw_rdf = np.zeros([n_props, n_bins], dtype=np.float64)
    for i, j in combinations(range(n_atoms), 2):
        cart_i = frac2cart @ frac[i]
        cart_j = (frac2cart @ (super_cell + frac[j]).T).T
        dist_ij = min(np.linalg.norm(cart_j - cart_i, axis=1))
        rdf = np.exp(smooth * (bins - dist_ij) ** 2)
        rdf = rdf.repeat(n_props).reshape(n_bins, n_props)
        apw_rdf += (rdf * prop_dict[(elements[i], elements[j])]).T
    apw_rdf = np.round(apw_rdf.flatten() * factor / n_atoms, decimals=12)


    # Temporary fix to exclude properties which were skipped from being written
    # to the csv file. Should ideally not include this in the main() function...
    csv_header = [f"RDF_{prop}_{r:.3f}" for prop in props for r in bins]
    csv_header.insert(0, "Structure_Name")

    return ("{}," * len(apw_rdf) + "{}\n").format(
        name.split('/')[-1], *apw_rdf.tolist()), csv_header

if __name__ == "__main__":

    ap = argparse.ArgumentParser(
         description = "Calculate RDF descriptors for a MOF")
    ap.add_argument("cif", metavar="f", type=str,
                    help="CIF file for which to compute RDFs.")
    ap.add_argument("--props", nargs='+',
                    default=['all'],
                    help="List of properties to calculate. Default is 'all'.\
                          Choices are:\
                          'electronegativity',\
                          'hardness',\
                          'vdWaalsVolume',\
                          'polarizability',\
                          'mass',\
                          'none',\
                          'all'")
    ap.add_argument("--B", type=float, default=-10,
                    help="Smoothing parameter, default is -10.")
    ap.add_argument("--f", type=float, default=0.001,
                    help="Scaling factor, default is 0.001.")

    args = ap.parse_args()
    filename = args.cif
    props = args.props
    if any('all' in prop for prop in props):
        props = list(apd.keys())
    B = args.B
    f = args.f

    start = datetime.now()
    print("====================================================================")
    print("Start: ", start.strftime("%c"))
    print("")
    print("")
    print("Starting RDF calculation on {}".format(filename))
    print("RDFs will be written to: {}".format(csv_path))

    with open(csv_path, 'w') as csv:
        result, csv_header = main(filename, props, B, f)
        csv.write(','.join(csv_header) + '\n')
        csv.flush()
        csv.write(result)
        csv.flush()

    print("")
    print("")
    print("Finished! RDFs have been saved to: {}".format(csv_path))
    end = datetime.now()
    print("End: ", end.strftime("%c"))
    print("====================================================================")
