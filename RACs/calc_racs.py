#!/usr/bin/env python3

import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from pymatgen.io.cif import CifParser
from molSimplify.Informatics.MOF.MOF_descriptors import get_MOF_descriptors
import argparse
import subprocess

def get_primitive(datapath, writepath):
    s = CifParser(datapath, occupancy_tolerance=1).get_structures()[0]
    sprim = s.get_primitive_structure()
    sprim.to("cif",writepath)

def run_bash(cmd):
    p = subprocess.Popen(['{}'.format(cmd)],
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = p.communicate()

def main(cif_file):

    cwd = os.getcwd()

    featurization_list = []

    run_bash('mkdir -p {}/primitive'.format(cwd))

    get_primitive(datapath='{}/{}'.format(cwd, cif_file),
                  writepath='{}/primitive/{}'.format(cwd, cif_file))
    
    full_names, full_descriptors = get_MOF_descriptors(data='{}/primitive/{}'.format(cwd, cif_file),
                                                       depth=3, path=cwd,
                                                       xyzpath='{}/xyz/{}'.format(cwd,
                                                                cif_file.replace('.cif','.xyz')))
    full_names.append('filename')
    full_descriptors.append(cif_file)
    featurization = dict(zip(full_names, full_descriptors))
    featurization_list.append(featurization)

    df = pd.DataFrame(featurization_list)

    return df

if __name__ == '__main__':

    ap = argparse.ArgumentParser(
         description = "Calculate RAC descriptors for a MOF")
    ap.add_argument("cif", metavar="f", type=str,
                    help="CIF file to run check on.")

    args = ap.parse_args()
    filename = args.cif

    results = main(filename)
    print(results)
    results.to_csv('{}/{}.csv'.format(os.getcwd(), filename))

