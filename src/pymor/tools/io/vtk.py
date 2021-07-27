# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pathlib import Path

import meshio
from xml.etree.ElementTree import fromstring
from collections import OrderedDict
from xmljson import BadgerFish
from lxml import etree


def _read_collection(xml, metadata_key):
    collection = xml['VTKFile']['Collection']
    files = collection['DataSet']
    data = [(fl[f'@{metadata_key}'], _read_single(fl['@file'])) for fl in files]
    data.sort(key=lambda t: t[0])
    return data


def _read_single(filename):
    mesh = meshio.read(filename)
    assert len(mesh.points)
    return mesh


def _get_collection_data(filename):
    path = Path(filename)
    assert path.is_file()
    bf = BadgerFish(dict_type=OrderedDict)
    return path, bf.data(fromstring(open(path, 'rb').read()))


def _get_vtk_type(path):
    """Parse given file until a VTKFile element is found.

    We use the incremental event emitting parser here since we can expect to encounter appended
    binary data in the xml which lxml cannot parse.

    Parameters:
    -----------
        path
            vtk file to peek into

    Returns
    -------
    None if no VTKFile element found, else the type attribute of the VTKFile element
    """
    parser = etree.XMLPullParser(events=('start',))
    with open(path, 'rb') as xml:
        for lines in xml.readlines():
            parser.feed(lines)
            for action, element in parser.read_events():
                if element.tag == 'VTKFile':
                    return element.get('type')
    return None


def read_vtkfile(filename, metadata_key='timestep'):
    """Try to read a given file into a Sequence of meshio.Mesh instances

    Parameters
    ----------
    metadata_key
        Which metadata to extract and return alongside the meshio.Mesh instances.

    Returns
    -------
    A list of (metadata_value, meshio.Mesh) tuples. The length of the list is either 1 for
        a singular vtk/vtu/vtp input file (None is returned as metadata),
        or however many members are in the collection file (pvd).
    """
    from pymor.tools.io import change_to_directory
    vtk_type = _get_vtk_type(filename)
    if vtk_type == 'Collection':
        path, xml = _get_collection_data(filename)
        with change_to_directory(path.parent):
            return _read_collection(xml, metadata_key=metadata_key)
    return [(None, _read_single(filename, vtk_type))]
