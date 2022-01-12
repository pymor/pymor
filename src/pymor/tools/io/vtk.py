# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config
config.require('VTKIO')


from pathlib import Path

import meshio
from xml.etree.ElementTree import fromstring
from collections import OrderedDict
from xmljson import BadgerFish
from lxml import etree

from pymor.core.exceptions import IOLibsMissing


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

    Parameters
    ----------
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


def write_vtk_collection(filename_base, meshes, metadata=None):
    """Output grid-associated data in vtk format

    filename_base
        common component for output files in collection
    meshes
        Sequence of meshio.Mesh objects
    metadata
        dict of {key1: sequence1, key2: sequence2} where sequence must be of len(meshes) or len == 1
        currently supported keys are "timestep", "name", "group" and "part"
        used to describe datapoints in Vtk collection file
        defaults to { 'timestep': list(range(len(meshes))) }

    Returns
    -------
    full filename of saved file
    """
    if not config.HAVE_VTKIO:
        raise IOLibsMissing()
    from pyevtk.vtk import VtkGroup

    fn_tpl = '{}_{:08d}.vtu'
    metadata = metadata or {'timestep': list(range(len(meshes)))}

    def _meta(key, i):
        if key in metadata.keys():
            return metadata[key][0] if len(metadata[key]) == 1 else metadata[key][i]
        # carry over defaults from pyevtk to not break backwards compat
        return {'timestep': 0, 'group': '', 'name': '', 'part': '0'}[key]

    group = VtkGroup(filename_base)
    for i, mesh in enumerate(meshes):
        fn = fn_tpl.format(filename_base, i)
        mesh.write(fn)
        group.addFile(filepath=fn, sim_time=_meta('timestep', i), group=_meta('group', i), name=_meta('name', i),
                      part=_meta('part', i))
    group.save()
    return f'{filename_base}.pvd'
