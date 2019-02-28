from pathlib import Path
from xml.etree.ElementTree import fromstring
from collections import OrderedDict
from xmljson import BadgerFish
import vtk


def _read_collection(xml):
    collection = xml['VTKFile']['Collection']
    files = collection['DataSet']
    data = [_read_single(f['@file']) for f in files]
    return data[0]


def _read_single(type, path):
    if type == 'UnstructuredGrid':
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise NotImplementedError('f"VTK Files of type {type} can not yet be processed"')
    reader.SetFileName(path)
    reader.Update()

    geometryFilter = vtk.vtkGeometryFilter()
    geometryFilter.SetInputData(reader.GetOutput())
    geometryFilter.Update()

    return geometryFilter.GetOutput()


def read_vtkfile(filename):
    path = Path(filename)
    assert path.is_file()
    bf = BadgerFish(dict_type=OrderedDict)

    xml = bf.data(fromstring(open(path, 'rt').read()))
    try:
        type = xml['VTKFile']['@type']
    except KeyError:
        type = None
    if type == 'Collection':
        return _read_collection(xml, path)
    return _read_single(type, path)
