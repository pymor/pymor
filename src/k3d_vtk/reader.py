from pathlib import Path
from xml.etree.ElementTree import fromstring
from collections import OrderedDict
from xmljson import BadgerFish
import vtk
from lxml import etree


def _read_collection(xml):
    collection = xml['VTKFile']['Collection']
    files = collection['DataSet']
    data = [(f['@timestep'],_read_single(f['@file'])) for f in files]
    data.sort(key = lambda t: t[0])
    return data


def _read_single(filename, vtk_type=None):
    vtk_type = vtk_type or _get_vtk_type(filename)
    if vtk_type == 'UnstructuredGrid':
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise NotImplementedError(f"VTK Files of type {vtk_type} can not yet be processed")
    reader.SetFileName(filename)
    reader.Update()
    geometryFilter = vtk.vtkGeometryFilter()
    geometryFilter.SetInputData(reader.GetOutput())
    geometryFilter.Update()
    return geometryFilter.GetOutput()


def _get_collection_data(filename):
    path = Path(filename)
    assert path.is_file()
    bf = BadgerFish(dict_type=OrderedDict)
    return path, bf.data(fromstring(open(path, 'rb').read()))


def _get_vtk_type(path):
    '''We use the incremental event emitting parser
    here since we can expect to encounter appended binary data in the xml
    which lxml cannot parse.
    :param path: vtk file to peek into
    :return: None if no VTKFile element found, ellse the type attribute of the VTKFile element
    '''
    parser = etree.XMLPullParser(events=('start',))
    with open(path, 'rb') as xml:
        for lines in xml.readlines():
            parser.feed(lines)
            for action, element in parser.read_events():
                if element.tag == 'VTKFile':
                    return element.get('type')
    return None


def read_vtkfile(filename):
    vtk_type = _get_vtk_type(filename)
    if vtk_type == 'Collection':
        path, xml = _get_collection_data(filename)
        return _read_collection(xml)
    return [None, _read_single(filename, vtk_type)]
