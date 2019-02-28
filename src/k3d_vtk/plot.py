from k3d_vtk.reader import read_vtkfile

import k3d
import vtk

def plot(vtkfile_path, vmin, vmax):
    data = read_vtkfile(vtkfile_path)

    geometryFilter = vtk.vtkGeometryFilter()
    geometryFilter.SetInputData(data)
    geometryFilter.Update()

    plot = k3d.plot()
    poly_data = geometryFilter.GetOutput()

    model_matrix = (
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    )
    data = k3d.vtk_poly_data(poly_data, color_attribute=('Data', vmin, vmax),
                             color_map=k3d.basic_color_maps.CoolWarm, model_matrix=model_matrix)
    plot += data
    plot.display()
    return plot
