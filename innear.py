'''Toolkit to analyse inner ear development'''
import numpy as np
import pandas as pd
import scipy.spatial as spatial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import vtk


def register(target_df, source_df, df_to_transform):
    '''Register source on top of target by ICP and transform df in place'''
    # Create vtk data structures for source points
    TargetPoints = vtk.vtkPoints()
    TargetVertices = vtk.vtkCellArray()
    for cell in target_df.index:
        id = TargetPoints.InsertNextPoint(
            target_df['x'][cell], 
            target_df['y'][cell],
             target_df['z'][cell])
        TargetVertices.InsertNextCell(1)
        TargetVertices.InsertCellPoint(id)

    TargetPolyData = vtk.vtkPolyData()
    TargetPolyData.SetPoints(TargetPoints)
    TargetPolyData.SetVerts(TargetVertices)
    if vtk.VTK_MAJOR_VERSION <= 5:
        TargetPolyData.Update()
        
    # Create vtk data structures for target points
    SourcePoints = vtk.vtkPoints()
    SourceVertices = vtk.vtkCellArray()
    for cell in source_df.index:
        id = SourcePoints.InsertNextPoint(
            source_df['x'][cell], 
            source_df['y'][cell], 
            source_df['z'][cell])
        SourceVertices.InsertNextCell(1)
        SourceVertices.InsertCellPoint(id)

    SourcePolyData = vtk.vtkPolyData()
    SourcePolyData.SetPoints(SourcePoints)
    SourcePolyData.SetVerts(SourceVertices)
    if vtk.VTK_MAJOR_VERSION <= 5:
        SourcePolyData.Update()

    # Create vtk data structures for all points that need to be transformed
    PointsToTransform = vtk.vtkPoints()
    VerticesToTransform = vtk.vtkCellArray()
    for cell in df_to_transform.index:
        id = PointsToTransform.InsertNextPoint(
            df_to_transform['x'][cell],
            df_to_transform['y'][cell],
            df_to_transform['z'][cell])
        VerticesToTransform.InsertNextCell(1)
        VerticesToTransform.InsertCellPoint(id)

    PolyDataToTransform = vtk.vtkPolyData()
    PolyDataToTransform.SetPoints(PointsToTransform)
    PolyDataToTransform.SetVerts(VerticesToTransform)
    if vtk.VTK_MAJOR_VERSION <= 5:
        PolyDataToTransform.Update()

    # Register source df on top of target df
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(SourcePolyData)
    icp.SetTarget(TargetPolyData)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(20)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()

    # Transform all cells
    icpTransformFilter = vtk.vtkTransformPolyDataFilter()
    if vtk.VTK_MAJOR_VERSION <= 5:
        icpTransformFilter.SetInput(PolyDataToTransform)
    else:
        icpTransformFilter.SetInputData(PolyDataToTransform)

    icpTransformFilter.SetTransform(icp)
    icpTransformFilter.Update()

    transformedSource = icpTransformFilter.GetOutput()

    for cell in range(df_to_transform.shape[0]):
        point = [0,0,0]
        transformedSource.GetPoint(cell, point)
        df_to_transform.loc[cell, 'x'] = point[0]
        df_to_transform.loc[cell, 'y'] = point[1]
        df_to_transform.loc[cell, 'z'] = point[2]


def trace_lineage(df): # Not tested
    '''Adds column with cell_id'''
    n_cells = df.loc[df.timestep == 1].shape[0]
    df.loc[df.timestep == 1, 'cell_id'] = range(n_cells)
    for timestep in set([ts for ts in df.timestep.values if ts > 1]):
        previous_df = df[df.timestep == timestep - 1]
        for index, row in df[df.timestep == timestep].iterrows():
            mother_row = previous_df[previous_df.id_center == row['id_mother']]
            try:
                df.loc[df.id_center == row['id_center'], 'cell_id'] = \
                    int(mother_row['cell_id'].values[0])
            except:
                print('Warning: motherless cell')
                n_cells = n_cells + 1
                df.loc[df.id_center == row['id_center'], 'cell_id'] = n_cells
        cells = df[df.timestep == timestep]['cell_id'].values
        divisions = set([c_id for c_id in cells if cells.tolist().count(c_id) > 1])
        for division in divisions:
            cells = df.loc[df.cell_id == division]
            for index, _ in cells[-2:].iterrows():
                n_cells = n_cells + 1
                df.loc[index, 'cell_id'] = n_cells
    print('{} cells found'.format(n_cells))


def nn_distance(df): # Not tested
    '''Adds column with distance to nearest neighbour'''
    for timestep in set(df.timestep.values):
        tree = spatial.cKDTree(
            df[df.timestep == timestep].as_matrix(['x', 'y', 'z']))
        df.loc[df.timestep == timestep, 'NN distance'] = \
            [tree.query(point, 2)[0][1] for point in tree.data]
    df.replace(np.inf, np.nan, inplace=True)


def estimate_density(df, radius=0.1):
    '''Adds column w/ local density est. by counting points within radius'''
    for timestep in set(df.timestep.values):
        tree = spatial.cKDTree(df.as_matrix(['x', 'y', 'z']))
        volume = 4*np.pi*radius**3/3
        df.loc[:, 'r = {:1.2f}'.format(radius)] = \
            [(tree.query_ball_point(point, radius).__len__())/volume 
                for point in tree.data]


def sweep_radii(df, r_min=0.025, r_max=0.3, n=7):
    '''Estimates density for different radii to find a sensible one'''
    for i in range(n):
        radius = i*(r_max-r_min)/(n-1) + r_min
        estimate_density(df, radius=radius)


def plot_densities(df, **kwargs):
    '''Plot distributions of density estimates'''
    densities = [column for column in df.columns if column.startswith('r = ')]
    sns.violinplot(df[densities], **kwargs)


def equalize_axis3d(ax, zoom=1):
    ''' after http://stackoverflow.com/questions/8130823/
    set-matplotlib-3d-plot-aspect-ratio'''
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    spread = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    max_spread = max(abs(spread))
    r = max_spread/2
    for center, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(center - r/zoom, center + r/zoom)


if __name__ == '__main__':
    '''Demonstrates registration and density estimation'''

    # Create pyramids to register on top of each other
    target_pyramid = pd.DataFrame(
        {'x': [0, 0, 1, 1, 0.5],
         'y': [0, 1, 1, 0, 0.5],
         'z': [0, 0, 0, 0, 1]})

    source_pyramid = pd.DataFrame(
        {'x': target_pyramid['x']*np.cos(1) - target_pyramid['y']*np.sin(1),
         'y': target_pyramid['x']*np.sin(1) + target_pyramid['y']*np.cos(1),
         'z': target_pyramid['z'] + 0.25,
         'selection': ['pyramid', 'pyramid', 'pyramid', 'pyramid', 'pyramid']})

    # Create additional points
    n_points = 150
    source_points = source_pyramid.append(pd.DataFrame(
        {'x': source_pyramid['x'][4] + np.random.randn(n_points)/10,
         'y': source_pyramid['y'][4] + np.random.randn(n_points)/10,
         'z': source_pyramid['z'][4] + np.random.randn(n_points)/10 + 0.2,
         'selection': ['points' for _ in range(n_points)]})).reset_index()

    # Registration
    before_pyramid = source_pyramid.copy()
    before_points = source_points[source_points.selection == 'points'].copy()
    register(target_pyramid, source_pyramid, source_points)
    after_pyramid = source_points[source_points.selection == 'pyramid']
    after_points = source_points[source_points.selection == 'points']

    # Estimate density
    after_points.loc[:, 'timestep'] = 1
    sweep_radii(after_points, n=12)
    sns.set(style="whitegrid")
    plt.title('Density estimtates for different radii')
    plot_densities(after_points, color='PuRd_r')
    plt.show()

    # Plot before & after
    sns.set(style="white")
    before = plt.subplot(1,2,1, projection='3d')
    plt.title('Before registration')
    before.set_aspect('equal')
    before.axis('off')
    before.plot_trisurf(target_pyramid['x'], target_pyramid['y'], target_pyramid['z'],
        shade=False, color='Green', alpha=0.25, linewidth=0.2)
    before.plot_trisurf(before_pyramid['x'], before_pyramid['y'], before_pyramid['z'],
        shade=False, color='Red', alpha=0.25, linewidth=0.2)
    before.scatter(before_points['x'], before_points['y'], before_points['z'])
    equalize_axis3d(before, 1.75)
    after = plt.subplot(1,2,2, projection='3d')
    plt.title('After registration, with density estimates')
    after.set_aspect('equal')
    after.axis('off')
    after.plot_trisurf(target_pyramid['x'], target_pyramid['y'], target_pyramid['z'],
        shade=False, color='Green', alpha=0.25, linewidth=0.2)
    after.plot_trisurf(after_pyramid['x'], after_pyramid['y'], after_pyramid['z'],
        shade=False, color='Red', alpha=0.25, linewidth=0.2)
    after.scatter(after_points['x'], after_points['y'], after_points['z'], 
        c=after_points['r = 0.12'], cmap='RdBu_r')
    equalize_axis3d(after, 1.5)

    plt.tight_layout()
    plt.show()
