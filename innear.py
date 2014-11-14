'''Script to register cell positions'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        df_to_transform['x'][cell] = point[0]
        df_to_transform['y'][cell] = point[1]
        df_to_transform['z'][cell] = point[2]


if __name__ == '__main__':
    '''Demonstrates registration of two pyramids'''

    # Create pyramids
    target_pyramid = pd.DataFrame(
        {'x': [0, 0, 1, 1, 0.5],
         'y': [0, 1, 1, 0, 0.5],
         'z': [0, 0, 0, 0, 1]})

    source_pyramid = pd.DataFrame(
        {'x': target_pyramid['x']*np.cos(1) - target_pyramid['y']*np.sin(1),
         'y': target_pyramid['x']*np.sin(1) + target_pyramid['y']*np.cos(1),
         'z': target_pyramid['z'] + 0.25,
         'selection': ['pyramid', 'pyramid', 'pyramid', 'pyramid', 'pyramid']})

    source_points = source_pyramid.append(pd.DataFrame(
        {'x': source_pyramid['x'][4] + np.random.randn(10)/10,
         'y': source_pyramid['y'][4] + np.random.randn(10)/10,
         'z': source_pyramid['z'][4] + np.random.randn(10)/10,
         'selection': ['points' for _ in range(10)]})).reset_index()

    # Registration
    before_pyramid = source_pyramid.copy()
    before_points = source_points[source_points.selection == 'points'].copy()
    register(target_pyramid, source_pyramid, source_points)
    after_pyramid = source_points[source_points.selection == 'pyramid']
    after_points = source_points[source_points.selection == 'points']

    # Plot before & after
    before = plt.subplot(1,2,1, projection='3d')
    plt.title('Before registration')
    before.auto_scale_xyz([-1,1], [-1,1], [-1,1])
    before.plot_trisurf(target_pyramid['x'], target_pyramid['y'], target_pyramid['z'],
        shade=False, color='Green', alpha=0.25, linewidth=0.2)
    before.plot_trisurf(before_pyramid['x'], before_pyramid['y'], before_pyramid['z'],
        shade=False, color='Red', alpha=0.25, linewidth=0.2)
    before.scatter(before_points['x'], before_points['y'], before_points['z'])

    after = plt.subplot(1,2,2, projection='3d')
    plt.title('After registration')
    after.auto_scale_xyz([-1,1], [-1,1], [-1,1])
    after.plot_trisurf(target_pyramid['x'], target_pyramid['y'], target_pyramid['z'],
        shade=False, color='Green', alpha=0.25, linewidth=0.2)
    after.plot_trisurf(after_pyramid['x'], after_pyramid['y'], after_pyramid['z'],
        shade=False, color='Red', alpha=0.25, linewidth=0.2)
    after.scatter(after_points['x'], after_points['y'], after_points['z'])

    plt.tight_layout()
    plt.show()
