import vtk
from vtk.util import numpy_support


def show(fileName):
    colors = vtk.vtkNamedColors()

    # colors.SetColor("SkinColor", [255, 125, 64, 255])
    # colors.SetColor("SkinColor", [0, 0, 0, 0])
    # colors.SetColor("BkgColor", [51, 77, 102, 255])

    # Create the renderer, the render window, and the interactor. The renderer
    # draws into the render window, the interactor enables mouse- and
    # keyboard-based interaction with the data within the render window.
    #
    aRenderer = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(aRenderer)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # The following reader is used to read a series of 2D slices (images)
    # that compose the volume. The slice dimensions are set, and the
    # pixel spacing. The data Endianness must also be specified. The reader
    # uses the FilePrefix in combination with the slice number to construct
    # filenames using the format FilePrefix.%d. (In this case the FilePrefix
    # is the root name of the file: quarter.)
    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(fileName)

    Gfilter = vtk.vtkImageGaussianSmooth()
    Gfilter.SetInputConnection(reader.GetOutputPort())
    Gfilter.SetRadiusFactor(3)
    Gfilter.SetStandardDeviation(1)
    Gfilter.Update()

    boneExtractor = vtk.vtkMarchingCubes()
    boneExtractor.SetInputConnection(Gfilter.GetOutputPort())
    # boneExtractor.SetInputConnection(Gfilter.GetOutputPort())
    # boneExtractor.SetValue()
    boneExtractor.SetValue(0, 600)  # 0, 1150

    boneStripper = vtk.vtkStripper()
    boneStripper.SetInputConnection(boneExtractor.GetOutputPort())

    curvaturesFilter = vtk.vtkCurvatures()  # 实现了一个网格模型的曲率计算
    curvaturesFilter.SetInputConnection(boneStripper.GetOutputPort())
    curvaturesFilter.SetCurvatureTypeToMaximum()  # 计算最大主曲率
    # curvaturesFilter.SetCurvatureTypeToMinimum()  # 最小曲率
    # curvaturesFilter.SetCurvatureTypeToGaussian()  # 高斯曲率
    # curvaturesFilter.SetCurvatureTypeToMean()    # 平均曲率
    curvaturesFilter.Update()

    scalarRange = curvaturesFilter.GetOutput().GetScalarRange()


    lut = vtk.vtkLookupTable()  # 定义一个256色的vtkLookupTable对象
    lut.SetHueRange(0.9, 0.0)
    lut.SetAlphaRange(1.0, 1.0)
    lut.SetSaturationRange(1.0, 1.0)
    lut.SetNumberOfTableValues(255)
    # lut.SetObjectName("最大主曲率")
    lut.SetRange(-1, 3)
    lut.Build()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(curvaturesFilter.GetOutput())
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(scalarRange)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    scalarBar = vtk.vtkScalarBarActor()  # 将颜色以图形的形式显示，并支持设置图形相应的名字和显示的数据label的个数
    scalarBar.SetLookupTable(mapper.GetLookupTable())
    # scalarBar.SetTitle(curvaturesFilter.GetOutput().GetPointData().GetScalars().GetName())  # 获取相应的曲率数据
    scalarBar.SetNumberOfLabels(5)  # 显示5个数据label

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.AddActor2D(scalarBar)
    renderer.SetBackground(1.0, 1.0, 1.0)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(640, 480)
    renderWindow.Render()
    renderWindow.SetWindowName('最大主曲率')

    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.Initialize()
    renderWindowInteractor.Start()

    STLWriter = vtk.vtkSTLWriter()#vtk.vtkOBJWriter

    STLWriter.SetFileName('G:/CTs/0625mmBone_3'+'low_bond_STL_Model.stl')
    STLWriter.SetInputConnection(curvaturesFilter.GetOutputPort())
    STLWriter.Write()
    #STLWriter.Update()

    #OBJWriter = vtk.vtkOBJWriter()
    #OBJWriter.SetFileName('G:/CTs/'+'OBJ_Model.obj')
    #OBJWriter.SetInputConnection(curvaturesFilter.GetOutputPort())
    #OBJWriter.Write()
