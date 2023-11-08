import os
import vtk
import gdcm
import numpy as np
import pydicom as pd
import SimpleITK as sitk

workfp = "G:/CTs"
Rawfp = "Dcm_to_Raw"
dcmfilename = "0625mmBone_3"

def save_mhd(imageDirPath, SaveRawDicom, RawName):
    # 路径
    RawName = RawName + ".mhd"

    # 读取文件夹内图像
    seriesReader = sitk.ImageSeriesReader()
    seriesIds = seriesReader.GetGDCMSeriesIDs(imageDirPath)
    filenames = seriesReader.GetGDCMSeriesFileNames(imageDirPath, seriesIds[0])
    seriesReader.SetFileNames(filenames)
    image = seriesReader.Execute()

    # 获取图像参数
    RefDs = pd.read_file(filenames[0])
    # 得到x方向和y方向的Spacing并得到z方向的层厚
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
    # 得到图像的原点
    Origin = RefDs.ImagePositionPatient

    # 图像预处理
    img_arr = sitk.GetArrayViewFromImage(image)  # 转换为array格式

    # 切片
    ArrayDicom = img_arr
    ArrayDicom2 = ArrayDicom[210:230, 220:270, 225:280]  # Neck10_3----195:237, 230:289, 213:305

    # 高通阈值化
    b = len(ArrayDicom2.flat)
    a = ArrayDicom2.reshape(b)
    for i in range(0, len(a)):
        if a[i] < 236:
            a[i] = img_arr[0, 0, 0]
    ArrayDicom2 = a.reshape(ArrayDicom2.shape)

    # 将numpy数组通过SimpleITK转化为mhd和raw文件
    sitk_img = sitk.GetImageFromArray(ArrayDicom, isVector=False)
    sitk_img.SetSpacing(ConstPixelSpacing)
    sitk_img.SetOrigin(Origin)
    print(sitk_img)
    sitk.WriteImage(sitk_img, os.path.join(SaveRawDicom, RawName))


def show(fileName):
    colors = vtk.vtkNamedColors()

    # colors.SetColor("SkinColor", [255, 125, 64, 255])
    colors.SetColor("SkinColor", [0, 0, 0, 0])
    colors.SetColor("BkgColor", [51, 77, 102, 255])

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

    # 设置高斯滤波，平滑处理
    Gfilter = vtk.vtkImageGaussianSmooth()
    Gfilter.SetInputConnection(reader.GetOutputPort())
    Gfilter.SetRadiusFactor(1)
    Gfilter.SetStandardDeviation(1)
    Gfilter.Update()

    # vtk.vtkImageGaussianSmooth.GetImageDataInput()
    # An isosurface, or contour value of 1150 is known to correspond to the
    # bone of the patient.
    # The triangle stripper is used to create triangle strips from the
    # isosurface these render much faster on may systems.
    boneExtractor = vtk.vtkMarchingCubes()
    boneExtractor.SetInputConnection(reader.GetOutputPort())
    # boneExtractor.SetInputConnection(Gfilter.GetOutputPort())
    #boneExtractor.SetValue()
    boneExtractor.SetValue(0, 600)  # 0, 1150

    boneStripper = vtk.vtkStripper()
    boneStripper.SetInputConnection(boneExtractor.GetOutputPort())

    boneMapper = vtk.vtkPolyDataMapper()
    boneMapper.SetInputConnection(boneStripper.GetOutputPort())
    boneMapper.ScalarVisibilityOff()

    bone = vtk.vtkActor()
    bone.SetMapper(boneMapper)
    bone.GetProperty().SetDiffuseColor(colors.GetColor3d("Ivory"))

    # An outline provides context around the data.
    #
    outlineData = vtk.vtkOutlineFilter()
    outlineData.SetInputConnection(Gfilter.GetOutputPort())

    mapOutline = vtk.vtkPolyDataMapper()
    mapOutline.SetInputConnection(outlineData.GetOutputPort())

    outline = vtk.vtkActor()
    outline.SetMapper(mapOutline)
    outline.GetProperty().SetColor(colors.GetColor3d("Black"))

    # It is convenient to create an initial view of the data. The FocalPoint
    # and Position form a vector direction. Later on (ResetCamera() method)
    # this vector is used to position the camera to look at the data in
    # this direction.
    aCamera = vtk.vtkCamera()
    aCamera.SetViewUp(0, 0, -1)
    aCamera.SetPosition(0, -1, 0)
    aCamera.SetFocalPoint(0, 0, 0)
    aCamera.ComputeViewPlaneNormal()
    aCamera.Azimuth(30.0)
    aCamera.Elevation(30.0)

    # Actors are added to the renderer. An initial camera view is created.
    # The Dolly() method moves the camera towards the FocalPoint,
    # thereby enlarging the image.
    # aRenderer.AddActor(outline)
    # aRenderer.AddActor(skin)
    aRenderer.AddActor(bone)
    aRenderer.SetActiveCamera(aCamera)
    aRenderer.ResetCamera()
    aCamera.Dolly(1.5)

    # Set a background color for the renderer and set the size of the
    # render window (expressed in pixels).
    aRenderer.SetBackground(colors.GetColor3d("BkgColor"))
    renWin.SetSize(1080, 960)

    # Note that when camera movement occurs (as it does in the Dolly()
    # method), the clipping planes often need adjusting. Clipping planes
    # consist of two planes: near and far along the view direction. The
    # near plane clips out objects in front of the plane the far plane
    # clips out objects behind the plane. This way only what is drawn
    # between the planes is actually rendered.
    aRenderer.ResetCameraClippingRange()

    # Initialize the event loop and then start it.
    iren.Initialize()
    iren.Start()


save_mhd(os.path.join(workfp, dcmfilename), workfp, Rawfp)

show(os.path.join(workfp, Rawfp + ".mhd"))
