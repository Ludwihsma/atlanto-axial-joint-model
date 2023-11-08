import pydicom as pd
import SimpleITK as sitk
import os
import numpy as np
import cv2 as cv


def save_as_mhd2(fp,SaveRawDicom):
    lstFilesDCM = []

    # 将PathDicom文件夹下的dicom文件地址读取到lstFilesDCM中
    for dirName, subdirList, fileList in os.walk(fp):
        for filename in fileList:
            if ".dcm" in filename.lower():  # 判断文件是否为dicom文件
                # 			print(filename)
                lstFilesDCM.append(os.path.join(dirName, filename))  # 加入到列表中
    # 第一步：将第一张图片作为参考图片，并认为所有图片具有相同维度
    RefDs = pd.read_file(lstFilesDCM[0])  # 读取第一张dicom图片

    # 第二步：得到dicom图片所组成3D图片的维度
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))  # ConstPixelDims是一个元组

    # 第三步：得到x方向和y方向的Spacing并得到z方向的层厚
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

    # 第四步：得到图像的原点
    Origin = RefDs.ImagePositionPatient

    # 根据维度创建一个numpy的三维数组，并将元素类型设为：pixel_array.dtype
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)  # array is a numpy array
    # ArrayDicom = np.zeros(ConstPixelDims, dtype=np.int)

    # 第五步:遍历所有的dicom文件，读取图像数据，存放在numpy数组中
    i = 0
    for i, filenameDCM in enumerate(lstFilesDCM):
        ds = sitk.ReadImage(filenameDCM)
        # 	ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array
        image_array = sitk.GetArrayFromImage(ds)
        # ArrayDicom[:, :, i] = np.asarray(Image.fromarray(image_array[0]))
        ArrayDicom[:, :, i] = np.asarray(image_array[0])
    #     cv2.imwrite("out_" + str(i) + ".png", ArrayDicom[:, :, i])

    # 第六步：对numpy数组进行转置，即把坐标轴（x,y,z）变换为（z,y,x）,这样是dicom存储文件的格式，即第一个维度为z轴便于图片堆叠
    ArrayDicom = np.transpose(ArrayDicom, (2, 0, 1))

    ArrayDicom = np.append(ArrayDicom[280:, :, :], ArrayDicom[:280, :, :], axis=0)
    t, ArrayDicom2 = cv.threshold(ArrayDicom, 230, 3000, cv.THRESH_TOZERO)
    ArrayDicom2 = ArrayDicom2[29:40, 230:289, 215:308]  #
    # 第七步：将现在的numpy数组通过SimpleITK转化为mhd和raw文件
    sitk_img = sitk.GetImageFromArray(ArrayDicom2, isVector=False)
    sitk_img.SetSpacing(ConstPixelSpacing)
    sitk_img.SetOrigin(Origin)
    print(sitk_img)
    sitk.WriteImage(sitk_img, os.path.join(SaveRawDicom, "neck10-9" + ".mhd"))



    for i in upbond:
            for k in i < lowbond:
                if k:
                    sub.append(lowbond[i < lowbond])   # 在一个下界前的上界位置
                else:
                    continue

        # location = float(sub+0.3), float(sub)+float(abs(lowbond - sub)+0.3)
        if sub:
            location = [sub, sub + abs(lowbond - sub)+1]
            return location
        else:
            return []