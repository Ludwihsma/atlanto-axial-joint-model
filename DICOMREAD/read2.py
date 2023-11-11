# import cv2.cv2
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pydicom as pd
import SimpleITK as sitk
import pandas
import gdcm
import cv2 as cv

# 模型裁剪范围，二值化阈值点
range_of_model_Z_left = 195   # 模型裁剪范围，数组第一个位置的左端点
range_of_model_Z_right = 210    # 模型裁剪范围，数组第一个位置的右端点
range_of_model_X_left = 230   # 二，左
range_of_model_X_right = 290    # 二，右
range_of_model_Y_left = 213   # 三，左
range_of_model_Y_right = 305   # 三，右
binarization_threshold = 0   # 二值化的阈值，低于此数值的位置赋予数组第一个数的数值

# 模型修剪和不处理的范围
model_change_range_X_left = 30
model_change_range_X_right = 40
model_change_range_Y_left1 = 3
model_change_range_Y_right1 = 10
model_change_range_Z_left = 6
model_change_range_Z_right = 8
model_skip_range_Y_left = 22
model_skip_range_Y_right = 36

max_lenth_of_aaj = 12   # 寰枢椎关节最大间隙长度
right_endpoint_of_left_aaj_domain_Y = 15   # 左寰枢椎关节区域的右侧边界点
left_endpoint_of_right_aaj_domaih_Y = 35   # 右寰枢椎关节区域的左侧边界点

#高斯滤波相关参数
GaussianBlur_kernel_size = 3   # 卷积核大小（n*n）尺寸越大越平滑
GaussianBlur_sigma_value = 1   # 核函数标准差，值越大中心像素权重越大


def build_model(fp, save_path, model_name):
    # 路径
    imageDirPath = fp
    model_name = model_name + ".mhd"
    np.set_printoptions(threshold=sys.maxsize)

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

    ArrayDicom2 = img_arr
    ArrayDicom2 = ArrayDicom2[range_of_model_Z_left:range_of_model_Z_right,
                  range_of_model_X_left:range_of_model_X_right,
                  range_of_model_Y_left:range_of_model_Y_right]  # Neck10_3----195:237, 230:289, 213:305 0625mm:210:230, 220:270, 225:280

    b = len(ArrayDicom2.flat)
    a = ArrayDicom2.reshape(b)
    for i in range(len(a)):
        if a[i] < binarization_threshold:
            a[i] = img_arr[0, 0, 0]
        # if a[i] > 500:
        #    a[i] = 1000
    ArrayDicom2 = a.reshape(ArrayDicom2.shape)  # 20,50,55

    model = np.zeros(ArrayDicom2.shape)
    upbonds = np.zeros(ArrayDicom2.shape)
    lowbonds = np.zeros(ArrayDicom2.shape)

    tall = []
    # 通过扫描建立模型
    for i in range(ArrayDicom2.shape[1]):
        if model_change_range_X_left < i < model_change_range_X_right:
            for j in range(ArrayDicom2.shape[2]):
                if model_change_range_Y_left1 < j < model_change_range_Y_right1:
                    ArrayDicom2[model_change_range_Z_left:model_change_range_Z_right, i, j] = img_arr[0, 0, 0]
        for j in range(ArrayDicom2.shape[2]):
            if model_skip_range_Y_left < j < model_skip_range_Y_right:
                continue

            tmp = find_gap(ArrayDicom2[:, i, j])

            if tmp:
                upbond = tmp[0]
                lowbond = tmp[1]

                for k in range(0, len(upbond)):
                    if abs(lowbond[k] - upbond[k]) > max_lenth_of_aaj:
                        continue
                    else:
                        if j < right_endpoint_of_left_aaj_domain_Y or j > left_endpoint_of_right_aaj_domaih_Y:
                            model[upbond[0]:lowbond[0], i, j] = 1050   # 将找到的第一个上界和下界作为模型的边界
                            # model[:upbond[0], i, j] = 1050
                        else:
                            model[upbond[k]:lowbond[k], i, j] = 1050   # 将找到的最后一个上界和下界作为模型的边界
                            # model[:upbond[k], i, j] = 1050
    # 对模型高斯滤波
    for i in range(model.shape[0]):
        model[i, :, :] = cv.GaussianBlur(model[i, :, :], (GaussianBlur_kernel_size, GaussianBlur_kernel_size), 1)

    talls = np.array([0])   # 计算高度
    bondscoord = [np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])]

    # model2 = cv.GaussianBlur(model, (3, 3), 2)
    model2 = model
    for i in range(model2.shape[1]):
        for j in range(model2.shape[2]):
            tmp = model2[:, i, j]
            up = low = 0 > 1   # false
            ttall = np.array(0)

            for k in range(len(tmp)):
                if tmp[k] > 0 or up:
                    if not up:
                        upbonds[k, i, j] = 1050
                        up = 1 > 0   # true
                        ttall = k

                    elif tmp[k] == 0 and not low:
                        lowbonds[k - 1, i, j] = 1050
                        low = 1 > 0
                        ttall = np.append(ttall, abs(k - ttall))

            ttall = np.delete(ttall, 0)   # 删去初始化值：0
            talls = np.append(talls, ttall)

    # 间隙高度
    talls = np.delete(talls, 0)

    left_upbond = upbonds[:, :, :28]
    right_upbond = upbonds[:, :, 28:]
    left_lowbond = lowbonds[:, :, :28]
    right_lowbond = lowbonds[:, :, 28:]
    bonds4 = [left_upbond, right_upbond, left_lowbond, right_lowbond]
    bondlensX = [99, 99, 0, 0]
    bondlensY = [99, 99, 0, 0]
    areas = np.array((0, 0, 0, 0))
    slopes = [0, 0, 0, 0]
    VorC = [0, 0, 0, 0]
    curvatures = [0, 0, 0, 0]
    curlocation = np.array([[0, 0, 0]])
    coord1 = coord2 = coord3 = coord4 = np.array([[0, 0, 0, 0, 0]])
    coorddatas = [coord1, coord2, coord3, coord4]

    # 计算平均斜率，平均倾斜角，曲度，搜索矢径和横径
    for e in range(0, 4):
        slope = np.array(0)
        curvature = np.array(0)
        concavity = np.array([0])
        # 计算曲面平均斜率
        for i in range(bonds4[e].shape[1]):  # 深度、X,一次一张图片，计算一次斜率
            tmpx = np.array(0)
            tmpy = np.array(0)

            for j in range(bonds4[e].shape[2]):
                tmp = bonds4[e][:, i, j]
                tmp = tmp > 0
                for k in range(len(tmp)):
                    if tmp[k]:
                        tmpx = np.append(tmpx, j)
                        tmpy = np.append(tmpy, k)

            tmpx = np.delete(tmpx, 0)
            tmpy = np.delete(tmpy, 0)
            if len(tmpx) > 0 and len(tmpy) > 0:

                # 计算斜率
                tslope, intercept = np.polyfit(tmpx, tmpy, 1)
                slope = np.append(slope, tslope)
                # 计算曲度
                L = int(len(tmpx))
                coordA = np.array((tmpx[0], tmpy[0]))
                coordB = np.array((tmpx[L - 1], tmpy[L - 1]))

                # if L % 2 == 1:
                #    coordO = np.array((tmpx[int((L - 1) / 2)], tmpy[int((L - 1) / 2)]))
                # else:
                #    coordO = np.array(( (tmpx[int(L / 2)] + tmpx[int(L / 2 - 1)]) / 2, (tmpy[int(L / 2)] + tmpy[int(L / 2 - 1)]) / 2))
                if len(tmpx) > 2 and len(tmpy) > 2:
                    curvature2 = np.array(0)
                    # 搜寻大弯曲度
                    for q in range(1, len(tmpx) - 1):
                        coordO = np.array([tmpx[q], tmpy[q]])
                        coordA2 = coordA - coordO
                        coordB2 = coordB - coordO
                        tcur = coordA2.dot(coordB2) / (np.sqrt(np.square(coordA2[0]) + np.square(coordA2[1])) * np.sqrt(
                            np.square(coordB2[0]) +
                            np.square(coordB2[1])))
                        if np.abs(tcur) > 1:
                            tcur = int(tcur)

                        # 计算每个点的凹凸性
                        tk = (coordA[1] - coordB[1]) / (coordA[0] - coordB[0])
                        tconcavity = 0
                        if (coordA[0] - coordO[0]) * tk - np.abs(coordA[1] - coordO[1]) < 0:  # A是最左边的点，如果是凸的
                            tconcavity = 1
                        elif (coordA[0] - coordO[0]) * tk - np.abs(coordA[1] - coordO[1]) > 0:  # 如果是凹的
                            tconcavity = -1
                        else:
                            tconcavity = 0

                        tcur = np.arccos(tcur) * 180 / np.pi
                        concavity = np.append(concavity, tconcavity)
                        curvature2 = np.append(curvature2, tcur)
                        coorddatas[e] = np.concatenate((coorddatas[e],
                                                        np.array([[tmpy[q], i, tmpx[q], tcur, tconcavity]])), axis=0)

                    curvature2 = np.delete(curvature2, 0)
                    curvature = np.append(curvature, np.min(curvature2))

                    # 记录弯曲角度最小值的下标
                    tlocation = np.where(curvature2 == np.min(curvature2))[0]
                    if len(tlocation) > 1:
                        if len(tlocation) % 2 == 1:
                            tlocation = tlocation[int((len(tlocation) + 1) / 2)]
                        else:
                            tlocation = tlocation[int(len(tlocation) / 2)]
                    elif len(tlocation) == 1:
                        tlocation = tlocation[0]
                    # 记录最小弯曲角度所在顶点的坐标
                    curlocation = np.concatenate((curlocation, np.array([[i, tmpx[tlocation], tmpy[tlocation]]])),
                                                 axis=0)

        slope = np.delete(slope, 0)
        slopes[e] = np.average(slope)
        curlocation = np.delete(curlocation, 0, 0)
        curvature = np.delete(curvature, 0)
        curvatures[e] = np.average(curvature)
        VorC[e] = np.average(concavity)

        for i in range(0, bonds4[e].shape[0]):

            # 搜索矢径
            for j in range(0, bonds4[e].shape[2]):
                for k in range(0, bonds4[e].shape[1]):
                    tmp = bonds4[e][i, :, j] > 0
                    if tmp[k]:
                        areas[e] += 1  # 面积计数
                        if e < 2:
                            if k < bondlensX[e]:
                                bondlensX[e] = k
                        else:
                            if k > bondlensX[e]:
                                bondlensX[e] = k

            # 搜索横径
            for j in range(0, bonds4[e].shape[1]):
                for k in range(0, bonds4[e].shape[2]):
                    tmp = bonds4[e][i, j, :] > 0
                    if tmp[k]:
                        if e < 2:
                            if k < bondlensY[e]:
                                bondlensY[e] = k
                        else:
                            if k > bondlensY[e]:
                                bondlensY[e] = k

    bondX = [ConstPixelSpacing[0] * abs(bondlensX[2] - bondlensX[0]),
             ConstPixelSpacing[0] * abs(bondlensX[3] - bondlensX[1])]
    bondY = [ConstPixelSpacing[1] * abs(bondlensY[2] - bondlensY[0]),
             ConstPixelSpacing[1] * abs(bondlensY[3] - bondlensY[1])]
    slope_angle = np.arctan(np.abs(slopes)) * 180 / np.pi
    areas = areas * ConstPixelSpacing[2] * ConstPixelSpacing[2] / np.cos(np.arctan(np.abs(slopes)))
    talls = np.delete(talls, np.where(talls == 0)[0])
    tall_average = np.average(talls * ConstPixelSpacing[2] * np.cos(np.arctan(np.abs(np.average(slopes)))))

    # 利用pandas数据整理，代码未完成
    dfur = pandas.DataFrame({'z': coord3[:, 0],
                             'x': coord3[:, 1],
                             'y': coord3[:, 2],
                             'curvature': coord3[:, 3],
                             'vorc': coord3[:, 4]})
    dfur.sort_values(by='curvature')

    # 在环境终端输出部分计算结果
    print("平均间隙高度:", tall_average)
    print("矢径:", bondX)
    print("横径:", bondY)
    print("各个解剖面的倾斜角:", slope_angle)
    print("各个解剖面的面积:", areas)
    print("各个解剖面的曲度:", curvatures)
    print('各个面的凹凸性:', VorC)

    # 将边界数据写入文件
    writecsvnames = [save_path + "/low_right.txt",
                     save_path + "/low_left.txt",
                     save_path + "/up_right.txt",
                     save_path + "/up_left.txt"]

    for i in range(len(coorddatas)):
        file = open(writecsvnames[i], "w+")
        file.write(str(coorddatas[i]))
        file.close()

    # 将模型几何特征参数写入文件
    file = open(save_path + "/parameters.txt", "w+")
    file.write("平均间隙高度:\t" + str(tall_average) + '\n\n' +
               "矢径:\t" + str(bondX) + '\n\n' +
               "横径:\t" + str(bondY) + '\n\n' +
               "各个解剖面的倾斜角:\t" + str(slope_angle) + '\n\n' +
               "各个解剖面的面积:\t" + str(areas) + '\n\n' +
               "各个解剖面的曲度:\t" + str(curvatures) + '\n\n' +
               '各个面的凹凸性:\t' + str(VorC))

    # 将numpy数组通过SimpleITK转化为mhd和raw文件
    sitk_img = sitk.GetImageFromArray(model, isVector=False)
    sitk_img.SetSpacing(ConstPixelSpacing)
    sitk_img.SetOrigin(Origin)

    sitk.WriteImage(sitk_img, os.path.join(save_path, model_name))


def find_gap(a):
    # 删去毛刺点
    for i in range(1, len(a) - 2):
        if a[i - 1] == a[i + 1] and a[i - 1] < 0:
            a[i] = a[i - 1]

    # 计算一阶导数
    dst = np.zeros(len(a) - 1)
    for i in range(0, len(a) - 1):
        dst[i] = a[i] - a[i + 1]

    # 确定上下界所在位置
    upbond = np.where(dst > 430)[0]
    lowbond = np.where(dst < -430)[0]
    sub = []

    if len(upbond) != 0 and len(lowbond) != 0:
        # 剔除在第一个上界前的下界
        for j in range(len(lowbond)):
            if lowbond[j] < upbond[0]:
                sub.append(j)
        lowbond = np.delete(lowbond, sub)
        sub = []

        # 剔除在最后一个下界后的上界
        if len(lowbond) > 0:
            for e in range(len(upbond)):
                if upbond[e] > lowbond[len(lowbond) - 1]:
                    sub.append(e)
            upbond = np.delete(upbond, sub)

        if len(upbond) > len(lowbond):
            upbond = upbond[:len(lowbond)]
        else:
            lowbond = lowbond[:len(upbond)]

        return [upbond, lowbond]

    else:
        return []




def show_mhd_slices(fp):
    imageDirPath = fp

    seriesReader = sitk.ImageSeriesReader()
    seriesIds = seriesReader.GetGDCMSeriesIDs(imageDirPath)
    filenames = seriesReader.GetGDCMSeriesFileNames(imageDirPath, seriesIds[0])
    seriesReader.SetFileNames(filenames)
    image = seriesReader.Execute()
    img_arr = sitk.GetArrayViewFromImage(image)
    img_arr = np.append(img_arr[280:, :, :], img_arr[:280, :, :], axis=0)

    slc = img_arr[210:250, 260, 200:320]
    t, slc = cv.threshold(slc, 430, 2600, cv.THRESH_TOZERO)
    # 高斯滤波
    gaussian_blur = cv.GaussianBlur(slc, (1, 1), 0)

    # Roberts算子边缘识别
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv.filter2D(gaussian_blur, cv.CV_16S, kernelx)
    y = cv.filter2D(gaussian_blur, cv.CV_16S, kernely)
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    Roberts = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

    plt.subplot(1, 2, 1)
    plt.imshow(slc, cmap=plt.cm.Greys_r)
    plt.subplot(1, 2, 2)
    plt.imshow(Roberts, cmap=plt.cm.Greys_r)

    plt.show()


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
    ArrayDicom2 = img_arr
    ArrayDicom2 = ArrayDicom2[100:237, 100:289, 100:305]  # Neck10_3----195:237, 230:289, 213:305

    # 高通阈值化
    b = len(ArrayDicom2.flat)
    a = ArrayDicom2.reshape(b)
    for i in range(0, len(a)):
        if a[i] < 236:
            a[i] = a[0]
    ArrayDicom2 = a.reshape(ArrayDicom2.shape)

    # 将numpy数组通过SimpleITK转化为mhd和raw文件
    sitk_img = sitk.GetImageFromArray(ArrayDicom2, isVector=False)
    sitk_img.SetSpacing(ConstPixelSpacing)
    sitk_img.SetOrigin(Origin)
    print(sitk_img)
    sitk.WriteImage(sitk_img, os.path.join(SaveRawDicom, RawName))

    # ArrayDicom = np.append(img_arr[300:, :, :], img_arr[:300, :, :], axis=0)
    # t, ArrayDicom2 = cv.threshold(ArrayDicom, 230, 3000, cv.THRESH_TOZERO)
