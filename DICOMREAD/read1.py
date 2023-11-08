import read2
import vtk_dicom_read
import SimpleITK as sitk
import matplotlib.pyplot as plt
import vtk_curvature_show

fp = "G:/CTs/0000.dcm"
ifp = "G:/CTs/sample.mhd"
kfp = "G:/CTs/Neck10_3"


#read2.show_mhd_slices(kfp)
#read2.save_mhd(kfp, kfp, "Raw")

read2.build_model(kfp, kfp, "Model")

vtk_dicom_read.show(kfp+"/Model.mhd")
#vtk_curvature_show.show(workfp+"/Model.mhd")


