import read2
import vtk_dicom_read
import SimpleITK as sitk
import matplotlib.pyplot as plt
import vtk_curvature_show

filepath = "G:/CTs/Neck10_3"
save_path = filepath


#read2.show_mhd_slices(kfp)
#read2.save_mhd(kfp, kfp, "Raw")

read2.build_model(filepath, save_path, "Model")

vtk_dicom_read.show(filepath + "/Model.mhd")
#vtk_curvature_show.show(workfp+"/Model.mhd")


