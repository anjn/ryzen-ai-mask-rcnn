
import numpy as np
import os
import torch
import torchvision
from torch.utils.data import DataLoader
import onnx
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod, quantize_static
import vai_q_onnx
import glob

input_model_path = "./model/maskrcnn_backbone_rpn.onnx"
output_model_path = "./model/maskrcnn_backbone_rpn_quant.onnx"

class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self):
        super().__init__()
        self.paths = glob.glob("./model_io/*/input_backbone.npy")
        self.index = 0
        print(self.paths)

    def get_next(self) -> dict:
        if self.index < len(self.paths):
            return np.load(self.paths[self.index])
        else:
            return None

data_reader = MyCalibrationDataReader()

vai_q_onnx.quantize_static(
    input_model_path,
    output_model_path,
    data_reader,
    quant_format = vai_q_onnx.QuantFormat.QDQ,
    calibrate_method = vai_q_onnx.PowerOfTwoMethod.MinMSE,
    activation_type = vai_q_onnx.QuantType.QUInt8,
    weight_type = vai_q_onnx.QuantType.QInt8,
    enable_ipu_cnn = True, 
    extra_options = {'ActivationSymmetric': True} 
)
