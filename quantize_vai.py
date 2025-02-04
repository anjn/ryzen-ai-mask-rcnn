import argparse
import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod, quantize_static
import vai_q_onnx
import glob

input_model_path = "./model/maskrcnn_backbone_rpn.onnx"
output_model_path = "./model/maskrcnn_backbone_rpn_quant.onnx"

class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data):
        super().__init__()
        self.paths = glob.glob(f"./model_io/*/{data}.npy")
        self.index = 0
        print(self.paths)

    def get_next(self):
        if self.index < len(self.paths) and self.index < 8:
            next_index = self.index
            self.index += 1
            return {'input': np.load(self.paths[next_index])}
        else:
            return None

def main(args):
    data_reader = MyCalibrationDataReader(args.data)
    
    vai_q_onnx.quantize_static(
        args.input,
        args.output,
        data_reader,
        quant_format = vai_q_onnx.QuantFormat.QDQ,
        calibrate_method = vai_q_onnx.PowerOfTwoMethod.MinMSE,
        activation_type = vai_q_onnx.QuantType.QUInt8,
        weight_type = vai_q_onnx.QuantType.QInt8,
        enable_ipu_cnn = True, 
        extra_options = {'ActivationSymmetric': True} 
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantize onnx model')
    parser.add_argument('data')
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    main(args)
