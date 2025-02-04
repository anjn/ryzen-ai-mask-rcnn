import argparse
import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod, quantize_static
import vai_q_onnx
import glob

input_model_path = "./model/maskrcnn_backbone_rpn.onnx"
output_model_path = "./model/maskrcnn_backbone_rpn_quant.onnx"

class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, args):
        super().__init__()
        self.paths = glob.glob(f"./model_io/*/{args.data}.npy")
        self.index = 0
        self.input_name = args.name
        self.batch_size = args.batch_size
        self.max_samples = args.max_samples
        print(self.paths)

    def get_next(self):
        if self.index < len(self.paths) and self.index < self.max_samples:
            next_index = self.index
            self.index += 1

            data = np.load(self.paths[next_index])

            # バッチサイズが指定されていなかったら最初のデータに合わせる
            if self.batch_size < 1:
                self.batch_size = data.shape[0]
                print(f"{self.batch_size=}")

            # 量子化するときは各データのshapeを合わせる必要がある
            if data.shape[0] < self.batch_size:
                print(f"{data.shape=}, {self.batch_size=}")
                new_data = np.zeros((self.batch_size, *data.shape[1:]), dtype=np.float32)
                new_data[0:data.shape[0]] = data
                data = new_data
            elif data.shape[0] > self.batch_size:
                data = data[0:self.batch_size]

            return {self.input_name: data}
        else:
            return None

def main(args):
    data_reader = MyCalibrationDataReader(args)
    
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
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--data', type=str, default='input_backbone')
    parser.add_argument('--name', type=str, default='input')
    parser.add_argument('--batch_size', type=int, default=-1,
                      help='Batch size')
    parser.add_argument('--max_samples', type=int, default=8,
                      help='Maximum number of samples to process (default: all samples)')
    args = parser.parse_args()

    main(args)
