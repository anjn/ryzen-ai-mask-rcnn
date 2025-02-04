## 概要

バックボーンや予測器など単純なCNN構造の部分と、制御構造やリスト操作などNPUでの実行に不向きな処理を含む部分を分離し、CNN部分のみをNPUで実行します。

## Mask-RCNNをNPUで実行する手順

> [!IMPORTANT]
> 以下はConda環境`ryzen-ai-1.3.0`で実行してください。

### 必要なパッケージをインストールする

```sh
pip install -r ./requirements.txt
```

### モデルを実行してキャリブレーションデータを生成する

データセットのパスを指定して、8サンプル分のキャリブレーションデータを生成します。

```sh
python evaluate_maskrcnn.py --ann_file SOMEWHERE/instances_val2017.json --img_dir SOMEWHERE/val2017 --batch_size 1 --num_workers 1 --max_samples 8 --device cpu --save_io
```

`model_io`ディレクトリ以下にモデルの入出力や中間データが保存されます。

### モデルからNPU実行する部分を切り出して出力する

キャリブレーションデータのうちひとつを入力として推論を実行しつつ、ONNXモデルを`model`ディレクトリ以下に出力します。

```sh
python custom_maskrcnn_model.py --device cpu --input 87038 --export
```

### ONNXモデルを最適化する

```sh
python simplify.py ./model/maskrcnn_backbone.onnx ./model/maskrcnn_backbone.onnx
python simplify.py ./model/maskrcnn_box_predictor.onnx ./model/maskrcnn_box_predictor.onnx
python simplify.py ./model/maskrcnn_mask_predictor.onnx ./model/maskrcnn_mask_predictor.onnx
```

### バックボーンモデルのMaxPoolをNPU実行できるように修正する

カーネルサイズ1x1のMaxPoolはNPUで実行不可のため、3x3に変更する。出力のshapeが変わってしまわないようにpadsも変更する。
（カーネルサイズを変更してしまうと処理結果に影響がありそうに思うが、精度への影響はほとんどない模様）

```sh
sam4onnx -if ./model/maskrcnn_backbone.onnx -of ./model/maskrcnn_backbone.onnx -on /backbone/fpn/extra_blocks/MaxPool --attributes kernel_shape int64 [3,3] --attributes pads int64 [1,1,1,1]
```

### 変換したONNXモデルを使って正しく推論できることを確認する

キャリブレーションデータを期待値として、出力の差分の絶対値が一定値以下であれば正しく推論できているとする。

```sh
python custom_maskrcnn_model.py --device cpu --input 87038 --onnx_backbone ./model/maskrcnn_backbone.onnx --onnx_box_predictor ./model/maskrcnn_box_predictor.onnx --onnx_mask_predictor ./model/maskrcnn_mask_predictor.onnx
```

### モデルを量子化する

各ONNXモデルの入力に相当する中間データをキャリブレーションデータとして使用してモデルを量子化します。

```sh
python quantize_vai.py input_backbone ./model/maskrcnn_backbone.onnx ./model/maskrcnn_backbone_quant.onnx
python quantize_vai.py box_features ./model/maskrcnn_box_predictor.onnx ./model/maskrcnn_box_predictor_quant.onnx
python quantize_vai.py mask_features ./model/maskrcnn_mask_predictor.onnx ./model/maskrcnn_mask_predictor_quant.onnx
```

### 量子化したモデルを使用してCPUで推論する

量子化の影響により元のモデルとは異なる出力となります。`--onnx_ep`オプションにcpuを指定しているためCPUで実行されます。10回実行したときの平均の実行時間が出力されます。mask_predictorの実行時間は検出数によって変化します。

```sh
python custom_maskrcnn_model.py --device cpu --input 87038 --onnx_backbone ./model/maskrcnn_backbone_quant.onnx --onnx_box_predictor ./model/maskrcnn_box_predictor_quant.onnx --onnx_mask_predictor ./model/maskrcnn_mask_predictor_quant.onnx --onnx_ep cpu --warm_up --test_num 10
```

### 量子化したモデルを使用してNPUで推論する

```sh
python custom_maskrcnn_model.py --device cpu --input 87038 --onnx_backbone ./model/maskrcnn_backbone_quant.onnx --onnx_box_predictor ./model/maskrcnn_box_predictor_quant.onnx --onnx_mask_predictor ./model/maskrcnn_mask_predictor_quant.onnx --onnx_ep npu --warm_up --test_num 10
```

### 量子化したモデルの精度を評価する

```sh
python evaluate_maskrcnn.py --ann_file SOMEWHERE/instances_val2017.json --img_dir SOMEWHERE/val2017 --batch_size 1 --num_workers 1 --max_samples 100 --device cpu --onnx_backbone ./model/maskrcnn_backbone_quant.onnx --onnx_box_predictor ./model/maskrcnn_box_predictor_quant.onnx --onnx_mask_predictor ./model/maskrcnn_mask_predictor_quant.onnx --onnx_ep npu
```

`--max_samples`オプションを削除するとすべてのValidationデータで評価を行います。

## 参考

元のモデルに5000枚のValidationデータを入力したときの精度は以下です。

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.474
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.679
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.519
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.309
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.512
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.611
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.367
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.589
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.618
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.445
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.649
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.764

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.417
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.651
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.449
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.245
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.453
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.568
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.337
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.527
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.551
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.363
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.586
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.717

Results:
Box mAP: 0.4739
Mask mAP: 0.4172
```

