## 概要

バックボーンや予測器など単純なCNN構造の部分と、制御構造やリスト操作などNPUでの実行に不向きな処理を含む部分を分離し、CNN部分のみをNPUで実行します。

## Mask-RCNNをNPUで実行する手順

> [!IMPORTANT]
> 以下はConda環境`ryzen-ai-1.3.0`で実行してください。

### 必要なパッケージをインストールする

```sh
pip install -r ./requirements.txt
```

### COCOデータセットをダウンロードし展開する

- http://images.cocodataset.org/zips/val2017.zip
- http://images.cocodataset.org/annotations/annotations_trainval2017.zip

### モデルを実行してキャリブレーションデータを生成する

データセットのパスを指定して、8サンプル分のキャリブレーションデータを生成します。次の例では入力画像サイズを高さ544、幅960に固定しています。画像サイズの固定は必須です。

```sh
python evaluate_maskrcnn.py --ann_file SOMEWHERE/instances_val2017.json --img_dir SOMEWHERE/val2017 --max_samples 8 --fix_input_size 544 960 --save_io
```

`model_io`ディレクトリ以下にモデルの入出力や中間データが保存されます。

### モデルからNPU実行する部分を切り出して出力する

キャリブレーションデータのうちひとつを入力として推論を実行しつつ、ONNXモデルを`model`ディレクトリ以下に出力します。

```sh
python custom_maskrcnn_model.py --input 397133 --export
```

### ONNXモデルを最適化する

```sh
python simplify.py ./model/maskrcnn_backbone.onnx ./model/maskrcnn_backbone.onnx
python simplify.py ./model/maskrcnn_box_predictor.onnx ./model/maskrcnn_box_predictor.onnx
python simplify.py ./model/maskrcnn_mask_predictor.onnx ./model/maskrcnn_mask_predictor.onnx
```

### バックボーンモデルのMaxPoolをNPU実行できるように修正する

カーネルサイズ1x1のMaxPoolはNPUで実行不可のため3x3に変更します。出力のshapeが変わってしまわないようにpadsも変更します。
（カーネルサイズを変更してしまうと処理結果に影響がありそうに思いますが精度への影響はほとんどない模様）

```sh
sam4onnx -if ./model/maskrcnn_backbone.onnx -of ./model/maskrcnn_backbone.onnx -on /backbone/fpn/extra_blocks/MaxPool --attributes kernel_shape int64 [3,3] --attributes pads int64 [1,1,1,1]
```

### 変換したONNXモデルを使って正しく推論できることを確認する

キャリブレーションデータを期待値として、出力の差分の絶対値が一定値以下であれば正しく推論できているとします。

```sh
python custom_maskrcnn_model.py --input 397133 --onnx_backbone ./model/maskrcnn_backbone.onnx --onnx_box_predictor ./model/maskrcnn_box_predictor.onnx --onnx_mask_predictor ./model/maskrcnn_mask_predictor.onnx
```

### モデルを量子化する

各ONNXモデルの入力に相当する中間データをキャリブレーションデータとして使用してモデルを量子化します。メモリ使用率が100%近くなってしまう場合は`--max_samples`オプションの数値を調整してください。

```sh
python quantize_vai.py ./model/maskrcnn_backbone.onnx ./model/maskrcnn_backbone_quant.onnx  --data input_backbone --name input --max_samples 6
python quantize_vai.py ./model/maskrcnn_box_predictor.onnx ./model/maskrcnn_box_predictor_quant.onnx --data box_features --name box_features --batch_size 1000
python quantize_vai.py ./model/maskrcnn_mask_predictor.onnx ./model/maskrcnn_mask_predictor_quant.onnx --data mask_features --name mask_features --batch_size 100
```

### 量子化したモデルを使用してCPUで推論する

量子化の影響により元のモデルとは異なる出力となります。`--onnx_ep`オプションにcpuを指定しているためCPUで実行されます。量子化による影響で検出数も変化するため期待値比較がエラーになる場合があります。

```sh
python custom_maskrcnn_model.py --input 397133 --onnx_backbone ./model/maskrcnn_backbone_quant.onnx --onnx_box_predictor ./model/maskrcnn_box_predictor_quant.onnx --onnx_mask_predictor ./model/maskrcnn_mask_predictor_quant.onnx --onnx_ep cpu
```

### 量子化したモデルを使用してNPUで推論する

10回実行したときの平均の実行時間が出力されます。mask_predictorの実行時間は検出数によって変化します。

```sh
python custom_maskrcnn_model.py --input 397133 --onnx_backbone ./model/maskrcnn_backbone_quant.onnx --onnx_box_predictor ./model/maskrcnn_box_predictor_quant.onnx --onnx_mask_predictor ./model/maskrcnn_mask_predictor_quant.onnx --onnx_ep vai --warm_up --test_num 10
```

現状、NPUでは推論をバッチで実行することができないようです。box_predictor、mask_predictorでは入力を一つずつ推論することになりオーバーヘッドが大きく、CPU実行より時間がかかります。backboneのみNPU実行する場合は次のようにします。

```sh
python custom_maskrcnn_model.py --input 397133 --onnx_backbone ./model/maskrcnn_backbone_quant.onnx --onnx_ep vai --warm_up --test_num 10
```

> [!NOTE]
> NPU推論する際にモデルをコンパイルする処理が自動で走るため初回は時間がかかります。コンパイルした結果は`cache`ディレクトリ以下にキャッシュされ、次回以降はコンパイルは省略されます。古いキャッシュが残っている場合や`enable_analyzer`を切り替えたときは手動でキャッシュを削除してください。

### 量子化したモデルの精度を評価する

```sh
python evaluate_maskrcnn.py --ann_file SOMEWHERE/instances_val2017.json --img_dir SOMEWHERE/val2017  --fix_input_size 544 960 --max_samples 100 --onnx_backbone ./model/maskrcnn_backbone_quant.onnx --onnx_ep vai
```

すべてのValidationデータで評価を行う場合は、`--max_samples`オプションに5000を指定します。

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

