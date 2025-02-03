
バックボーンやヘッドなど単純なCNN構造の処理と、制御構造やリスト操作などNPUでの実行に不向きな処理を分離します。例えば以下のような処理はNPUには不向きです。

- [MultiScaleRoIAlign](https://github.com/pytorch/vision/blob/867521ec82c78160b16eec1c3a02d4cef93723ff/torchvision/ops/poolers.py#L230)

ONNX変換時にバッチサイズを1に固定します。次の理由があります。

- Resize (interpolate) オペレータがあり、バッチサイズが不定のとき正しく形状を推論できません。
- NPU では複数のコアを並列に動作させたときに最も効率が良くなります。バッチを大きくしても効率は上がりません（要確認）

HWについては固定することを強くおすすめします。

作業方針
- モデル全体をONNXで出力
- 前処理を削除（NPU推論時にはnumpyで処理する）
- backbone+rpn、boxまで、box、maskまで、mask、残り、の5つに分解する
- 形状を確定させる、もしくはバッチにする

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

カーネルサイズ1x1のMaxPoolはNPUで実行不可のため、3x3に変更する。出力のshapeが変わってしまわないようにpadsも変更する。
カーネルサイズを変更してしまうと処理結果に影響がありそうに思うが、精度への影響はほとんどない様子。
```
sam4onnx -if ./model/maskrcnn_backbone_rpn_quant.onnx -of ./model/maskrcnn_backbone_rpn_quant_fix.onnx -on /backbone/fpn/extra_blocks/MaxPool --attributes kernel_shape int64 [3,3] --attributes pads int64 [1,1,1,1]
```
