REM 環境に合わせて変更してください
set coco_ann_file=C:/Users/juna/Work/dataset/coco/annotations/instances_val2017.json
set coco_img_dir=C:/Users/juna/Work/dataset/coco/val2017
REM 64の倍数にしなくてよい？
set height=544
set width=960
set data_id=397133

REM Clean
rmdir /s /q model
rmdir /s /q model_io
rmdir /s /q cache

pip install -r ./requirements.txt

python evaluate_maskrcnn.py --ann_file %coco_ann_file% --img_dir %coco_img_dir% --max_samples 8 --fix_input_size %height% %width% --save_io

python custom_maskrcnn_model.py --input %data_id% --export

python simplify.py ./model/maskrcnn_backbone.onnx ./model/maskrcnn_backbone.onnx
python simplify.py ./model/maskrcnn_box_predictor.onnx ./model/maskrcnn_box_predictor.onnx
python simplify.py ./model/maskrcnn_mask_predictor.onnx ./model/maskrcnn_mask_predictor.onnx

sam4onnx -if ./model/maskrcnn_backbone.onnx -of ./model/maskrcnn_backbone.onnx -on /backbone/fpn/extra_blocks/MaxPool --attributes kernel_shape int64 [3,3] --attributes pads int64 [1,1,1,1]

python custom_maskrcnn_model.py --input %data_id% --onnx_backbone ./model/maskrcnn_backbone.onnx --onnx_box_predictor ./model/maskrcnn_box_predictor.onnx --onnx_mask_predictor ./model/maskrcnn_mask_predictor.onnx

python quantize_vai.py ./model/maskrcnn_backbone.onnx ./model/maskrcnn_backbone_quant.onnx  --data input_backbone --name input --max_samples 6
python quantize_vai.py ./model/maskrcnn_box_predictor.onnx ./model/maskrcnn_box_predictor_quant.onnx --data box_features --name box_features --batch_size 1000
python quantize_vai.py ./model/maskrcnn_mask_predictor.onnx ./model/maskrcnn_mask_predictor_quant.onnx --data mask_features --name mask_features --batch_size 100

python custom_maskrcnn_model.py --input %data_id% --onnx_backbone ./model/maskrcnn_backbone_quant.onnx --onnx_ep cpu

python custom_maskrcnn_model.py --input %data_id% --onnx_backbone ./model/maskrcnn_backbone_quant.onnx --onnx_ep vai --warm_up --test_num 10

python evaluate_maskrcnn.py --ann_file %coco_ann_file% --img_dir %coco_img_dir% --max_samples 10 --fix_input_size %height% %width% --onnx_backbone ./model/maskrcnn_backbone_quant.onnx --onnx_ep vai
