
sam4onnx \
    -if ./model/maskrcnn_backbone.onnx \
    -of ./model/maskrcnn_backbone.onnx \
    -on /backbone/fpn/extra_blocks/MaxPool \
    --attributes kernel_shape int64 [3,3] \
    --attributes pads int64 [1,1,1,1]

