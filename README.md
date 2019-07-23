# Tensorflow-SinusitisDetCls
+ Project for Maxillary Sinusitis Detection & Classification

## models.py
+ ResNetDouble3_128 (classifier)
+ ResNetDouble4_448 (detector)
+ SeResNetDouble3_128 (classifier)
+ SeResNetDouble4_448 (detector)
+ PatchProcessor (bridge btw det & cls)

## datasets.py
 + prepare information for image & label dataset
 + read and augment DICOM images
 + return dataset object
 
## metrics.py
 + prob, pred, acc, iou
 + loss functions
 + sens, spec, auc, roc curve
 
## visualizer.py
 + plot bounding box and labels
