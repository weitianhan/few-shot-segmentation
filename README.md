Training:
python oneshot.py
python kshot.py

Evaluate:
python eval_oneshot.py

Testing:
python test_oneshot.py

Logs:
Use COCO dataset in training.
Finish k-shot eval and test codes.

TODO:
Replace Resnet-101 as backbone feature extractor

feature extractor is now VGG16_bn, last feature maps 7*7 and upsample by 2x gradually.
