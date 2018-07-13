Training:
python oneshot.py
python kshot.py

Testing:
python test_oneshot.py

Logs:
Change upsampling in kshot.py to coarse-to-fine upsampling.

feature extractor is now VGG16_bn, last feature maps 7*7 and upsample by 2x gradually.
