Training:
python oneshot.py
python kshot.py

Evaluate:
python eval_oneshot.py

Testing:
python test_oneshot.py

Logs:
fix a bug in training codes, now input is 4-channel image (last channel is input label).
Add testing new classes codes.

TODO:
k-shot eval and test codes.

feature extractor is now VGG16_bn, last feature maps 7*7 and upsample by 2x gradually.
