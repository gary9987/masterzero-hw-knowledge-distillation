# masterzero-hw-knowledge-distillation

## Teacher Model
- torchvision.models.resnet18(pretrained=True)
- 40 epochs, no more updates after reaching the 28th epoch.

### Fix some layers
```python
for k, v in net.named_parameters():
    print(k)
    if (k == 'conv1.weight' or k == 'bn1.weight' or k == 'bn1.bias'):
        v.requires_grad = False
    if (k[0:6] == 'layer1' or k[0:6] == 'layer2'):
        v.requires_grad = False
```
### Evaluation
```bash
Top 1 Accuracy of class  0 is 312/368  84.78%
Top 1 Accuracy of class  1 is 135/148  91.22%
Top 1 Accuracy of class  2 is 223/231  96.54%
Top 1 Accuracy of class  3 is 420/500  84.00%
Top 1 Accuracy of class  4 is 309/335  92.24%
Top 1 Accuracy of class  5 is 257/287  89.55%
Top 1 Accuracy of class  6 is 397/432  91.90%
Top 1 Accuracy of class  7 is 142/147  96.60%
Top 1 Accuracy of class  8 is  94/ 96  97.92%
Top 1 Accuracy of class  9 is 286/303  94.39%
Top 1 Accuracy of class 10 is 491/500  98.20%
Top 1 accuracy of the network on the 3347 test images: 3066/3347  91.60 %
91.60442187033163

Top 3 Accuracy of class  0 is 354/368  96.20%
Top 3 Accuracy of class  1 is 139/148  93.92%
Top 3 Accuracy of class  2 is 230/231  99.57%
Top 3 Accuracy of class  3 is 489/500  97.80%
Top 3 Accuracy of class  4 is 332/335  99.10%
Top 3 Accuracy of class  5 is 282/287  98.26%
Top 3 Accuracy of class  6 is 428/432  99.07%
Top 3 Accuracy of class  7 is 146/147  99.32%
Top 3 Accuracy of class  8 is  95/ 96  98.96%
Top 3 Accuracy of class  9 is 298/303  98.35%
Top 3 Accuracy of class 10 is 498/500  99.60%
Top 3 accuracy of the network on the 3347 test images: 3291/3347  98.33 %
98.3268598745145
```

## MobileNet_V3
- Training without teacher.
### Evaluation
```bash
Top 1 Accuracy of class  0 is 294/368  79.89%
Top 1 Accuracy of class  1 is 121/148  81.76%
Top 1 Accuracy of class  2 is 210/231  90.91%
Top 1 Accuracy of class  3 is 432/500  86.40%
Top 1 Accuracy of class  4 is 289/335  86.27%
Top 1 Accuracy of class  5 is 241/287  83.97%
Top 1 Accuracy of class  6 is 383/432  88.66%
Top 1 Accuracy of class  7 is 144/147  97.96%
Top 1 Accuracy of class  8 is  91/ 96  94.79%
Top 1 Accuracy of class  9 is 265/303  87.46%
Top 1 Accuracy of class 10 is 483/500  96.60%
Top 1 accuracy of the network on the 3347 test images: 2953/3347  88.23 %
88.2282641171198

Top 3 Accuracy of class  0 is 360/368  97.83%
Top 3 Accuracy of class  1 is 140/148  94.59%
Top 3 Accuracy of class  2 is 230/231  99.57%
Top 3 Accuracy of class  3 is 484/500  96.80%
Top 3 Accuracy of class  4 is 325/335  97.01%
Top 3 Accuracy of class  5 is 273/287  95.12%
Top 3 Accuracy of class  6 is 425/432  98.38%
Top 3 Accuracy of class  7 is 145/147  98.64%
Top 3 Accuracy of class  8 is  95/ 96  98.96%
Top 3 Accuracy of class  9 is 291/303  96.04%
Top 3 Accuracy of class 10 is 497/500  99.40%
Top 3 accuracy of the network on the 3347 test images: 3265/3347  97.55 %
97.55004481625336

Process finished with exit code 0
```