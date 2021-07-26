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
- pretrained = False
- batch_size = 80
- epochs = 30
### Evaluation
```bash
Top 1 Accuracy of class  0 is 250/368  67.93%
Top 1 Accuracy of class  1 is  96/148  64.86%
Top 1 Accuracy of class  2 is 209/231  90.48%
Top 1 Accuracy of class  3 is 309/500  61.80%
Top 1 Accuracy of class  4 is 265/335  79.10%
Top 1 Accuracy of class  5 is 209/287  72.82%
Top 1 Accuracy of class  6 is 347/432  80.32%
Top 1 Accuracy of class  7 is 134/147  91.16%
Top 1 Accuracy of class  8 is  81/ 96  84.38%
Top 1 Accuracy of class  9 is 228/303  75.25%
Top 1 Accuracy of class 10 is 400/500  80.00%
Top 1 accuracy of the network on the 3347 test images: 2528/3347  75.53 %
75.53032566477442

Top 3 Accuracy of class  0 is 344/368  93.48%
Top 3 Accuracy of class  1 is 132/148  89.19%
Top 3 Accuracy of class  2 is 225/231  97.40%
Top 3 Accuracy of class  3 is 463/500  92.60%
Top 3 Accuracy of class  4 is 317/335  94.63%
Top 3 Accuracy of class  5 is 266/287  92.68%
Top 3 Accuracy of class  6 is 415/432  96.06%
Top 3 Accuracy of class  7 is 144/147  97.96%
Top 3 Accuracy of class  8 is  91/ 96  94.79%
Top 3 Accuracy of class  9 is 281/303  92.74%
Top 3 Accuracy of class 10 is 465/500  93.00%
Top 3 accuracy of the network on the 3347 test images: 3143/3347  93.90 %
93.90498954287422
```