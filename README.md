# masterzero-hw-knowledge-distillation

## Teacher Model
- `teacher.py`
- torchvision.models.resnet18(pretrained=True)
- 40 epochs, no more updates after reaching the 28th epoch.
- batch_size = 80

### Fix some layers
```python
for k, v in net.named_parameters():
    print(k)
    if (k == 'conv1.weight' or k == 'bn1.weight' or k == 'bn1.bias'):
        v.requires_grad = False
    if (k[0:6] == 'layer1' or k[0:6] == 'layer2'):
        v.requires_grad = False
```
### Sampoler
- ImbalancedDatasetSampler(trainset, num_samples=25000)
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
## Student Model
- `student.py`
- Training with teacher
- torchvision.models.mobilenet_v3_small(pretrained=False)
- batch_size = 80
- epochs = 30

### Sampler
- ImbalancedDatasetSampler(trainset, num_samples=18000)
### KD Loss
```python
def KdLoss(output, target, soft_targer, alpha, T):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KD_loss = nn.KLDivLoss()(nn.functional.log_softmax(output / T, dim=1),
                             nn.functional.softmax(soft_targer / T, dim=1)) * (alpha * T * T) + \
              nn.functional.cross_entropy(output, target) * (1. - alpha)

    return KD_loss
```
### Evaluation
```bash
Top 1 Accuracy of class  0 is 213/368  57.88%
Top 1 Accuracy of class  1 is 103/148  69.59%
Top 1 Accuracy of class  2 is 209/231  90.48%
Top 1 Accuracy of class  3 is 322/500  64.40%
Top 1 Accuracy of class  4 is 268/335  80.00%
Top 1 Accuracy of class  5 is 234/287  81.53%
Top 1 Accuracy of class  6 is 316/432  73.15%
Top 1 Accuracy of class  7 is 123/147  83.67%
Top 1 Accuracy of class  8 is  85/ 96  88.54%
Top 1 Accuracy of class  9 is 254/303  83.83%
Top 1 Accuracy of class 10 is 460/500  92.00%
Top 1 accuracy of the network on the 3347 test images: 2587/3347  77.29 %
77.29309829698238

Top 3 Accuracy of class  0 is 322/368  87.50%
Top 3 Accuracy of class  1 is 130/148  87.84%
Top 3 Accuracy of class  2 is 225/231  97.40%
Top 3 Accuracy of class  3 is 469/500  93.80%
Top 3 Accuracy of class  4 is 309/335  92.24%
Top 3 Accuracy of class  5 is 278/287  96.86%
Top 3 Accuracy of class  6 is 415/432  96.06%
Top 3 Accuracy of class  7 is 141/147  95.92%
Top 3 Accuracy of class  8 is  92/ 96  95.83%
Top 3 Accuracy of class  9 is 295/303  97.36%
Top 3 Accuracy of class 10 is 486/500  97.20%
Top 3 accuracy of the network on the 3347 test images: 3162/3347  94.47 %
94.47266208544966
```
## MobileNet_V3
- `mobilenet_v3.py`
- Training without teacher
- torchvision.models.mobilenet_v3_small(pretrained=False)
- batch_size = 80
- epochs = 30
### Sampler
- ImbalancedDatasetSampler(trainset, num_samples=18000)
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

## Reference
- [???????????????Knowledge Distillation??????Pytorch??????????????????](https://blog.csdn.net/shi2xian2wei2/article/details/84570620)
- [peterliht/knowledge-distillation-pytorch](https://github.com/peterliht/knowledge-distillation-pytorch/blob/ef06124d67a98abcb3a5bc9c81f7d0f1f016a7ef/model/net.py#L100)
- [Pytorch??????????????????????????? torch.save()???torch.load()???torch.nn.Module.load_state_dict()](https://blog.csdn.net/weixin_40522801/article/details/106563354)