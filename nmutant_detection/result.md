# CIFAR10

## Detect Adversrial Samples from Artifacts

BANDWIDTHS 0.26

| Attack | ROC-AUC | Accuracy | Presion | Recall |
| --- | --- | --- | --- | --- |
| cw | 0.8426 | 0.7795 | 0.7695 | 0.7980 |
| blackbox | 0.8716 | 0.8271 | 0.7887 | 0.8936 |
| fgsm | 0.7659 | 0.7001 | 0.7098 | 0.6770 |
| jsma | 0.8982 | 0.8583 | 0.8159 | 0.9252 |
| error | 0.8071 | 0.7554 | 0.7507 | 0.7649 |

## Bayesian Uncertainty Estimates

| Attack | ROC-AUC | Accuracy | Presion | Recall |
| --- | --- | --- | --- | --- |
| cw | 0.8411 | 0.7725 | 0.7685 | 0.7800 |
| blackbox | 0.8691 | 0.8218 | 0.7895 | 0.8777 |
| fgsm | 0.7783 | 0.7019 | 0.7146 | 0.6722 |
| jsma | 0.8980 | 0.8551 | 0.8178 | 0.9138 |
| error | 0.8127 | 0.7514 | 0.7541 | 0.7459 |

## Density Estimates

BANDWIDTHS 0.26

| Attack | ROC-AUC | Accuracy | Presion | Recall |
| --- | --- | --- | --- | --- |
| cw | 0.5682 | 0.5950 | 0.5661 | 0.8140 |
| blackbox | 0.5428 | 0.5851 | 0.5576 | 0.8245 |
| fgsm | 0.5623 | 0.5695 | 0.5493 | 0.7743 |
| jsma | 0.5627 | 0.6064 | 0.5732 | 0.8339 |
| error | 0.5603 | 0.5689 | 0.5478 | 0.7892 |

## Local Intrinsic Dimensionality

| Attack | ROC-AUC | Accuracy | Presion | Recall |
| --- | --- | --- | --- | --- |
| cw | 0.8766 | 0.8070 | 0.8259 | 0.7780 |
| blackbox | 0.9173 | 0.8457 | 0.8421 | 0.8511 |
| fgsm | 0.7924 | 0.7245 | 0.7380 | 0.6960 |
| jsma | 0.9501 | 0.8790 | 0.9038 | 0.8484 |
| error | 0.8400 | 0.7568 | 0.7861 | 0.7054 |

## our knor=0.10 mu=1.01 /100 0.5
| Attack | ROC-AUC | Accuracy | Presion | Recall |
| --- | --- | --- | --- | --- |
| cw | 0.8647 | 0.7835 | 0.7810 | 0.7849 |
| blackbox | 0.9417 | 0.8729 | 0.9576 | 0.8828 |
| fgsm | 0.8025 | 0.7509 | 0.7145 | 0.7707 |
| jsma | 0.9450 | 0.8694 | 0.9519 | 0.8171 |
| error | 0.8477 | 0.7838 | 0.7649 | 0.7949 |

| normal | jsma | cw | fgsm | blackbox | error |
| --- | --- | --- | --- | --- | --- |
| 1000 | 873 | 1000 | 809 | 118 | 370 |
| 214 | 831 | 781 | 578 | 113 | 283 |


# MNIST

## Detect Adversrial Samples from Artifacts

BANDWIDTHS 1.20

| Attack | ROC-AUC | Accuracy | Presion | Recall |
| --- | --- | --- | --- | --- |
| cw | 0.9841 | 0.9655 | 0.9595 | 0.9720 |
| blackbox | 0.9379 | 0.9345 | 0.9084 | 0.9666 |
| fgsm | 0.9587 | 0.9140 | 0.9107 | 0.9180 |
| jsma | 0.9900 | 0.9670 | 0.9651 | 0.9690 |
| error | 0.9864 | 0.9286 | 0.9500 | 0.9048 |

## Bayesian Uncertainty Estimates

| Attack | ROC-AUC | Accuracy | Presion | Recall |
| --- | --- | --- | --- | --- |
| cw | 0.9828 | 0.9640 | 0.9594 | 0.9690 |
| blackbox | 0.9359 | 0.9359 | 0.9129 | 0.9638 |
| fgsm | 0.9598 | 0.9160 | 0.9361 | 0.8930 |
| jsma | 0.9893 | 0.9665 | 0.9651 | 0.9680 |
| error | 0.9932 | 0.9524 | 0.9524 | 0.9524 | 

## Density Estimates

BANDWIDTHS 1.20

| Attack | ROC-AUC | Accuracy | Presion | Recall |
| --- | --- | --- | --- | --- |
| cw | 0.8919 | 0.8170 | 0.8090 | 0.8300 |
| blackbox | 0.8561 | 0.8259 | 0.7647 | 0.9415 |
| fgsm | 0.8764 | 0.8370 | 0.7910 | 0.9160 |
| jsma | 0.9076 | 0.7825 | 0.8221 | 0.7210 |
| error | 0.7596 | 0.6905 | 0.6818 | 0.7143 |

## Local Intrinsic Dimensionality

| Attack | ROC-AUC | Accuracy | Presion | Recall |
| --- | --- | --- | --- | --- |
| cw | 0.9682 | 0.9220 | 0.9137 | 0.9320 |
| blackbox | 0.9764 | 0.9526 | 0.9265 | 0.9833 |
| fgsm | 0.9877 | 0.9510 | 0.9413 | 0.9620 |
| jsma | 0.9727 | 0.9255 | 0.9159 | 0.9370 |
| error | 0.5283 | 0.5000 | 0.0000 | 0.0000 |

## our mu=1.1 knor=0.016 /100 0.5
| Attack | ROC-AUC | Accuracy | Presion | Recall |
| --- | --- | --- | --- | --- |
| cw | 0.9970 | 0.9590 | 0.9950 | 0.9282 |
| blackbox | 0.9930 | 0.9559 | 0.9890 | 0.9276 |
| fgsm | 0.9671 | 0.9030 | 0.8830 | 0.9198 |
| jsma | 0.9919 | 0.9600 | 0.9970 | 0.9283 |
| error | 1.0000 | 0.9762 | 1.0000 | 0.9545 |

| normal | jsma | cw | fgsm | blackbox | error |
| --- | --- | --- | --- | --- | --- |
| 1000 | 1000 | 1000 | 1000 | 272 | 21 |
| 77 | 997 | 995 | 883 | 269 | 21 |
