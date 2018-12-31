# shake-shake-tensorflow
Simple Code Implementation of ["Shake-Shake Regularization"](https://arxiv.org/pdf/1705.07485.pdf) using TensorFlow.
![](https://github.com/hoya012/shake-shake-tensorflow/blob/master/assets/shake-shake.PNG)


*Last update : 2018/12/31*

## Contributor
* hoya012

## Paper Review(Korean only)
I wrote [some posting](http://research.sualab.com/machine-learning/computer-vision/2018/06/28/shake-shake-regularization-review.html) about this paper review. 

## Requirements
Python 3.5
```
numpy >= 1.13.3
matplotlib >= 2.0.2
scikit-learn >= 0.19.1
scikit-image >= 0.13.0
tensorflow-gpu == 1.4.1
```

## Usage
For training, testing, i used `CIFAR-10` Dataset and you can simply run my code.

```
python train.py
```

## Result
This is my result of `CIFAR-10` dataset and is similar with result of original paper.

| - | Original paper | My implementation  |
| - | :-: | :-: | 
| Accuracy | 96.45% | 96.33% |  

This is plot of my learning curve. Blue line means `accuracy of training set` and red line means `accuracy of validation set`. Almost, we need 1800 or more epochs for saturation.

![](https://github.com/hoya012/shake-shake-tensorflow/blob/master/assets/plot.PNG)


## Reference
- [SUALAB classification code implementation](https://github.com/sualab/asirra-dogs-cats-classification)

