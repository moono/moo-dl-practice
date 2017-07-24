# Weight init schemes & batch normalization
## Trying MNIST data set on one layer with different weight initialization scheme and batch normalization

### Train Accuracies

* With or without batch normalization (normal init)
![w-wo](assets/accuracy-with-without-batch-norm.PNG)

* With batch normalization
![with](assets/accuracy-with-batch-norm.PNG)

* Without batch normalization
![without](assets/accuracy-without-batch-norm.PNG)

### Logits and Activation layer

* With normal initialization

![normal](assets/normal-histogram.PNG)

* With truncated normal initialization

![truncated normal](assets/truncated-histogram.PNG)

* With xavier initialization

![xavier](assets/xavier-histogram.PNG)

* With he initialization

![he](assets/he-histogram.PNG)

## Reference code
[hwalsuk.lee](https://github.com/hwalsuklee/tensorflow-mnist-MLP-batch_normalization-weight_initializers)
