# GAN(Generative Adversarial Network) with 1D Gaussian distribution

## Reference
1. http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
2. http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html
3. https://github.com/hwalsuklee/tensorflow-GAN-1d-gaussian-ex
4. https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f


### In tensorflow - 1D Gaussian(mu, sigma = 1.0, 1.5)

Initial decision boundary after pre-training:  

![1D-initial-tensorflow](./assets/1D-initial-tensorflow.png)

Training loss:  

![1D-loss-tensorflow](./assets/1D-loss-tensorflow.png)

Result:  

![1D-result-tensorflow](./assets/1D-result-tensorflow.png)


### In pytorch - 1D Gaussian(mu, sigma = 1.0, 1.0)

Initial decision boundary after pre-training:  

![1D-initial-pytorch](./assets/1D-initial-pytorch.png)

Training loss:  

![1D-loss-pytorch](./assets/1D-loss-pytorch.png)

Result:  

![1D-result-pytorch](./assets/1D-result-pytorch.png)


### In tensorflow - 1D Mixture of Gaussian(mu1, sigma1, mu2, sigma2 = -3.0, 1.0, 3.0, 1.0)

Initial decision boundary after pre-training:  

![1D-Mixture-initial-tensorflow](./assets/1D-Mixture-initial-tensorflow.png)

Training loss:  

![1D-Mixture-loss-tensorflow](./assets/1D-Mixture-loss-tensorflow.png)

Result:  

![1D-result-tensorflow](./assets/1D-Mixture-result-tensorflow.png)


### Note on implementation

* Use weight initialization with truncated normal
* Don't use activation function on generator and no normalization on input
* Use optimzer with gradient descent
