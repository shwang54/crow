# [Conditional Recurrent Flow: Conditional Generation of Longitudinal Samples with Applications to Neuroimaging](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hwang_Conditional_Recurrent_Flow_Conditional_Generation_of_Longitudinal_Samples_With_Applications_ICCV_2019_paper.pdf)

Seong Jae Hwang, Zirui Tao, Wom Hwa Kim, Vikas Singh, "Conditional Recurrent Flow: Conditional Generation of Longitudinal Samples with Applications to Neuroimaging", International Conference on Computer Vision (ICCV), 2019.

http://pitt.edu/~sjh95/

## Code
1. The code largely leverages an excellent repository for invertible architectures: [FrEIA](https://github.com/VLL-HD/FrEIA)
2. Dependencies:
- 

## Data
1. [Moving Digit MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/) from [this paper](http://www.cs.toronto.edu/~nitish/unsup_video.pdf) is essentially built off of the traditional MNIST digits. Various properties of the sequences (e.g., directions, speed, sizes, etc.) can be adjust within the data generator.
2. [Moving Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) can simply replace the Moving Digit MNIST within the data generator since they have the identical image sizes (and number of labels).
3. Neuroimaging data ([TADPOLE](https://tadpole.grand-challenge.org/)): The longitudinal neuroimaging data is a part of the [TADPOLE](https://tadpole.grand-challenge.org/) Challenge by the [EuroPOND consortium](http://europond.eu/) in collaboration with the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/). 
