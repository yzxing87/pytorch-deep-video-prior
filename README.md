# pytorch-deep-video-prior (DVP)
Official PyTorch implementation for NeurIPS 2020 paper: Blind Video Temporal Consistency via Deep Video Prior

[TensorFlow implementation](https://github.com/ChenyangLEI/deep-video-prior)
| [paper](https://arxiv.org/abs/2010.11838)
| [project website](https://chenyanglei.github.io/DVP/index.html)


## Introduction
Our method is a general framework to improve the temporal consistency of video processed by image algorithms. 
<!-- For example, our frame can take the results of single image colorization or single image dehazing algorithm results as input, and produce consistent video colorization or video dehazing results.  -->
For example, combining single image colorization or single image dehazing algorithm with our framework, we can achieve the goal of video colorization or video dehazing. 


<img src="example/example_in.gif" height="220px"/> <img src="example/example_out.gif" height="220px"/> 
<img src="example/example2_in.gif" height="220px"/> <img src="example/example2_out.gif" height="220px"/> 


## Dependency

### Environment
This code is based on PyTorch. It has been tested on Ubuntu 18.04 LTS.

Anaconda is recommended: [Ubuntu 18.04](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-18-04)
| [Ubuntu 16.04](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04)

After installing Anaconda, you can setup the environment simply by

```
conda env create -f environment.yml
```


## Inference

### Demo 
```
bash test.sh
```
The results will be saved in ./result

### Use your own data 
For the video with unimodal inconsistency:

```
python main_IRT.py --max_epoch 25 --input PATH_TO_YOUR_INPUT_FOLDER --processed PATH_TO_YOUR_PROCESSED_FOLDER --model NAME_OF_YOUR_MODEL --with_IRT 0 --IRT_initialization 0 --output ./result/OWN_DATA
```

For the video with multimodal inconsistency:

```
python main_IRT.py --max_epoch 25 --input PATH_TO_YOUR_INPUT_FOLDER --processed PATH_TO_YOUR_PROCESSED_FOLDER --model NAME_OF_YOUR_MODEL --with_IRT 1 --IRT_initialization 1 --output ./result/OWN_DATA
```


## Citation
If you find this work useful for your research, please cite:
```
@inproceedings{lei2020dvp,
  title={Blind Video Temporal Consistency via Deep Video Prior},
  author={Lei, Chenyang and Xing, Yazhou and Chen, Qifeng},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}                
```


## Contact
Feel free to contact me if there is any question. (Yazhou Xing,  yzxing87@gmail.com)
