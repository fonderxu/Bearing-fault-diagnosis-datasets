# Introduction

The **py-cwru** is a bearing fault diagnosis dataset and all pythonic, which can be applied to **Deep Learning**, **Transfer Learning** and **Semi-Supervised Learning** (domain adaptation), etc.. Original data comes from the official bearing vibration data of Western Reserve University [1]. The data is based on the original vibration data without repeated segmentation (the sampling window lengthis equal to hop length) and is provided in numpy.Array format.

Although the original '*. mat' file can be downloaded automatically at the first call of **py-cwru**, we still recommend you, for possible time consumption and transmission interruption, manually download the dataset file through the link below:

Download Link：https://pan.xunlei.com/s/VNNBE0GiGHD5r5_l4BsYOTP-A1?pwd=3syk# 

Extraction code：3syk

You may need to reorganize file path if you download from the official file:

```html
Cwru
	12DriveEndFault
		1797
			12DriveEndFault
				1797	
					0.007-Ball.mat
					..........
			NormalBaseline
				1797
					.........
	12FanEndFault
		......
	48DriveEndFault
```

Our code refers to [[Litchiware/cwru (github.com)](https://github.com/Litchiware/cwru)] and modifies a bit of impractical code.

[1] https://engineering.case.edu/bearingdatacenter/download-data-file.

# Example

```python
import py_cwru
cwru = py_cwru.CWRU(exp='12DriveEndFault', rpm='1797', length=1024, root=r'.')
train_x, train_y = cwru.x, cwru.y
print(train_x.shape, train_y.shape)
print(type(train_x))

--------
(2013, 1024) (2013,)
<class 'numpy.ndarray'>
```



