# Introduction

The **bearing-python** is package which include common bearing fault diagnosis dataset implement in numpy, CWRU[1], Paderborn[2] and Jnu[3], etc., and will include more in the future. This package can be used to **Deep Learning**, **Transfer Learning** and **Semi-Supervised Learning** (domain adaptation), etc.. Original data comes from the official bearing vibration. The data is based on the original vibration data without repeated segmentation (the sampling window length is equal to hop length) and is provided in numpy.Array format.

Although the original '*. mat' file of CWRU can be downloaded automatically at the first call , we still recommend you, for possible time consumption and transmission interruption, manually download the dataset file through the link below:

CWRU:
Download Link：https://pan.xunlei.com/s/VNNBE0GiGHD5r5_l4BsYOTP-A1?pwd=3syk# 

Extraction Code：3syk

Paderborn:
Download Link: https://www.aliyundrive.com/s/AanRmmBNZna 

Extraction Code: 4qq9

Jnu:
Download Link: https://pan.xunlei.com/s/VNNVrpOoaEAPPdIODQaJQQcQA1?pwd=jnri# 

Extraction Code: jnri

You may need to reorganize dir if you download CWRU from the official file by following format:

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

# install package 
```shell script
pip install bearing-py
```

# Example
## datasets
```python
import bearing
dataset = bearing.datasets.Cwru(exp='12DriveEndFault', rpm='1797', length=1024, root=r'.')
print(dataset.x.shape, dataset.y.shape)
print(type(dataset.x))
cout >>:
 (2013, 1024) (2013,)
 <class 'numpy.ndarray'>
```
--------

## transform
```python
import torch
import bearing.transform as T
transform = T.Compose(
    T.Normalization()
)
x = torch.randn(1, 1, 1024)
x_transform = transform(x)
```

# Declaration
Our code refers to [[Litchiware/cwru (github.com)](https://github.com/Litchiware/cwru)] and modifies a bit.

for more details about each dataset please visit the following link:
[1] CWRU: https://engineering.case.edu/bearingdatacenter/download-data-file.

[2] Paderborn: https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/

[3] Jnu: 