
# Usage Guide

## Environment Requirements
Make sure you have the following versions installed:

- Python 3.6  
- Torch 1.4.0  
- Numpy 1.18.1  
- Scikit-learn 0.22.1  
- Scikit-image 0.16.2  
- Scipy 1.4.1  


## Data Prepare


> We selected a subset of images from the [Stanford Lytro dataset](http://lightfields.stanford.edu/LF2016.html) as a randomly synthesized training set for training. The list of images in the training set can be found in the file `trainfile_list.txt`.
> The download link for the [test dataset](https://bjtueducn-my.sharepoint.com/:u:/g/personal/23120336_bjtu_edu_cn/Ed9g1rL2QHxCujluV7r9PBUBNUu05OXJxatdkOAvG6yCjA?e=v8Pvwz).



## Train the Model
To train the model, run:

```bash
python train.py
```


## Pretrained Model

```bash
NetworkSave/ourmode/DeRefLF_best.pkl
```


## Test the Model

To test the model, run:

```bash
python test.py
````

