## Object Recognition

Get the weights file from [here](https://www.dropbox.com/s/4d8tx16wuz45o66/yolo-obj_45000.weights?dl=0) and place the downloaded file in the bin/ directory.

To install the module (Only for the first time), run the following commands  one after another:
```
python3 setup.py build_ext --inplace
```
```
pip3 install -e .
```
```
pip3 install .
```

To run the module, run the following command :
```
flow --model cfg/yolo-voc-bdd.cfg --load bin/yolo-obj_45000.weights --demo camera --gpu 1.0
```


