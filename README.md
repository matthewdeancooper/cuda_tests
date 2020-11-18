# cuda_tests
I have found it valuable to have a minimum working CNN for both TensorFlow
and PyTorch - allowing me to easily test CUDA and cuDNN support on new installs.


Activate a python virtual environment (I use pyenv and venv).
```
pyenv local 3.7.5
python3.7 -m venv env
source env/bin/activate
```
Install requirements via pip
```
pip install --upgrade pip
pip install -r requirements.txt
```
Run each test to see if your GPU is being utilised
```
python tf_test.py
python torch_test.py
```
