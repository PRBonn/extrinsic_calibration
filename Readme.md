Motion Based Multi-Sensor Extrinsic Calibration 
------------
These python scripts support multi-sensor extrinsic calibration using only odmetry/motion infomation.
The main feature is a non-linear least squares optimization based on Gauss-Helmert framework, as described in the following papers:

	Kaihong Huang and Cyrill Stachniss,
	Extrinsic Multi-Sensor Calibration For Mobile Robots Using the Gauss-Helmert Model, 
	In Proc. of the IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), 2017


To run the script, you need to install *cppad* (and its python wrapper *pycppad* ) for automatic differentiation.
### Installing cppad and pycppad###
First install cppad. For Ubuntu 16.04 users, 
```
sudo apt-get install cppad
```
For Ubuntu 14.04, complie and install it from source code
```
git clone https://github.com/coin-or/CppAD.git
cd CppAD/
mkdir build && cd build
cmake ..
sudo make install
```
Next is pycppad
```
git clone https://github.com/b45ch1/pycppad.git
cd pycppad/
gedit setup.py
```
change *cppad_include_dir* to *['/usr/include']*  in the file setup.py then
```
python setup.py build_ext --inplace --debug --undef NDEBUG
pip install .
python ./test_example.py
```
### How to use ###
Please check *demo_and_test* function in calibration.py file for example usage.