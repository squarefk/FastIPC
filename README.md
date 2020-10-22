## Usage

```
export PYTHONPATH=/path/to/FastIPC:$PYTHONPATH
```

```
python3 -m pip install taichi
python3 -m pip install taichi_three
python3 -m pip install taichi taichi_glsl
python3 -m pip install meshio
python3 -m pip install scipy

cd wrapper
g++ wrapper.cpp EVCTCD/CTCD.cpp -o a.so -fPIC -shared -std=c++1z -mavx2 -mfma -I .
cd ..
python3 NEWTON_IPC.py
```
