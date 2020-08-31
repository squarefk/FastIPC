## Usage

```
cd wrapper
g++ wrapper.cpp EVCTCD/CTCD.cpp -o a.so -fPIC -shared -std=c++1z -mavx2 -mfma -I .
cd ..
python3 NEWTON_IPC.py
```
