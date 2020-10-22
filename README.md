## Usage

```
export PYTHONPATH=/path/to/FastIPC:$PYTHONPATH
```

```
pip3 install taichi taichi_three taichi_glsl meshio scipy
cd common/math/wrapper
g++ wrapper.cpp EVCTCD/CTCD.cpp -o a.so -fPIC -O2 -shared -std=c++1z -mavx2 -mfma -I .
```

## Taichi Programming Tips
1. `Profiling your code`

    There is a timer implemented in [timer.py](https://github.com/penn-graphics-research/FastIPC/blob/master/common/utils/timer.py). The usage is like:
    
        with Timer("Process 1"):
            ### code snippet 1
            ...
        with Timer("Process 2"):
            ### code snippet 2
            ...
        Timer_Print()

    It will always output average timing for each component. It will compute the compile time automatically which is calculated with the first round running time.
2. `Restart functionality`

    Restart is easy to implement with pickle so it is not implemented as a separate file:

        # load data
        [x_, v_, boundary] = pickle.load(open(directory + f'caches/{f_start:06d}.p', 'rb'))
        x.from_numpy(x_)
        v.from_numpy(v_)

        # save data
        pickle.dump([x.to_numpy(), v.to_numpy(), boundary], open(directory + f'caches/{f + 1:06d}.p', 'wb'))
        
        
3. `Use ti.template()`
    1. The following four will be recognized as four different types if passed in for ti.template(). Only first two of them can be used to access matrix indices.
        1. `[1, 2, 3]`
        2. `ti.Vector([1, 2, 3])`
        3. variable `a` (assigned by [1, 2, 3])
        4. variable `a` (assigned by ti.Vector([1, 2, 3]))
        
       A good example is in [math_tools.py](https://github.com/penn-graphics-research/FastIPC/blob/master/common/math/math_tools.py):
       
           @ti.func
           def extract_vec(v, idx: ti.template()):
               vec = ti.Matrix.zero(ti.get_runtime().default_fp, len(idx))
               for i, j in ti.static(enumerate(idx)):
                   vec[i] = v[j]
               return vec
    2. `ti.template()` can be used in @ti.kernel to call with different fields. Official documentation only mentions @ti.kernel can only hold 8 scalar parameters. These are done in compile time. `ti.static()` is just like if-constexpr in C++ which will optimize running time a lot.
    3. `A.n` and `A.m` can retrieve the dimensions of matrix. `ti.get_runtime().default_fp` can be used to get current precision.

4. [taichi](http://github.com/taichi-dev/taichi) repo has more complete API to search for than documentation.
