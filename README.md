## Build with raw Cholmod

1. Install SuiteSparse
```
sudo apt-get install libsuitesparse-dev
```

2. Configure FastIPC
```
## Add the following line into ~/.bashrc
export PYTHONPATH=/path/to/FastIPC:$PYTHONPATH

pip3 install taichi taichi_three taichi_glsl meshio scipy scikit-sparse
cd common/math/wrapper
g++ wrapper.cpp EVCTCD/CTCD.cpp -o a.so -fPIC -O2 -shared -std=c++1z -mavx2 -mfma -I .
```

## Build with MKL-Enhanced Cholmod (Not Necessary)

1. Install [MKL](https://software.intel.com/content/www/us/en/develop/articles/qualify-for-free-software.html#student) (Intel Math Kernel Library, free tools for students)

2. Build SuiteSparse from source (with MKL linking flags)
```
sudo apt install libomp-dev libmpc-dev

## Add the following lines into ~/.zshrc
export PATH=/snap/clion/current/bin/cmake/linux/bin:$PATH
export LIBRARY_PATH=/opt/intel/oneapi/mkl/2021.1.1/lib/intel64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/2021.1.1/lib/intel64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2021.1.1/linux/compiler/lib/intel64_lin:$LD_LIBRARY_PATH
export LD_PRELOAD=/opt/intel/oneapi/mkl/2021.1.1/lib/intel64/libmkl_def.so.1:/opt/intel/oneapi/mkl/2021.1.1/lib/intel64/libmkl_avx2.so.1:/opt/intel/oneapi/mkl/2021.1.1/lib/intel64/libmkl_core.so:/opt/intel/oneapi/mkl/2021.1.1/lib/intel64/libmkl_intel_lp64.so:/opt/intel/oneapi/mkl/2021.1.1/lib/intel64/libmkl_intel_thread.so:/opt/intel/oneapi/compiler/2021.1.1/linux/compiler/lib/intel64_lin/libiomp5.so

git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
cd SuiteSparse
vim SuiteSparse_config/SuiteSparse_config.mk

## Modify CUDA_PATH like: CUDA_PATH = /usr/local/cuda-10.1
## Update CUDA architecture (e.g. remove -gencode=arch=compute_30,code=sm_30 \)

make library BLAS='-lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -lmkl_blacs_intelmpi_lp64 -liomp5' LAPACK='-lmkl_scalapack_lp64' -j 12
sudo cp -r lib /usr/local
sudo cp -r include /usr/local
```

3. Same as step 2 in the previous section 


## Taichi Programming/Debugging Tips
0. `Be cautious with type`

    ti.Matrix([[0, 0], [0, 0]]) and ti.Matrix([[0.0, 0.0], [0.0, 0.0]]) have two different types.
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

4. Code in [taichi](http://github.com/taichi-dev/taichi) repo has more complete API to search for than documentation.

## Matrix Derivative Ordering Convention

Column ordering convention for

<img src="http://latex.codecogs.com/gif.latex?\frac{dA}{dX}=\frac{d\mathrm{vec}(A)}{d\mathrm{vec}(X)}" border="0"/>

e.g. in 2d

<img src="http://latex.codecogs.com/gif.latex?\frac{dA}{dX}=\begin{pmatrix}A_{11,11}&A_{11,21}&A_{11,12}&A_{11,22}\\A_{21,11}&A_{21,21}&A_{21,12}&A_{21,22}\\A_{12,11}&A_{12,21}&A_{12,12}&A_{12,22}\\A_{22,11}&A_{22,21}&A_{22,12}&A_{22,22}\\\end{pmatrix}" border="0"/>
