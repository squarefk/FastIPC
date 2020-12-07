""" Sparse matrix (CSR) class and linear solver"""
import taichi as ti
import numpy as np

real = ti.float64

MAX_LINEAR = 1000000 # maximum size of linear system supported

@ti.data_oriented
class SparseMatrix:
    '''
        At present we assume every column must be less then self.cols items.
    '''
    def __init__(self, shape_ = None):
        self.rows = ti.field(ti.i32, shape=()) # num of rows
        self.cols = ti.field(ti.i32, shape=()) # num of columns
        # if shape_ == None:
        #     shape_ = (MAX_LINEAR, MAX_LINEAR)
        # else:
        #     assert isinstance(shape_, tuple)
        #     self.rows = shape_[0]
        #     self.cols = shape_[1]
        self.coef_value = ti.field(real, shape=MAX_LINEAR*1000)
        self.col_index = ti.field(ti.i32, shape=MAX_LINEAR*1000)
        self.outerIndex = ti.field(ti.i32, shape=MAX_LINEAR)
        self.innerNonZeros = ti.field(ti.i32, shape=MAX_LINEAR)
        # self.nonZeros = ti.field(ti.i32, shape=())
        # For computation storage only
        self.Ap = ti.field(real, shape=MAX_LINEAR)

    ############ initialization ############
    def setAllZero(self):
        self.outerIndex.fill(0)
        self.innerNonZeros.fill(0)
        for i in range(self.rows[None]):
            self.outerIndex[i] = i * self.cols[None]        

    def setIdentity(self, num):
        self.rows[None] = num
        self.cols[None] = num
        self.outerIndex.fill(0)
        self.innerNonZeros.fill(0)
        for i in range(self.rows[None]):
            self.outerIndex[i] = i * 100
        for i in range(self.rows[None]):
            idx = self.outerIndex[i]
            self.coef_value[idx] = 1.0
            self.col_index[idx] = i
            self.innerNonZeros[i] = 1

    def setFromTriplets(self, data_row, data_col, data_val, shape_ = None):
        '''
            Set from numpy Triplets (data_row, data_col, data_val)
        '''
        ## TODO: Make this function a kernel
        if not shape_ == None:
            assert isinstance(shape_, tuple)
            self.rows[None] = shape_[0]
            self.cols[None] = shape_[1]
        self.outerIndex.fill(0)
        self.innerNonZeros.fill(0)
        for i in range(self.rows[None]):
            self.outerIndex[i] = i * self.cols[None]
        n = data_row.shape[0]
        for i in range(n):
            row, col, val = data_row[i], data_col[i], data_val[i]
            idx = self.outerIndex[row] + self.innerNonZeros[row]
            self.coef_value[idx] = val
            self.col_index[idx] = col
            self.innerNonZeros[row] += 1

    def setFromFullMatrix(self, fullMat):
        '''
            Set from numpy matrix
        '''
        ## TODO: Make this function a kernel
        self.rows[None], self.cols[None] = fullMat.shape[0], fullMat.shape[1]
        self.outerIndex.fill(0)
        self.innerNonZeros.fill(0)
        for i in range(self.rows[None]):
            self.outerIndex[i] = i * self.cols[None]
        for i in range(self.rows[None]):
            start_idx = self.outerIndex[i]
            end_idx = 0
            for j in range(self.cols[None]):
                val = fullMat[i, j]              
                if not val == 0:
                    self.col_index[start_idx + end_idx] = j
                    self.coef_value[start_idx + end_idx] = val
                    end_idx += 1
                self.innerNonZeros[i] = end_idx    

    def init4(self):
        CV = np.array([2.0, \
                        3.0,-1.0,1.0, \
                        2.0, \
                        -1.0,3.0,-1.0, \
                        1.0,-1.0,1.0, \
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        CI = np.array([0, \
                        1,3,4, \
                        2, \
                        1,3,4, \
                        1,3,4, \
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.coef_value.from_numpy(CV)
        self.col_index.from_numpy(CI)


        self.outerIndex.from_numpy(np.array([0,1,4,5,8]))
        self.innerNonZeros.from_numpy(np.array([1,3,1,3,3]))

    def prepareColandVal(self, num, d=2):
        self.rows[None] = d*num
        self.cols[None] = d*num
        self.outerIndex.fill(0)
        self.innerNonZeros.fill(0)
        for i in range(self.rows[None]):
            self.outerIndex[i] = i * 1000
        # for i in range(self.rows[None]):
        #     idx = self.outerIndex[i]
        #     self.coef_value[idx] = 1.0
        #     self.col_index[idx] = i
        #     self.innerNonZeros[i] = 1

    @ti.kernel
    def setFromColandVal(self, entryCol:ti.template(), entryVal:ti.template(), num:ti.i32):
        for i in range(num):
            num_entry = 0
            start_idx = self.outerIndex[2*i]
            num_entry1 = 0
            start_idx1 = self.outerIndex[2*i+1]
            for k in range(25):
                c = i * 25 + k
                j = entryCol[c]
                M = entryVal[c]
                if not j == -1:
                    self.col_index[start_idx + num_entry] = 2*j
                    self.coef_value[start_idx + num_entry] = M[0,0]
                    self.col_index[start_idx + num_entry + 1] = 2*j+1
                    self.coef_value[start_idx + num_entry + 1] = M[0,1]
                    num_entry += 2
                    self.col_index[start_idx1 + num_entry1] = 2*j
                    self.coef_value[start_idx1 + num_entry1] = M[1,0]
                    self.col_index[start_idx1 + num_entry1 + 1] = 2*j+1
                    self.coef_value[start_idx1 + num_entry1 + 1] = M[1,1]
                    num_entry1 += 2
            self.innerNonZeros[2*i] = num_entry
            self.innerNonZeros[2*i+1] = num_entry1

    @ti.kernel
    def setFromColandVal3(self, entryCol:ti.template(), entryVal:ti.template(), num:ti.i32):
        for i in range(num):
            num_entry = 0
            start_idx = self.outerIndex[3*i]
            num_entry1 = 0
            start_idx1 = self.outerIndex[3*i+1]
            num_entry2 = 0
            start_idx2 = self.outerIndex[3*i+2]
            for k in range(125):
                c = i * 125 + k
                j = entryCol[c]
                M = entryVal[c]
                if not j == -1:
                    self.col_index[start_idx + num_entry] = 3*j
                    self.coef_value[start_idx + num_entry] = M[0,0]
                    self.col_index[start_idx + num_entry + 1] = 3*j+1
                    self.coef_value[start_idx + num_entry + 1] = M[0,1]
                    self.col_index[start_idx + num_entry + 2] = 3*j+2
                    self.coef_value[start_idx + num_entry + 2] = M[0,2]
                    num_entry += 3
                    self.col_index[start_idx1 + num_entry1] = 3*j
                    self.coef_value[start_idx1 + num_entry1] = M[1,0]
                    self.col_index[start_idx1 + num_entry1 + 1] = 3*j+1
                    self.coef_value[start_idx1 + num_entry1 + 1] = M[1,1]
                    self.col_index[start_idx1 + num_entry1 + 2] = 3*j+2
                    self.coef_value[start_idx1 + num_entry1 + 2] = M[1,2]
                    num_entry1 += 3
                    self.col_index[start_idx2 + num_entry2] = 3*j
                    self.coef_value[start_idx2 + num_entry2] = M[2,0]
                    self.col_index[start_idx2 + num_entry2 + 1] = 3*j+1
                    self.coef_value[start_idx2 + num_entry2 + 1] = M[2,1]
                    self.col_index[start_idx2 + num_entry2 + 2] = 3*j+2
                    self.coef_value[start_idx2 + num_entry2 + 2] = M[2,2]
                    num_entry2 += 3
            self.innerNonZeros[3*i] = num_entry
            self.innerNonZeros[3*i+1] = num_entry1
            self.innerNonZeros[3*i+2] = num_entry2

    ############ sparsity ############
    def initSpace(self):
        pass


    @ti.kernel
    def makeCompressed(self):
        '''
            Compress sparse rows
        '''
        for i in range(self.rows[None]):
            start_idx = self.outerIndex[i]
            num_idx = self.innerNonZeros[i]
            c_set = []
            c_num = 0
            for k in range(num_idx):
                val = self.coef_value[start_idx + k]
                j = self.col_index[start_idx + k]
                for s in range(c_num):
                    if c_set[s] == j:
                        pass



    ############ matrix computation ############
    @ti.kernel
    def multiply(self, p:ti.template()):
        '''
            Ap = A*p
        '''
        for i in range(self.rows[None]):
            start_idx = self.outerIndex[i]
            num_idx = self.innerNonZeros[i]

            sum = 0.0
            for k in range(num_idx):
                j = self.col_index[start_idx + k]
                val = self.coef_value[start_idx + k]
                sum += val * p[j]
            
            self.Ap[i] = sum      

    ############ index operations ############
    @ti.pyfunc
    def get_value(self, row, col):
        start_idx = self.outerIndex[row]
        num_idx = self.innerNonZeros[row]
        # print(row, start_idx, num_idx)
        result = 0.0
        for k in range(num_idx):
            if self.col_index[start_idx + k] == col:
                result = self.coef_value[start_idx + k]
                
        return result

    def __getitem__(self, index):
        assert isinstance(index, tuple)
        assert len(index) == 2
        return self.get_value(index[0],index[1])

    # @ti.pyfunc
    def set_value(self, row:ti.i32, col:ti.i32, value:real):
        start_idx = self.outerIndex[row]
        num_idx = self.innerNonZeros[row]

        isExist = False
        for k in range(num_idx):
            if self.col_index[start_idx + k] == col:
                self.coef_value[start_idx + k] = value
                isExist = True
                break
        if not isExist:
            self.col_index[start_idx + num_idx] = col
            self.coef_value[start_idx + num_idx] = value
            self.innerNonZeros[row] += 1
            # assert start_idx + self.innerNonZeros[row] < self.outerIndex[row+1]

    def __setitem__(self, index, value):
        assert isinstance(index, tuple)
        assert len(index) == 2
        self.set_value(index[0],index[1],value)

    def toFullMatrix(self):
        fullMat = np.zeros(shape=(self.rows[None], self.cols[None]))
        for i in range(self.rows[None]):
            start_idx = self.outerIndex[i]
            end_idx = self.innerNonZeros[i]
            for k in range(end_idx):
                j = self.col_index[start_idx + k]
                val = self.coef_value[start_idx + k]
                fullMat[i, j] += val
        return fullMat

    def display(self):
        print(self.coef_value.to_numpy())
        print(self.col_index.to_numpy())
        print(self.outerIndex.to_numpy())
        print(self.innerNonZeros.to_numpy())




@ti.data_oriented
class CGSolver:
    def __init__(self, num = MAX_LINEAR):
        self.N = ti.field(ti.i32, shape=())
        self.stride = 1
        self.r = ti.field(real, shape=num) # residual
        self.q = ti.field(real, shape=num) # z
        self.x = ti.field(real, shape=num) # solution
        self.p = ti.field(real, shape=num)
        # TODO final sol: consider using col compressed matrix this is the most efficient way. But just a bit complex in coding
        # self.A = ti.field(real, shape=(Nx*Ny, 5))
        # self.RHS = ti.field(real, shape=(Nx, Ny))
        self.Ap = ti.field(real, shape=num)
        self.alpha = ti.field(real, shape=())
        self.beta = ti.field(real, shape=())

        self.boundary = ti.field(real, shape=num)
        

    def compute(self, A, stride = 2):
        '''
            A: Sparse Matrix
        '''
        assert isinstance(A, SparseMatrix)
        assert A.rows[None] == A.cols[None]
        self.A = A
        self.N[None] = A.rows[None]
        self.stride = stride
        self.boundary.fill(0)


    ############ functions ############
    @ti.kernel
    def dotProduct(self, p:ti.template(), q:ti.template()) -> real:
        result = 0.0
        for I in p:
            result += p[I] * q[I]
        return result

    @ti.kernel
    def copyVector(self, p:ti.template(), q:ti.template()):
        for I in range(self.N[None]):
            p[I] = q[I]

    @ti.kernel
    def computAp(self, p:ti.template()):
        '''
            self.Ap = self.A*p
        '''
        for i in range(self.N[None]):
            # start_idx = 0
            # if i > 0:
            #     start_idx = self.A.row_range[i-1]+1      
            # end_idx = self.A.row_range[i]+1
            start_idx = self.A.outerIndex[i]
            num_idx = self.A.innerNonZeros[i]

            sum = 0.0
            for k in range(num_idx):
                j = self.A.col_index[start_idx + k]
                val = self.A.coef_value[start_idx + k]
                sum += val * p[j]
            
            self.Ap[i] = sum

    @ti.kernel
    def update_p(self):
        for I in range(self.N[None]):
        # for I in ti.grouped(self.p_base):
            self.p[I] = self.q[I]+self.beta[None]*self.p[I]

    @ti.kernel
    def update_residual(self):  # r = r - alpha * Ap
        for I in range(self.N[None]):
            self.r[I] -= self.alpha[None]*self.Ap[I]

    @ti.kernel
    def update_sol(self): # x = x + alpha * p
        for I in range(self.N[None]):
            self.x[I] += self.alpha[None]*self.p[I]

    @ti.kernel
    def precondition(self): # q = M^-1 r
        for I in range(self.N[None]):
            self.q[I] = self.r[I] / self.A[I,I] # identity

    @ti.kernel
    def setBoundary(self, b:ti.template()):
        for I in range(self.N[None] // self.stride):
            self.boundary[I] = b[I]

    @ti.kernel
    def project(self, r:ti.template()):
        dim = self.stride
        for I in range(self.N[None] // dim):
            if self.boundary[I] == 1:
                # r[2*I] = 0
                # r[2*I+1] = 0
                for d in range(dim):
                    r[I*dim+d] = 0

    def solve(self, b):
        '''
            Diagonal preconditioned Conjugate Gradient method
        '''
        self.r.fill(0)
        self.p.fill(0)
        self.q.fill(0)
        self.x.fill(0) # zero initial guess !!!!!!!!!!!!!!!!!!
        # self.Ap.fill(0)
        self.alpha[None] = 0.0
        self.beta[None] = 0.0
        self.copyVector(self.r, b)
        # self.computAp(self.x)
        # print("aaaaaaaaaa", self.N[None], np.max(b.to_numpy()))
        self.project(self.r)
        self.precondition()
        self.update_p()


        zTr = self.dotProduct(self.r, self.q)
        residual_preconditioned_norm = ti.sqrt(zTr)
        max_iterations = 250
        for cnt in range(max_iterations):
            # print(cnt, residual_preconditioned_norm)
            if residual_preconditioned_norm < 1e-10:
                # print("CG terminates at", cnt, "; residual =", residual_preconditioned_norm)
                return
            # print(zTrK)
            self.computAp(self.p)
            self.project(self.Ap)
            self.alpha[None] = zTr / self.dotProduct(self.Ap, self.p)
         
            self.update_sol()
            self.update_residual()

            self.precondition()

            zTrk_last = zTr
            zTr = self.dotProduct(self.q, self.r)
            self.beta[None] = zTr / zTrk_last

            self.update_p()

            residual_preconditioned_norm = ti.sqrt(zTr)
        print("ConjugateGradient max iterations reached, iter =", max_iterations, "; residual =", residual_preconditioned_norm)