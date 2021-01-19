""" Sparse matrix (CSR) class and linear solver"""
import taichi as ti
import numpy as np

real = ti.float64


@ti.data_oriented
class SparseMatrix:
    '''
        At present we assume every column must be less then self.cols items.
    '''
    def __init__(self,
                 shape_=None,
                 max_row_num=1000000,
                 defualt_none_zero_width=1000):
        self.max_row_num = max_row_num
        self.defualt_none_zero_width = defualt_none_zero_width
        self.rows = ti.field(ti.i32, shape=())  # num of rows
        self.cols = ti.field(ti.i32, shape=())  # num of columns
        self.coef_value = ti.field(real,
                                   shape=max_row_num * defualt_none_zero_width)
        self.col_index = ti.field(ti.i32,
                                  shape=max_row_num * defualt_none_zero_width)
        self.outerIndex = ti.field(ti.i32, shape=max_row_num)
        self.innerNonZeros = ti.field(ti.i32, shape=max_row_num)
        # For computation storage only
        self.Ap = ti.field(real, shape=max_row_num)

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
            self.outerIndex[i] = i * self.defualt_none_zero_width
        for i in range(self.rows[None]):
            idx = self.outerIndex[i]
            self.coef_value[idx] = 1.0
            self.col_index[idx] = i
            self.innerNonZeros[i] = 1

    def setFromTriplets(self, data_row, data_col, data_val, shape_=None):
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
            self.outerIndex[i] = i * self.defualt_none_zero_width
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

        self.outerIndex.from_numpy(np.array([0, 1, 4, 5, 8]))
        self.innerNonZeros.from_numpy(np.array([1, 3, 1, 3, 3]))

    def prepareColandVal(self, num, d=2):
        self.initShape(d*num, d*num, self.defualt_none_zero_width)

    @ti.kernel
    def initShape(self, row:ti.i32, col:ti.i32, rowsize:ti.i32):
        self.rows[None] = row
        self.cols[None] = col
        for i in range(self.rows[None]):
            self.outerIndex[i] = i * rowsize
            self.innerNonZeros[i] = 0

    @ti.kernel
    def setFromColandVal(self, entryCol: ti.template(),
                         entryVal: ti.template(), num: ti.i32):
        for i in range(num):
            num_entry = 0
            start_idx = self.outerIndex[2 * i]
            num_entry1 = 0
            start_idx1 = self.outerIndex[2 * i + 1]
            for k in range(25):
                c = i * 25 + k
                j = entryCol[c]
                M = entryVal[c]
                if not j == -1:
                    self.col_index[start_idx + num_entry] = 2 * j
                    self.coef_value[start_idx + num_entry] = M[0, 0]
                    self.col_index[start_idx + num_entry + 1] = 2 * j + 1
                    self.coef_value[start_idx + num_entry + 1] = M[0, 1]
                    num_entry += 2
                    self.col_index[start_idx1 + num_entry1] = 2 * j
                    self.coef_value[start_idx1 + num_entry1] = M[1, 0]
                    self.col_index[start_idx1 + num_entry1 + 1] = 2 * j + 1
                    self.coef_value[start_idx1 + num_entry1 + 1] = M[1, 1]
                    num_entry1 += 2
            self.innerNonZeros[2 * i] = num_entry
            self.innerNonZeros[2 * i + 1] = num_entry1

    @ti.kernel
    def setFromColandVal3(self, entryCol: ti.template(),
                          entryVal: ti.template(), num: ti.i32):
        for i in range(num):
            num_entry = 0
            start_idx = self.outerIndex[3 * i]
            num_entry1 = 0
            start_idx1 = self.outerIndex[3 * i + 1]
            num_entry2 = 0
            start_idx2 = self.outerIndex[3 * i + 2]
            for k in range(125):
                c = i * 125 + k
                j = entryCol[c]
                M = entryVal[c]
                if not j == -1:
                    self.col_index[start_idx + num_entry] = 3 * j
                    self.coef_value[start_idx + num_entry] = M[0, 0]
                    self.col_index[start_idx + num_entry + 1] = 3 * j + 1
                    self.coef_value[start_idx + num_entry + 1] = M[0, 1]
                    self.col_index[start_idx + num_entry + 2] = 3 * j + 2
                    self.coef_value[start_idx + num_entry + 2] = M[0, 2]
                    num_entry += 3
                    self.col_index[start_idx1 + num_entry1] = 3 * j
                    self.coef_value[start_idx1 + num_entry1] = M[1, 0]
                    self.col_index[start_idx1 + num_entry1 + 1] = 3 * j + 1
                    self.coef_value[start_idx1 + num_entry1 + 1] = M[1, 1]
                    self.col_index[start_idx1 + num_entry1 + 2] = 3 * j + 2
                    self.coef_value[start_idx1 + num_entry1 + 2] = M[1, 2]
                    num_entry1 += 3
                    self.col_index[start_idx2 + num_entry2] = 3 * j
                    self.coef_value[start_idx2 + num_entry2] = M[2, 0]
                    self.col_index[start_idx2 + num_entry2 + 1] = 3 * j + 1
                    self.coef_value[start_idx2 + num_entry2 + 1] = M[2, 1]
                    self.col_index[start_idx2 + num_entry2 + 2] = 3 * j + 2
                    self.coef_value[start_idx2 + num_entry2 + 2] = M[2, 2]
                    num_entry2 += 3
            self.innerNonZeros[3 * i] = num_entry
            self.innerNonZeros[3 * i + 1] = num_entry1
            self.innerNonZeros[3 * i + 2] = num_entry2

    ####-----DFG VERSIONS-----####

    @ti.kernel
    def setFromColandValDFG(self, entryCol: ti.template(),
                            entryVal: ti.template(), num: ti.i32, nNbr: ti.i32):
        for i in range(num):
            num_entry = 0
            start_idx = self.outerIndex[2 * i]
            num_entry1 = 0
            start_idx1 = self.outerIndex[2 * i + 1]
            for k in range(nNbr):
                c = i * nNbr + k
                j = entryCol[c]
                M = entryVal[c]
                if not j == -1:
                    self.col_index[start_idx + num_entry] = 2 * j
                    self.coef_value[start_idx + num_entry] = M[0, 0]
                    self.col_index[start_idx + num_entry + 1] = 2 * j + 1
                    self.coef_value[start_idx + num_entry + 1] = M[0, 1]
                    num_entry += 2
                    self.col_index[start_idx1 + num_entry1] = 2 * j
                    self.coef_value[start_idx1 + num_entry1] = M[1, 0]
                    self.col_index[start_idx1 + num_entry1 + 1] = 2 * j + 1
                    self.coef_value[start_idx1 + num_entry1 + 1] = M[1, 1]
                    num_entry1 += 2
            self.innerNonZeros[2 * i] = num_entry
            self.innerNonZeros[2 * i + 1] = num_entry1

    @ti.kernel
    def setFromColandVal3DFG(self, entryCol: ti.template(),
                             entryVal: ti.template(), num: ti.i32, nNbr: ti.i32):
        for i in range(num):
            num_entry = 0
            start_idx = self.outerIndex[3 * i]
            num_entry1 = 0
            start_idx1 = self.outerIndex[3 * i + 1]
            num_entry2 = 0
            start_idx2 = self.outerIndex[3 * i + 2]
            for k in range(nNbr):
                c = i * nNbr + k
                j = entryCol[c]
                M = entryVal[c]
                if not j == -1:
                    self.col_index[start_idx + num_entry] = 3 * j
                    self.coef_value[start_idx + num_entry] = M[0, 0]
                    self.col_index[start_idx + num_entry + 1] = 3 * j + 1
                    self.coef_value[start_idx + num_entry + 1] = M[0, 1]
                    self.col_index[start_idx + num_entry + 2] = 3 * j + 2
                    self.coef_value[start_idx + num_entry + 2] = M[0, 2]
                    num_entry += 3
                    self.col_index[start_idx1 + num_entry1] = 3 * j
                    self.coef_value[start_idx1 + num_entry1] = M[1, 0]
                    self.col_index[start_idx1 + num_entry1 + 1] = 3 * j + 1
                    self.coef_value[start_idx1 + num_entry1 + 1] = M[1, 1]
                    self.col_index[start_idx1 + num_entry1 + 2] = 3 * j + 2
                    self.coef_value[start_idx1 + num_entry1 + 2] = M[1, 2]
                    num_entry1 += 3
                    self.col_index[start_idx2 + num_entry2] = 3 * j
                    self.coef_value[start_idx2 + num_entry2] = M[2, 0]
                    self.col_index[start_idx2 + num_entry2 + 1] = 3 * j + 1
                    self.coef_value[start_idx2 + num_entry2 + 1] = M[2, 1]
                    self.col_index[start_idx2 + num_entry2 + 2] = 3 * j + 2
                    self.coef_value[start_idx2 + num_entry2 + 2] = M[2, 2]
                    num_entry2 += 3
            self.innerNonZeros[3 * i] = num_entry
            self.innerNonZeros[3 * i + 1] = num_entry1
            self.innerNonZeros[3 * i + 2] = num_entry2

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
    def multiply(self, p: ti.template()):
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
        return self.get_value(index[0], index[1])

    # @ti.pyfunc
    def set_value(self, row: ti.i32, col: ti.i32, value: real):
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
            assert start_idx + self.innerNonZeros[row] < self.outerIndex[row +
                                                                         1]

    def __setitem__(self, index, value):
        assert isinstance(index, tuple)
        assert len(index) == 2
        self.set_value(index[0], index[1], value)

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
    def __init__(self, max_row_num=1000000, dim=2):
        self.dim = dim
        self.N = ti.field(ti.i32, shape=())
        self.stride = 1
        self.r = ti.field(real, shape=max_row_num)  # residual
        self.q = ti.field(real, shape=max_row_num)  # z
        self.x = ti.field(real, shape=max_row_num)  # solution
        self.p = ti.field(real, shape=max_row_num)
        self.A_diag = ti.field(real, shape=max_row_num)
        self.A_blocks = ti.Matrix.field(self.dim*2, self.dim*2, real, shape=max_row_num) #Hold [A B; B^T C] of separable nodes, this is (dim*2) x (dim*2)
        self.Ap = ti.field(real, shape=max_row_num)
        self.alpha = ti.field(real, shape=())
        self.beta = ti.field(real, shape=())
        
        self.boundary = ti.field(ti.i32, shape=max_row_num)
        self.boundary_normal = ti.field(real, shape=max_row_num)

        #For Implicit DFGMPM
        self.useBlockDiagonalPreconditioner = False
        self.sdof = 0
        self.ndof = 0

    def compute(self, A, stride=2, useBDP = False, sdof2ndof = [], sdof = 0):
        '''
            Set A (a Sparse Matrix) as the system left-hand-side
        '''
        assert isinstance(A, SparseMatrix)
        assert A.rows[None] == A.cols[None]
        self.A = A
        self.N[None] = A.rows[None]
        self.stride = stride
        self.boundary.fill(0)
        self.useBlockDiagonalPreconditioner = useBDP
        self.sdof = sdof
        self.sdof2ndof = sdof2ndof
        if useBDP:    
            self.setBlocks()
            self.ndof = (self.N[None] / self.stride) - sdof

    ############ functions ############
    @ti.kernel
    def dotProduct(self, p: ti.template(), q: ti.template()) -> real:
        result = 0.0
        for I in p:
            result += p[I] * q[I]
        return result

    @ti.kernel
    def copyVector(self, p: ti.template(), q: ti.template()):
        for I in range(self.N[None]):
            p[I] = q[I]

    @ti.kernel
    def computAp(self, p: ti.template()):
        '''
            self.Ap = self.A*p
        '''
        for i in range(self.N[None]):
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
            self.p[I] = self.q[I] + self.beta[None] * self.p[I]

    @ti.kernel
    def update_residual(self):  # r = r - alpha * Ap
        for I in range(self.N[None]):
            self.r[I] -= self.alpha[None] * self.Ap[I]

    @ti.kernel
    def update_sol(self):  # x = x + alpha * p
        for I in range(self.N[None]):
            self.x[I] += self.alpha[None] * self.p[I]

    @ti.kernel
    def setDiagonal(self):
        for I in range(self.N[None]):
            val = self.A[I,I]
            if ti.abs(val) > 1e-10:
                self.A_diag[I] = val
            else:
                self.A_diag[I] = 1.0

    @ti.kernel
    def setBlocks(self):
        for i2 in range(self.sdof):
            #for each separable node construct the 2*stride x 2*stride matrix, [A B; B^T C]
            i1 = self.sdof2ndof[i2]
            for i in ti.static(range(self.stride)):
                for j in ti.static(range(self.stride)):
                    self.A_blocks[i2][i, j] = self.A[(i1*self.stride) + i, (i1*self.stride) + j]                                #A
                    self.A_blocks[i2][i, j + self.stride] = self.A[(i1*self.stride) + i, (i2*self.stride) + j]                  #B
                    self.A_blocks[i2][i + self.stride, j] = self.A[(i2*self.stride) + i, (i1*self.stride) + j]                  #B^T
                    self.A_blocks[i2][i + self.stride, j + self.stride] = self.A[(i2*self.stride) + i, (i2*self.stride) + j]    #C
            #Now we must invert this block, taichi allows 4x4 inversion thankfully so we can test 2D #TODO:3D
            self.A_blocks[i2] = self.A_blocks[i2].inverse() #Construct [G H; H^T J]

    @ti.kernel
    def precondition(self):  # q = M^-1 r
        for I in range(self.N[None]):
            self.q[I] = self.r[I] / self.A_diag[I]

        if ti.static(self.useBlockDiagonalPreconditioner):
            #NOTE: we grab chunk from q not r because we also want the diagonal conditioning!
            #For each separable node do q = P_B * r
            for i2 in range(self.sdof):
                i1 = self.sdof2ndof[i2]
                rChunk = ti.Vector.zero(real, 2*self.stride) #have to grab the right parts of q to process
                for d in ti.static(range(self.stride)):
                    rChunk[d] = self.r[(i1 * self.stride) + d]
                    rChunk[d + self.stride] = self.r[((i1+i2) * self.stride) + d]
                #Now multiply our conditioning matrix times the residual chunk
                out = self.A_blocks[i2] @ rChunk
                #Map these results to q
                for d in ti.static(range(self.stride)):
                    self.q[(i1 * self.stride) + d] = out[d]
                    self.q[((i1+i2) * self.stride) + d] = out[d + self.stride]

    @ti.kernel
    def setBoundary(self, b: ti.template()):
        for I in range(self.N[None] // self.stride):
            self.boundary[I] = b[I]

    @ti.kernel
    def setBoundaryNormal(self, b_n: ti.template()):
        for I in range(self.N[None]):
            self.boundary_normal[I] = b_n[I]

    @ti.kernel
    def project(self, r: ti.template()):
        dim = self.stride
        for I in range(self.N[None] // dim):
            if self.boundary[I] == 1: # sticky
                for d in range(dim):
                    r[I * dim + d] = 0
            if self.boundary[I] == 2: # slip
                dot = 0.0
                for d in range(dim):
                    dot += r[I * dim + d] * self.boundary_normal[I * dim + d]
                for d in range(dim):
                    r[I * dim + d] -= dot * self.boundary_normal[I * dim + d]

    def solve(self,
              b,
              verbose=True,
              max_iterations=5000,
              terminate_residual=1e-9):
        '''
            Diagonal preconditioned Conjugate Gradient method
        '''
        self.setDiagonal()
        # self.r.fill(0)
        self.p.fill(0)
        self.q.fill(0)
        self.x.fill(0)  # zero initial guess !!!!!!!!!!!!!!!!!!
        self.alpha[None] = 0.0
        self.beta[None] = 0.0
        self.copyVector(self.r, b)
        self.project(self.r)
        self.precondition()
        self.project(self.q)
        self.update_p()

        zTr = self.dotProduct(self.r, self.q)
        residual_preconditioned_norm = ti.sqrt(zTr)
        for cnt in range(max_iterations):
            if residual_preconditioned_norm < terminate_residual:
                if verbose:
                    print("CG terminates at", cnt, "; residual =",
                          residual_preconditioned_norm)
                return cnt
            self.computAp(self.p)
            self.project(self.Ap)
            self.alpha[None] = zTr / self.dotProduct(self.Ap, self.p)

            self.update_sol()
            self.update_residual()

            self.precondition()
            self.project(self.q)

            zTrk_last = zTr
            zTr = self.dotProduct(self.q, self.r)
            self.beta[None] = zTr / zTrk_last

            self.update_p()

            residual_preconditioned_norm = ti.sqrt(zTr)
        if verbose:
            print("ConjugateGradient max iterations reached, iter =",
                  max_iterations, "; residual =", residual_preconditioned_norm)
            
        return max_iterations
