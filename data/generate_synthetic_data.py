from __future__ import print_function
import sys
import random
import numpy as np

RANGE=1000

if len(sys.argv) != 4:
    print("Usage: ./generate_synthetic_data.py n_examples(n_rows) model_size(n_cols) sparsity_factor")
    exit(0)

n_rows, n_cols, sparsity_factor = [float(x) for x in sys.argv[1:]]
nnz_elements = int((n_rows * n_cols) * sparsity_factor)

A = np.zeros((n_rows, n_cols))
b = np.zeros((n_rows))

elements = []

for i in range(nnz_elements):
    rand_row = random.randint(0, n_rows-1)
    rand_col = random.randint(0, n_cols-1)
    rand_value = random.uniform(-RANGE,RANGE)
    A[rand_row][rand_col] = rand_value
    elements.append((rand_row, rand_col, rand_value))

for i in range(int(n_rows)):
    b[i] = random.uniform(-RANGE,RANGE)

results = np.linalg.lstsq(A, b)
v = results[0]
resids = results[1]
print("Residual: %f" % resids)

f_matrix = open("data/synthetic_data_A.dat", "w")
f_b = open("data/synthetic_data_b.dat", "w")
f_answer = open("data/synthetic_data_answer.dat", "w")

for element in elements:
    print("%d %d %f" % (element[0], element[1], element[2]), file=f_matrix)
for element in b:
    print("%f" % element, file=f_b)
for element in v:
    print("%f" % element, file=f_answer)

f_matrix.close()
f_b.close()
f_answer.close()
