import sys
from pyspark import SparkConf, SparkContext

# Load linear regression data for matrix A and vector b
#
# For A, Expects a file with lines with format:
# y, x, value_of_matrix_A_x_y, B_[y]
#
# For b, expects 1 value per line
#
# Returns
# 1. an array with elements of the form
# [b_value, [list of tuples of form (column, value)]
# 2. number of columns (model size) of A
def load_lstsq_data(fname_A, fname_b):
    f_A = open(fname_A)
    f_b = open(fname_b)
    A = []
    n_cols = 0
    for line in f_b:
        row_element = [float(line), []]
        A.append(row_element)
    for line in f_A:
        values = line.split()
        row = int(values[0])
        col = int(values[1])
        val = float(values[2])
        A[row][1].append((col, val))
        n_cols = max(n_cols, col)
    f_A.close()
    f_b.close()
    return A, n_cols+1

# Spark distributed summing of models
def sum_models(model_1, model_2):
    summed_model = [0 for i in range(len(model_1))]
    for i in range(len(model_1)):
        summed_model[i] = model_1[i]+model_2[i]
    return summed_model

# Single node average elements of the model from src to dst
def average_model(src_model, dst_model, length):
    for i in range(len(src_model)):
        dst_model[i] = src_model[i]/float(length)

# Creates and returns a model of specified length
def initial_model(length):
    return [0 for i in range(length)]

# SGD function run on each executor
def sgd(model, row_iterator, n_epochs, stepsize):
    rows = list(row_iterator)
    for epoch in range(n_epochs):
        for row in rows:
            b_value, datapoints = row[0], row[1]
            gradient = 0
            for datapoint in datapoints:
                col, value = datapoint
                gradient += value * model[col]
            gradient = 2 * (gradient - b_value)
            for datapoint in datapoints:
                col, value = datapoint
                full_gradient = gradient * value
                model[col] -= stepsize * full_gradient
    yield model

def compute_loss(row, model):
    b_value = row[0]
    estimation = 0
    for datapoint in row[1]:
        row, value = datapoint
        estimation += value * model[row]
    return (estimation - b_value)**2

# Main function for distributed least squares regression
def SGD_Straggler(sc, data, model_size, n_workers, n_iters=20, n_epochs=10, stepsize=1e-7):
    main_model = initial_model(model_size)
    data_rdd = sc.parallelize(data, n_workers)
    for iteration in range(n_iters):
        loss = data_rdd.map(lambda x: compute_loss(x, main_model)).reduce(lambda x,y:x+y) / len(data)
        print("Loss: %f" % loss)
        summed_model = data_rdd.mapPartitions(lambda x : sgd(main_model, x, n_epochs, stepsize)).reduce(sum_models)
        average_model(summed_model, main_model, n_workers)
    print(main_model)

if __name__=="__main__":
    n_workers = sys.argv[1]
    sc = SparkContext(appName="SGD_Straggler")
    fname_A = "data/synthetic_data_A.dat"
    fname_b = "data/synthetic_data_b.dat"
    data, model_size = load_lstsq_data(fname_A, fname_b)
    SGD_Straggler(sc, data, model_size, n_workers)
    sc.stop()
