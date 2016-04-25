import sys
import json
import numpy as np
import matplotlib.pyplot as plt

def stats(vals):
    m = min(vals)
    mm = max(vals)
    mean = sum(vals)/float(len(vals))
    stddev = np.std(vals)
    print("Mean: %f Min: %f Max: %f Stddev: %f" % (mean, m, mm, stddev))

if len(sys.argv) != 2:
    print("Usage: ./worker_times_histogram.py spark_event_file")

f = open(sys.argv[1], "r")
relevant_objs = []
for line in f:
    obj = json.loads(line)
    if obj["Event"] == "SparkListenerTaskEnd":
        relevant_objs.append(obj)

f.close()

worker_times = []
for obj in relevant_objs:
    task_info = obj["Task Info"]
    start_time = task_info["Launch Time"]
    end_time = task_info["Finish Time"]
    total_time = end_time-start_time
    execution_deserialize_time = obj["Task Metrics"]["Executor Deserialize Time"]
    execution_result_serialization_time = obj["Task Metrics"]["Result Serialization Time"]
    execution_time = obj["Task Metrics"]["Executor Run Time"]
    communication_time = total_time-(execution_deserialize_time+execution_result_serialization_time+execution_time)
    worker_times.append((total_time, execution_time, communication_time))

plt.figure()
print("Total Time Stats:")
stats([x[0] for x in worker_times])
plt.hist([x[0] for x in worker_times], 100)
plt.xlabel("ms")
plt.savefig("TotalTimeHistogram.png")

plt.figure()
print("Execution Time Stats:")
stats([x[1] for x in worker_times])
plt.hist([x[1] for x in worker_times], 100)
plt.xlabel("ms")
plt.savefig("ExecutionTimeHistogram.png")

plt.figure()
print("Communication Time Stats:")
stats([x[2] for x in worker_times])
plt.hist([x[2] for x in worker_times], 100)
plt.xlabel("ms")
plt.savefig("CommunicationTimeHistogram.png")
