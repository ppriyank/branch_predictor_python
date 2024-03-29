import sys
from branch_predictor import S_Clustering, run_predictor
from time import perf_counter

m = int(sys.argv[1])
n = int(sys.argv[2])
method = sys.argv[3]
trace_file = sys.argv[4]

predictor = S_Clustering(m, n, method)
start = perf_counter()
num_predictions, num_mispredictions, detailed_output = run_predictor(predictor, trace_file, return_detailed_output=True)
runtime = perf_counter() - start

true_positive = detailed_output[(True, True)]
true_negative = detailed_output[(False, False)]
false_positive = detailed_output[(False, True)]
false_negative = detailed_output[(True, False)]

misprediction_rate = 100 * num_mispredictions / num_predictions
accuracy = (true_positive + true_negative) / num_predictions
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1 = 2 * precision * recall / (precision + recall)
data_line = {"misprediction_rate": f"{misprediction_rate:.2f}" ,  
    "accuracy": f"{accuracy:.4f}",  "precision": f"{precision:.4f}",
    "recall": f"{recall:.4f}", "F1": f"{f1:.4f}",  "Runtime": f"{runtime:.1f}",
    }

for key in data_line:
    print(f"{key:<15}:		{data_line[key]:>10}")

# print(predictor.labels)
print(predictor.counter)


# print(f"number of predictions:		{num_predictions}")
# print(f"number of mispredictions:	{num_mispredictions}")
# print(f"misprediction rate:		{misprediction_rate:.2f}%")

# srun --pty -t30:00:00 --cpus-per-task=12 bash    
# conda activate pathak3
# cd ~/branch_predictor_python
# python s_clustering.py 5 -1 "skmean2" traces/jpeg_trace.txt
# python s_clustering.py 5 20 "skmean" traces/jpeg_trace.txt

