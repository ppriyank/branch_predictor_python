import sys
from branch_predictor import GShare_ML, run_predictor
from time import perf_counter

m = int(sys.argv[1])
n = int(sys.argv[2])
method = sys.argv[3]
trace_file = sys.argv[4]

predictor = GShare_ML(m, n, method)
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


# print(f"number of predictions:		{num_predictions}")
# print(f"number of mispredictions:	{num_mispredictions}")
# print(f"misprediction rate:		{misprediction_rate:.2f}%")
# print("FINAL GSHARE CONTENTS")
# for i, counter in enumerate(predictor.prediction_table):
#     print(f"{i:<2} {counter}")

#     # This is just here so that the output doesn't flood the console
#     if i > 9:
#         break



# python gshare_ml.py 16 16 traces/gcc_trace.txt
# python gshare_ml.py 16 16 traces/perl_trace.txt
# python gshare_ml.py 16 16 "running_mean" traces/jpeg_trace.txt
# F1             :		    0.8988
# python gshare_ml.py 16 16 "running_mean2" traces/jpeg_trace.txt
# F1             :		    0.9527
# python gshare_ml.py 16 16 "nearest_pattern" traces/jpeg_trace.txt
# F1             :		    0.9567
# python gshare_ml.py 16 16 "nearest_pattern2" traces/jpeg_trace.txt
# F1             :		    0.9568
# python gshare_ml.py 16 16 "logistic" traces/jpeg_trace.txt
# F1             :		    0.9539
# python gshare_ml.py 16 16 "logistic2" traces/jpeg_trace.txt
# F1             :		    0.9195
# python gshare_ml.py 16 16 "Perceptron" traces/jpeg_trace.txt
# F1             :		    0.9470
# python gshare_ml.py 16 16 "Perceptron2" traces/jpeg_trace.txt
# F1             :		    0.9004
# python gshare_ml.py 16 16 "ALMA" traces/jpeg_trace.txt
# F1             :		    0.9499
# python gshare_ml.py 16 16 "ALMA2" traces/jpeg_trace.txt
# F1             :		    0.9053
# python gshare_ml.py 16 16 "GaussianNB" traces/jpeg_trace.txt
# F1             :		    0.7341
# python gshare_ml.py 16 16 "GaussianNB2" traces/jpeg_trace.txt
# F1             :		    0.8878
# python gshare_ml.py 16 16 "ExtremelyFastDecisionTreeClassifier" traces/jpeg_trace.txt
# python gshare_ml.py 16 16 "ExtremelyFastDecisionTreeClassifier2" traces/jpeg_trace.txt





