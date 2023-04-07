import sys
from branch_predictor import PShare, run_predictor

m1 = int(sys.argv[1])
m2 = int(sys.argv[2])
n = int(sys.argv[3])
k = int(sys.argv[4])
trace_file = sys.argv[5]

predictor = PShare(m1, m2, n, k)
num_predictions, num_mispredictions = run_predictor(predictor, trace_file)

misprediction_rate = 100 * num_mispredictions / num_predictions

print(f"number of predictions:		{num_predictions}")
print(f"number of mispredictions:	{num_mispredictions}")
print(f"misprediction rate:		{misprediction_rate:.2f}%")
print("FINAL PSHARE CONTENTS")
for i, counter in enumerate(predictor.prediction_table):
    print(f"{i:<2} {counter}")
