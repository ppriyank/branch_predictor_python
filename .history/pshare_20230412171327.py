import sys
from branch_predictor import PShare, run_predictor

m = int(sys.argv[1])
n = int(sys.argv[2])
trace_file = sys.argv[3]

predictor = PShare(m, n)
num_predictions, num_mispredictions = run_predictor(predictor, trace_file)

misprediction_rate = 100 * num_mispredictions / num_predictions

print(f"number of predictions:		{num_predictions}")
print(f"number of mispredictions:	{num_mispredictions}")
print(f"misprediction rate:		{misprediction_rate:.2f}%")
print("FINAL PSHARE CONTENTS")
for i, counter in enumerate(predictor.prediction_table):
    print(f"{i:<2} {counter}")