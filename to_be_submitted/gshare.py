import sys
from branch_predictor import GShare, run_predictor

QUIET = False

m = int(sys.argv[1])
n = int(sys.argv[2])
trace_file = sys.argv[3]

predictor = GShare(m, n)
num_predictions, num_mispredictions = run_predictor(predictor, trace_file)

misprediction_rate = 100 * num_mispredictions / num_predictions

print("COMMAND")
print(f"./sim gshare {m} {n} {trace_file}")
print("OUTPUT")
print(f"number of predictions:\t\t{num_predictions}")
print(f"number of mispredictions:\t{num_mispredictions}")
print(f"misprediction rate:\t\t{misprediction_rate:.2f}%")

if QUIET:
    exit(0)

print("FINAL GSHARE CONTENTS")
for i, counter in enumerate(predictor.prediction_table):
    print(f"{i}\t{counter}")
