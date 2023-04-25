import sys
from baseline_predictors import Bimodal, run_predictor

QUIET = False

m = int(sys.argv[1])
trace_file = sys.argv[2]

predictor = Bimodal(m)
num_predictions, num_mispredictions = run_predictor(predictor, trace_file)

misprediction_rate = 100 * num_mispredictions / num_predictions

print("COMMAND")
print(f"./sim bimodal {m} {trace_file}")
print("OUTPUT")
print(f"number of predictions:\t\t{num_predictions}")
print(f"number of mispredictions:\t{num_mispredictions}")
print(f"misprediction rate:\t\t{misprediction_rate:.2f}%")

if QUIET:
    exit(0)

print("FINAL BIMODAL CONTENTS")
for i, counter in enumerate(predictor.prediction_table):
    print(f"{i}\t{counter}")
