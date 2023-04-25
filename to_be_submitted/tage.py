import sys
from branch_predictor import Tage, run_predictor

m = int(sys.argv[1])
trace_file = sys.argv[2]

predictor = Tage(m)
num_predictions, num_mispredictions = run_predictor(predictor, trace_file)

misprediction_rate = 100 * num_mispredictions / num_predictions

print(f"number of predictions:		{num_predictions}")
print(f"number of mispredictions:	{num_mispredictions}")
print(f"misprediction rate:		{misprediction_rate:.2f}%")
print("FINAL TAGE CONTENTS")
for i, component in enumerate(predictor.tage_components):
    print(f"{i:<2} {component.counter}")
