import sys
from branch_predictor import Bimodal, run_predictor

m = int(sys.argv[1])
trace_file = sys.argv[2]

predictor = Bimodal(m)
num_predictions, num_mispredictions = run_predictor(predictor, trace_file)

misprediction_rate = 100 * num_mispredictions / num_predictions

print(f"number of predictions:		{num_predictions}")
print(f"number of mispredictions:	{num_mispredictions}")
print(f"misprediction rate:		{misprediction_rate:.2f}%")
print("FINAL BIMODAL CONTENTS")
for i, counter in enumerate(predictor.prediction_table):
    print(f"{i:<2} {counter}")

    # This is just here so that the output doesn't flood the console
    if i > 9:
        break
