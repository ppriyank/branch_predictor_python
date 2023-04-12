import sys
from branch_predictor import Smith, run_predictor

counter_bits = int(sys.argv[1])
trace_file = sys.argv[2]

predictor = Smith(counter_bits)
num_predictions, num_mispredictions = run_predictor(predictor, trace_file)

misprediction_rate = 100 * num_mispredictions / num_predictions

print(f"number of predictions:		{num_predictions}")
print(f"number of mispredictions:	{num_mispredictions}")
print(f"misprediction rate:		{misprediction_rate:.2f}%")
print(f"FINAL COUNTER CONTENT:		{predictor.get_counter()}")
