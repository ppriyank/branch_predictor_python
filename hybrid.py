import sys
from branch_predictor import Hybrid, run_predictor

k = int(sys.argv[1])
m_gshare = int(sys.argv[2])
n = int(sys.argv[3])
m_bimodal = int(sys.argv[4])
trace_file = sys.argv[5]

predictor = Hybrid(k, m_gshare, n, m_bimodal)
num_predictions, num_mispredictions = run_predictor(predictor, trace_file)

misprediction_rate = 100 * num_mispredictions / num_predictions

print(f"number of predictions:		{num_predictions}")
print(f"number of mispredictions:	{num_mispredictions}")
print(f"misprediction rate:		{misprediction_rate:.2f}%")
print("FINAL CHOOSER CONTENTS")
for i, chooser in enumerate(predictor.chooser_table):
    print(f"{i:<2} {chooser}")

    # This is just here so that the output doesn't flood the console
    if i > 9:
        break
