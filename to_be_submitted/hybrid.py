import sys
from branch_predictor import Hybrid, run_predictor

QUIET = False

k = int(sys.argv[1])
m_gshare = int(sys.argv[2])
n = int(sys.argv[3])
m_bimodal = int(sys.argv[4])
trace_file = sys.argv[5]

predictor = Hybrid(k, m_gshare, n, m_bimodal)
num_predictions, num_mispredictions = run_predictor(predictor, trace_file)

misprediction_rate = 100 * num_mispredictions / num_predictions

print("COMMAND")
print(f"./sim hybrid {k} {m_gshare} {n} {m_bimodal} {trace_file}")
print("OUTPUT")
print(f"number of predictions:\t\t{num_predictions}")
print(f"number of mispredictions:\t{num_mispredictions}")
print(f"misprediction rate:\t\t{misprediction_rate:.2f}%")

if QUIET:
    exit(0)

print("FINAL CHOOSER CONTENTS")
for i, chooser in enumerate(predictor.chooser_table):
    print(f"{i}\t{chooser}")

print("FINAL GSHARE CONTENTS")
for i, counter in enumerate(predictor.gshare.prediction_table):
    print(f"{i}\t{counter}")

print("FINAL BIMODAL CONTENTS")
for i, counter in enumerate(predictor.bimodal.prediction_table):
    print(f"{i}\t{counter}")