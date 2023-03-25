
trace_file = sys.argv[2]
function_name = sys.argv[1]
counter_bits = sys.argv[3:]
predictor = locals()[function_name](counter_bits)


num_predictions, num_mispredictions = run_predictor(predictor, trace_file)

misprediction_rate = 100 * num_mispredictions / num_predictions

print(f"number of predictions:		{num_predictions}")
print(f"number of mispredictions:	{num_mispredictions}")
print(f"misprediction rate:		{misprediction_rate:.2f}%")
print(f"FINAL COUNTER CONTENT:		{predictor.get_counter()}")
