from time import perf_counter
import os
from branch_predictor import Smith, Bimodal, GShare, Hybrid, YehPatt, Tage, run_predictor
from tqdm import tqdm

trace_files = 'gcc_trace.txt', 'jpeg_trace.txt', 'perl_trace.txt'
num_trace_files = len(trace_files)
output_file = 'benchmarks.csv'

headers = ['Tracefile', 'Predictor', 'Predictor Arguments', 'Misprediction Rate', 'Accuracy', 'Precision', 'Recall', 'F1', 'Runtime']
if not os.path.isfile(output_file):
    header_line = ','.join(headers)
    with open(output_file, 'w') as f:
        f.write(header_line)
        f.write('\n')


def run_benchmark(tracefile: str, predictor_class, predictor_name: str, predictor_args: tuple, outputfile: str):
    predictor = predictor_class(*predictor_args)
    start = perf_counter()
    num_predictions, num_mispredictions, detailed_output = run_predictor(predictor, tracefile, True)
    runtime = perf_counter() - start

    true_positive = detailed_output[(True, True)]
    true_negative = detailed_output[(False, False)]
    false_positive = detailed_output[(False, True)]
    false_negative = detailed_output[(True, False)]

    misprediction_rate = 100 * num_mispredictions / num_predictions
    accuracy = (true_positive + true_negative) / num_predictions
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)

    args_string = ', '.join(str(arg) for arg in predictor_args)
    args_string = f'"{args_string}"'
    data = [tracefile, predictor_name, args_string, f"{misprediction_rate:.2f}", f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{runtime:.1f}"]
    data_line = ','.join(data)

    with open(outputfile, 'a') as f:
        f.write(data_line)
        f.write('\n')


for i, trace_file in enumerate(trace_files):
    current_trace_file = i + 1

    ### Smith ###
    for counter_bits in tqdm(range(1, 6), desc=f'Smith {current_trace_file} of {num_trace_files}'):
        run_benchmark(trace_file, Smith, "Smith", (counter_bits,), output_file)

    ### Bimodal ###
    for m in tqdm(range(2, 17, 2), desc=f'Bimodal {current_trace_file} of {num_trace_files}'):
        run_benchmark(trace_file, Bimodal, "Bimodal", (m,), output_file)

    ### GShare ###
    for m in tqdm(range(2, 17, 2), desc=f'GShare {current_trace_file} of {num_trace_files}'):
        for n in range(2, m + 1, 2):
            run_benchmark(trace_file, GShare, "GShare", (m, n), output_file)

    ### Hybrid ###
    for k in tqdm(range(2, 11, 4), desc=f'Hybrid {current_trace_file} of {num_trace_files}'):
        for m_gshare in range(2, 17, 4):
            for n in range(2, m_gshare + 1, 4):
                for m_bimodal in range(2, 17, 4):
                    run_benchmark(trace_file, Hybrid, "Hybrid", (k, m_gshare, n, m_bimodal), output_file)

    ### YehPatt ###
    for m in tqdm(range(2, 17, 2), desc=f'YehPatt {current_trace_file} of {num_trace_files}'):
        for n in range(2, 11, 2):
            run_benchmark(trace_file, YehPatt, "YehPatt", (m, n), output_file)

    ### TAGE ###
    for m in tqdm(range(2, 17, 2), desc=f'TAGE {current_trace_file} of {num_trace_files}'):
        run_benchmark(trace_file, Tage, "TAGE", (m,), output_file)


