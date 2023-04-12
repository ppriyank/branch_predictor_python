from time import perf_counter
import os
from typing import List, Tuple
from branch_predictor import Smith, Bimodal, GShare, Hybrid, YehPatt, Tage, run_predictor, load_instructions
from tqdm import tqdm

TRACE_FILES = 'gcc_trace.txt', 'jpeg_trace.txt', 'perl_trace.txt'
INSTRUCTIONS = [load_instructions(file) for file in TRACE_FILES]
OUTPUT_FILE = 'benchmarks.csv'

headers = ['Tracefile', 'Predictor', 'Predictor Arguments', 'Misprediction Rate', 'Accuracy', 'Precision', 'Recall', 'F1', 'Runtime', 'TP', 'TN', 'FP', 'FN']
if not os.path.isfile(OUTPUT_FILE):
    header_line = ','.join(headers)
    with open(OUTPUT_FILE, 'w') as f:
        f.write(header_line)
        f.write('\n')


def run_benchmark_one_trace_file(trace_file: str, instructions: List[Tuple[int, bool]], predictor_class, predictor_args: tuple):
    predictor = predictor_class(*predictor_args)
    start = perf_counter()
    num_predictions, num_mispredictions, detailed_output = run_predictor(predictor, trace_file, True, instructions)
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
    data = [trace_file, predictor_class.__name__, args_string, f"{misprediction_rate:.2f}", f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{runtime:.1f}",
            true_positive, true_negative, false_positive, false_negative]
    data_line = ','.join(data)

    with open(OUTPUT_FILE, 'a') as f:
        f.write(data_line)
        f.write('\n')


def run_benchmark(predictor_class, predictor_args: tuple):
    for instructions, trace_file in zip(INSTRUCTIONS, TRACE_FILES):
        run_benchmark_one_trace_file(trace_file, instructions, predictor_class, predictor_args)


if __name__ == "__main__":
    ### Smith ###
    for counter_bits in tqdm(range(1, 21), desc="Smith"):
        run_benchmark(Smith, (counter_bits,))

    ### Bimodal ###
    for m in tqdm(range(1, 21), desc="Bimodal"):
        run_benchmark(Bimodal, (m,))

    ### TAGE ###
    for m in tqdm(range(2, 21), desc=f"TAGE"):
        run_benchmark(Tage, (m,))

    ### GShare ###
    gshare_args = []
    for m in range(2, 21, 2):
        for n in range(2, m + 1, 2):
            gshare_args.append((m, n))
    for args in tqdm(gshare_args, desc="GShare"):
        run_benchmark(GShare, args)

    ### YehPatt ###
    yehpatt_args = []
    for m in range(2, 21, 2):
        for n in range(2, 21, 2):
            yehpatt_args.append((m, n))
    for args in tqdm(yehpatt_args, desc="YehPatt"):
        run_benchmark(YehPatt, args)

    ### Hybrid (takes a *very* long time) ###
    hybrid_args = []
    for k in range(11):
        for m_gshare in range(2, 21, 4):
            for n in range(2, m_gshare + 1, 4):
                for m_bimodal in range(2, 21, 4):
                    hybrid_args.append((k, m_gshare, n, m_bimodal))
    for args in tqdm(hybrid_args, desc="Hybrid"):
        run_benchmark(Hybrid, args)


