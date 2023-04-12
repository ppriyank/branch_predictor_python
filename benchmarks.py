from time import perf_counter
import os
from typing import List, Tuple
from branch_predictor import Smith, Bimodal, GShare, Hybrid, YehPatt, Tage, GShare_ML, PShare, Tournament, run_predictor, load_instructions
from tqdm import tqdm

TRACE_FILES = 'gcc_trace.txt', 'jpeg_trace.txt', 'perl_trace.txt'
INSTRUCTIONS = [load_instructions(file) for file in TRACE_FILES]
OUTPUT_FILE = 'benchmarks.csv'


headers = ['Tracefile', 'Predictor', 'Predictor Arguments', 'Misprediction Rate', 'Accuracy', 'Precision', 'Recall', 'F1', 'Runtime', 'TP', 'TN', 'FP', 'FN', 'Size']
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

    try:
        size = predictor.size
    except Exception:
        size = None

    if size is None:
        size = "NA"
    else:
        size = str(size)

    args_string = ', '.join(str(arg) for arg in predictor_args)
    args_string = f'"{args_string}"'
    data = [trace_file, predictor_class.__name__, args_string, f"{misprediction_rate:.2f}", f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{runtime:.1f}",
            true_positive, true_negative, false_positive, false_negative, size]
    data_line = ','.join(data)

    with open(OUTPUT_FILE, 'a') as f:
        f.write(data_line)
        f.write('\n')


def run_benchmark(predictor_class, predictor_args: tuple):
    for instructions, trace_file in zip(INSTRUCTIONS, TRACE_FILES):
        try:
            run_benchmark_one_trace_file(trace_file, instructions, predictor_class, predictor_args)
        except Exception as e:
            print()
            print(f"Error running {predictor_class} with arguments {predictor_args} on trace file {trace_file}")
            print(e)
            print()


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

    ### YehPatt ###
    yehpatt_args = []
    for m in range(2, 21, 2):
        for n in range(2, 21, 2):
            yehpatt_args.append((m, n))

    for args in tqdm(yehpatt_args, desc="YehPatt"):
        run_benchmark(YehPatt, args)

    ### GShare ###
    gshare_args = []
    for m in range(2, 21, 2):
        for n in range(2, m + 1, 2):
            gshare_args.append((m, n))

    for args in tqdm(gshare_args, desc="GShare"):
        run_benchmark(GShare, args)

    ### PShare ###
    for args in tqdm(gshare_args, desc="PShare"):
        run_benchmark(PShare, args)

    ### Tournament ###
    for args in tqdm(gshare_args, desc="Tournament"):
        run_benchmark(Tournament, args)

    ## GShare: running_mean ###
    for args in tqdm(gshare_args, desc="GShare_ML Running Mean"):
        run_benchmark(GShare_ML, (*args, "running_mean"))

    ### GShare: running_mean 2 ###
    for args in tqdm(gshare_args, desc="GShare_ML Running Mean 2"):
        run_benchmark(GShare_ML, (*args, "running_mean2"))

    ### GShare: nearest_pattern ###
    for args in tqdm(gshare_args, desc="GShare_ML Nearest Pattern"):
        run_benchmark(GShare_ML, (*args, "nearest_pattern"))
    
    ### GShare: nearest_pattern 2 ###
    for args in tqdm(gshare_args, desc="GShare_ML Nearest Pattern 2"):
        run_benchmark(GShare_ML, (*args, "nearest_pattern2"))
    
    ### GShare: logistic ###
    for args in tqdm(gshare_args, desc="GShare_ML Logistic"):
        run_benchmark(GShare_ML, (*args, "logistic"))

    ### GShare: logistic 2 ###
    for args in tqdm(gshare_args, desc="GShare_ML Logistic 2"):
        run_benchmark(GShare_ML, (*args, "logistic2"))
    
    ### GShare: Perceptron ###
    for args in tqdm(gshare_args, desc="GShare_ML Perceptron"):
        run_benchmark(GShare_ML, (*args, "Perceptron"))

    ### GShare: Perceptron 2 ###
    for args in tqdm(gshare_args, desc="GShare_ML Perceptron 2"):
        run_benchmark(GShare_ML, (*args, "Perceptron2"))
    
    ### GShare: ALMA ###
    for args in tqdm(gshare_args, desc="GShare_ML ALMA"):
        run_benchmark(GShare_ML, (*args, "ALMA"))
    
    ### GShare: ALMA 2 ###
    for args in tqdm(gshare_args, desc="GShare_ML ALMA 2"):
        run_benchmark(GShare_ML, (*args, "ALMA2"))

    ### GShare: GaussianNB ###
    for args in tqdm(gshare_args, desc="GShare_ML GaussianNB"):
        run_benchmark(GShare_ML, (*args, "GaussianNB"))

    ### GShare: GaussianNB 2 ###
    for args in tqdm(gshare_args, desc="GShare_ML GaussianNB 2"):
        run_benchmark(GShare_ML, (*args, "GaussianNB2"))

    ### Hybrid (takes a *very* long time) ###
    hybrid_args = []
    for k in range(11):
        for m_gshare in range(2, 21, 4):
            for n in range(2, m_gshare + 1, 4):
                for m_bimodal in range(2, 21, 4):
                    hybrid_args.append((k, m_gshare, n, m_bimodal))

    for args in tqdm(hybrid_args, desc="Hybrid"):
        run_benchmark(Hybrid, args)
    