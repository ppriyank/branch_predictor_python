from time import perf_counter
import os
from typing import List, Tuple
from branch_predictor import Smith, Bimodal, GShare, Hybrid, YehPatt, Tage, GShare_ML, PShare, Tournament, run_predictor, load_instructions, S_Clustering
from tqdm import tqdm

TRACE_FILES = 'gcc_trace.txt', 'jpeg_trace.txt', 'perl_trace.txt'
INSTRUCTIONS = [load_instructions(file) for file in TRACE_FILES]
OUTPUT_FILE = 'benchmarks10.csv'
REPETITIONS = 20

headers = ['Tracefile', 'Predictor', 'Predictor Arguments', 'Misprediction Rate', 'Accuracy', 'Precision', 'Recall', 'F1', 'Runtime', 'TP', 'TN', 'FP', 'FN', 'Size']

default_algorithms =["Smith", "Bimodal", "TAGE", "YehPatt", "GShare", "Hybrid"]
eddy = ["PShare", "Tournament"]
simple_ml = ["running_mean", "running_mean2", "nearest_neighbour", "nearest_neighbour2"]
clustering_ml = ["skmean2", "skmean"]
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

    args_string = ', '.join(str(arg) for arg in predictor_args)
    args_string = f'"{args_string}"'
    data = [trace_file, predictor_class.__name__, args_string, f"{misprediction_rate:.2f}", f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{runtime:.2f}",
            f"{true_positive}", f"{true_negative}", f"{false_positive}", f"{false_negative}", f"{size}"]
    data_line = ','.join(data)

    with open(OUTPUT_FILE, 'a') as f:
        f.write(data_line)
        f.write('\n')


def run_benchmark(predictor_class, predictor_args: tuple):
    for instructions, trace_file in zip(INSTRUCTIONS, TRACE_FILES):
        try:
            for _ in range(REPETITIONS):
                run_benchmark_one_trace_file(trace_file, instructions, predictor_class, predictor_args)
        except Exception as e:
            print()
            print(f"Error running {predictor_class} with arguments {predictor_args} on trace file {trace_file}")
            print(e)
            print()
            continue

# to_run = default_algorithms + eddy + simple_ml
to_run = clustering_ml

if __name__ == "__main__":
    if "Smith" in to_run:
        ### Smith ###
        for counter_bits in tqdm(range(1, 17), desc="Smith"):
            run_benchmark(Smith, (counter_bits,))
    if "Bimodal" in to_run:
        ### Bimodal ###
        for m in tqdm(range(1, 17), desc="Bimodal"):
            run_benchmark(Bimodal, (m,))

    if "TAGE" in to_run:
        ### TAGE ###
        for m in tqdm(range(2, 17), desc=f"TAGE"):
            run_benchmark(Tage, (m,))
    
    if "YehPatt" in to_run:
        ### YehPatt ###
        yehpatt_args = []
        for m in range(2, 17, 2):
            for n in range(2, 17, 2):
                yehpatt_args.append((m, n))

        for args in tqdm(yehpatt_args, desc="YehPatt"):
            run_benchmark(YehPatt, args)

    ### GShare ###
    gshare_args = []
    for m in range(2, 17, 2):
        for n in range(2, m + 1, 2):
            gshare_args.append((m, n))

    if "GShare" in to_run:
        for args in tqdm(gshare_args, desc="GShare"):
            run_benchmark(GShare, args)

    if "PShare" in to_run:
        ## PShare ###
        for args in tqdm(gshare_args, desc="PShare"):
            run_benchmark(PShare, args)

    if "Tournament" in to_run:
        ## Tournament ###
        for args in tqdm(gshare_args, desc="Tournament"):
            run_benchmark(Tournament, args)
    
    if "running_mean" in to_run:
        ## GShare: running_mean ###
        for args in tqdm(gshare_args, desc="GShare_ML Running Mean"):
            run_benchmark(GShare_ML, (*args, "running_mean"))
    
    if "running_mean2" in to_run:
        ### GShare: running_mean 2 ###
        for args in tqdm(gshare_args, desc="GShare_ML Running Mean 2"):
            run_benchmark(GShare_ML, (*args, "running_mean2"))

    if "skmean2" in to_run:
        ### ML Clustering ###
        for counter_bits in tqdm(range(1, 10), desc="SClustering"):
            run_benchmark(S_Clustering, (counter_bits, -1, "skmean2"))

    if "skmean" in to_run:
        for counter_bits in tqdm(range(1, 10), desc="SClustering"):
            for m in range(2, 21, 2):
                run_benchmark(S_Clustering, (counter_bits, m, "skmean"))

    if "nearest_neighbour" in to_run:
        ### GShare: nearest_pattern ###
        for args in tqdm(gshare_args, desc="GShare_ML Nearest Pattern"):
            run_benchmark(GShare_ML, (*args, "nearest_pattern"))
    
    if "nearest_neighbour2" in to_run:
        ### GShare: nearest_pattern 2 ###
        for args in tqdm(gshare_args, desc="GShare_ML Nearest Pattern 2"):
            run_benchmark(GShare_ML, (*args, "nearest_pattern2"))
    
    # ### GShare: logistic ###
    # for args in tqdm(gshare_args, desc="GShare_ML Logistic"):
    #     run_benchmark(GShare_ML, (*args, "logistic"))

    # ### GShare: logistic 2 ###
    # for args in tqdm(gshare_args, desc="GShare_ML Logistic 2"):
    #     run_benchmark(GShare_ML, (*args, "logistic2"))
    
    # ### GShare: Perceptron ###
    # for args in tqdm(gshare_args, desc="GShare_ML Perceptron"):
    #     run_benchmark(GShare_ML, (*args, "Perceptron"))

    # ### GShare: Perceptron 2 ###
    # for args in tqdm(gshare_args, desc="GShare_ML Perceptron 2"):
    #     run_benchmark(GShare_ML, (*args, "Perceptron2"))
    
    # ### GShare: ALMA ###
    # for args in tqdm(gshare_args, desc="GShare_ML ALMA"):
    #     run_benchmark(GShare_ML, (*args, "ALMA"))
    
    # ### GShare: ALMA 2 ###
    # for args in tqdm(gshare_args, desc="GShare_ML ALMA 2"):
    #     run_benchmark(GShare_ML, (*args, "ALMA2"))
    
    if "Hybrid" in to_run:
        ## Hybrid (takes a *very* long time) ###
        hybrid_args = []
        for k in range(11):
            for m_gshare in range(2, 17, 4):
                for n in range(2, m_gshare + 1, 4):
                    for m_bimodal in range(2, 17, 4):
                        hybrid_args.append((k, m_gshare, n, m_bimodal))

        for args in tqdm(hybrid_args, desc="Hybrid"):
            run_benchmark(Hybrid, args)
    
# python benchmarks.py
