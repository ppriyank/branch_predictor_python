
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import functools

plt.rcParams['xtick.labelsize'] = 32
plt.rcParams['ytick.labelsize'] = 32
plt.rcParams['font.size'] = 32
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams["legend.labelspacing"] = 0.4
plt.rcParams["legend.labelcolor"] = "black"
plt.rcParams["legend.edgecolor"] = "black"
plt.figure(figsize=(20, 20))

plotting_y = "F1"
plotting_x = "Runtime"
plotting_size = "Accuracy"

OPACITY = 0.90
THRESHOLD = 0.05
TRACE_FILES = 'gcc_trace.txt', 'jpeg_trace.txt', 'perl_trace.txt'
REPETITIONS = 20
THRESHOLD_TIME = 10
baselines = ["benchmarks5.csv", "benchmarks6.csv", "benchmarks13.csv", "benchmarks12.csv"]
# baselines = ["benchmarks6.csv"]

Ignored_algorithms = ["GShare_ML-running_mean", "GShare_ML-nearest_pattern2", "GShare_ML-logistic2",
    "Tournament", "GShare_ML-logistic"  ]

columns=["Tracefile", "Predictor", "Predictor Arguments", "Misprediction Rate", "Accuracy",
"Precision", "Recall", "F1", "Runtime", "TP", 
"TN", "FP", "FN", 'Size']

colors = ["blue", "green", "orange", "red", "purple", 
    "yellow", "navy", "cyan", "lime", "dodgerblue", "violet", 
    "gold", "deeppink", "peru", "orangered", "teal", "dodgerblue","crimson",  "black", 
    "fuchsia", "aqua", ]

metrics = ["Accuracy", "Misprediction Rate", 'F1', 'Runtime', "Predictor Arguments"]
results = pd.read_csv(baselines[0])
# results = pd.read_csv(baselines[0], header=None, names=columns)
for file in baselines[1:]:
    df = pd.read_csv(file)
    results = pd.concat([results, df], ignore_index=True)    

def compare(x,y):
    if len(x.split(",")) > 0:
        x = x.split(",")
        y = y.split(",")
        length = len(x)
        if x[0] == y[0]:
            if x[1] != y[1]:
                return int(x[1]) - int(y[1])
            else:
                import pdb
                pdb.set_trace()    
        else:
            return int(x[0]) - int(y[0])
    else:
        return int(x) - int(y) 

def custom_average(series):
    return series[REPETITIONS // 2 :].astype(float).mean()
    
def indicator(x):
    if len(x.split(",")) > 0:
        x = x.split(",")
        length = len(x)
        already = 0
        for i in range(length):
            already += int(x[i])
        return already
    else:
        return int(x)

def handle_running_algorithm(x):
    if "Running" not in x['Predictor']:
        return x
    if len(x["Predictor Arguments"].split(",")) == 3:
        splits = x["Predictor Arguments"].split(",")
        splits = [float(e.strip()) for e in splits]
        x["Predictor Arguments"] =str(int(splits[0]))
        return x
    else:
        return x

def handle_ml(x):
    # print(x["Predictor Arguments"].split(","), )
    if len(x["Predictor Arguments"].split(",")) == 3:
        x['Predictor'] += "-" + x["Predictor Arguments"].split(",")[-1].strip()
        x["Predictor Arguments"] = ",".join(x["Predictor Arguments"].split(",")[:-1])
        return x
    else:
        return x


results = results.apply(handle_running_algorithm, axis=1)
results = results.apply(handle_ml, axis=1)
Runtime =  {}     
Weights = {}
for trace in TRACE_FILES:
    Weights[trace] = {}
    Runtime[trace] = {}
    trace_results = results[results["Tracefile"] == trace]
    algorithms = trace_results.Predictor.unique()
    for algo in algorithms:
        if algo in Ignored_algorithms:
            continue 
        print(f"\n\n {algo} \n\n")
        Weights[trace][algo] = 1000
        Vals_Y_to_be_plotted = []  
        Vals_X_to_be_plotted = []  
        Runtime[trace][algo] = {}
        algorithms_trace = trace_results[trace_results.Predictor == algo]
        if algo != "PShare" and algo != 'S_Clustering-skmean':
            assert (algorithms_trace.groupby(['Predictor Arguments']).count().FN == REPETITIONS).all()
        # filtered_df = algorithms_trace[metrics].groupby(['args_string']).mean()
        filtered_df = algorithms_trace[metrics].groupby(['Predictor Arguments']).agg(custom_average)
        print(filtered_df)
        
        indices = sorted(filtered_df.index, key=functools.cmp_to_key(compare))
        for args in indices:
            Weights[trace][algo] = min( Weights[trace][algo], indicator(args))
            vals = filtered_df[filtered_df.index == args].to_dict(orient='list')
            curr_y = vals[plotting_y][0]
            curr_x = vals[plotting_x][0]
            if curr_x > THRESHOLD_TIME:
                continue 
            if Vals_Y_to_be_plotted != []:
                closest_y = Vals_Y_to_be_plotted[min(range(len(Vals_Y_to_be_plotted)), key = lambda i: abs(Vals_Y_to_be_plotted[i]-curr_y))]
                closest_x = Vals_X_to_be_plotted[min(range(len(Vals_X_to_be_plotted)), key = lambda i: abs(Vals_X_to_be_plotted[i]-curr_x))]
                diff = abs(closest_y - curr_y) + abs(closest_x - curr_x)
                if diff < THRESHOLD:
                    continue
            Vals_Y_to_be_plotted.append(curr_y)
            Vals_X_to_be_plotted.append(curr_x)
            Runtime[trace][algo][args] = vals

def plotting1():
    for trace in TRACE_FILES:
        plt.grid(alpha=0.5)
        # plt.rcParams['font.size'] = 8
        legends = []
        labels = []
        for i,algo in enumerate(Runtime[trace].keys()):
            Y = []
            X = []
            Z = []
            labels.append(algo)
            for args in Runtime[trace][algo].keys():
                Y.append(Runtime[trace][algo][args][plotting_y][0])
                X.append(Runtime[trace][algo][args][plotting_x][0])
                Z.append(
                    Runtime[trace][algo][args][plotting_size][0]
                )
                label = algo + " " + args
                # plt.annotate(label, xy=(X[-1], Y[-1]), xycoords='data',)
            Z = np.array(Z)
            area = np.clip((500 * Z**2), 50, a_max=None)
            # plt.scatter(X, Y, s=area, c=colors[i], alpha=0.8, label=algo, edgecolors='black')
            plt.scatter(X, Y, s=area, c=colors[i], alpha=OPACITY, edgecolors='black')
            # Create dummy Line2D objects for legend
            h = Line2D([0], [0], marker='o', markersize=np.sqrt(100), color=colors[i], linestyle='None', alpha=OPACITY, markeredgecolor='black')
            legends.append(h)
        plt.legend(legends, labels, loc="lower right", markerscale=2, scatterpoints=0, fontsize=20)
        # plt.rcParams['font.size'] = 30
        # import pdb
        # pdb.set_trace()
        plt.title("BenchMarking: " + r"$\bf{" + str(trace.replace("_", "\_")) + "}$" + f", Thres. skip {THRESHOLD}")
        # plt.title(f"BenchMarking:{trace}, Threshold for skipping {THRESHOLD}")
        # plt.legend()
        plt.xlabel("Runtime (seconds) (Avg of 20 runs)")
        plt.ylabel(f"{plotting_y} Scores")
        plt.savefig(f"{trace}.png")
        plt.clf()  
        


def plotting2(fading=False):
    for trace in TRACE_FILES:
        plt.grid(alpha=0.5)
        # plt.rcParams['font.size'] = 8
        legends = []
        labels = []
        for i,algo in enumerate(Runtime[trace].keys()):
            Y = []
            X = []
            Z = []
            labels.append(algo)
            for args in Runtime[trace][algo].keys():
                Y = Runtime[trace][algo][args][plotting_y][0]
                X = Runtime[trace][algo][args][plotting_x][0]
                Z = Runtime[trace][algo][args][plotting_size][0]
                opacity = indicator(args)
                label = algo + " " + args
                area = max((800 * Z**2), 100)
                if fading:
                    plt.scatter(X, Y, s=area, c=colors[i], alpha=min(Weights[trace][algo] / opacity, 1), edgecolors='black')    
                else:
                    plt.scatter(X, Y, s=area, c=colors[i], alpha=OPACITY, edgecolors='black')    
            h = Line2D([0], [0], marker='o', markersize=np.sqrt(100), color=colors[i], linestyle='None', alpha=OPACITY, markeredgecolor='black')
            legends.append(h)
        plt.legend(legends, labels, loc="lower right", markerscale=2, scatterpoints=0, fontsize=20)
        plt.title("BenchMarking: " + r"$\bf{" + str(trace.replace("_", "\_")) + "}$" + f", Thres. skip {THRESHOLD}, t <{THRESHOLD_TIME}s")
        plt.xlabel("Runtime (seconds) (Avg of 20 runs)")
        plt.ylabel(f"{plotting_y} Scores")
        plt.savefig(f"{trace}_F={fading}.png")
        plt.clf()  
        

plotting2(fading=False)
plotting2(fading=True)

# conda activate bert
# python plotting.py