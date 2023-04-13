
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

plotting_y = "f1"
plotting_x = "runtime"
plotting_size = "accuracy"

OPACITY = 0.80
THRESHOLD = 0.02
TRACE_FILES = 'gcc_trace.txt', 'jpeg_trace.txt', 'perl_trace.txt'
REPETITIONS = 20
baselines = "benchmarks5.csv"
columns=["trace_file", "predictor", "args_string", "misprediction_rate", "accuracy",
"precision", "recall", "f1", "runtime", "true_positive", 
"true_negative", "false_positive", "false_negative", 'Size']

colors = ["blue", "green", "orange", "red", "purple", "cyan", "lime", "violet", 
    "gold", "peru", "orangered", "yellow", "dodgerblue", "black", "navy", 
    "deeppink", "fuchsia", "aqua", "teal", "crimson"]

metrics = ["accuracy", "misprediction_rate", 'f1', 'runtime', "args_string"]
results = pd.read_csv(baselines, header=None, names=columns)

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
    return series[REPETITIONS // 2 :].mean()
    
Runtime =  {}     
for trace in TRACE_FILES:
    Runtime[trace] = {}
    trace_results = results[results["trace_file"] == trace]
    algorithms = trace_results.predictor.unique()
    for algo in algorithms:
        print(f"\n\n {algo} \n\n")
        Vals_Y_to_be_plotted = []  
        Vals_X_to_be_plotted = []  
        Runtime[trace][algo] = {}
        algorithms_trace = trace_results[trace_results.predictor == algo]
        assert (algorithms_trace.groupby(['args_string']).count().false_negative == REPETITIONS).all()
        # filtered_df = algorithms_trace[metrics].groupby(['args_string']).mean()
        filtered_df = algorithms_trace[metrics].groupby(['args_string']).agg(custom_average)
        print(filtered_df)
        indices = sorted(filtered_df.index, key=functools.cmp_to_key(compare))
        for args in indices:
            vals = filtered_df[filtered_df.index == args].to_dict(orient='list')
            curr_y = vals[plotting_y][0]
            curr_x = vals[plotting_x][0]
            if Vals_Y_to_be_plotted != []:
                closest_y = Vals_Y_to_be_plotted[min(range(len(Vals_Y_to_be_plotted)), key = lambda i: abs(Vals_Y_to_be_plotted[i]-curr_y))]
                closest_x = Vals_X_to_be_plotted[min(range(len(Vals_X_to_be_plotted)), key = lambda i: abs(Vals_X_to_be_plotted[i]-curr_x))]
                diff = abs(closest_y - curr_y) + abs(closest_x - curr_x)
                if diff < THRESHOLD:
                    continue
            Vals_Y_to_be_plotted.append(curr_y)
            Vals_X_to_be_plotted.append(curr_x)
            Runtime[trace][algo][args] = vals



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
    plt.title(f"BenchMarking:{trace}, Threshold for skipping {THRESHOLD}")
    # plt.legend()
    plt.xlabel("Runtime (seconds) (Avg of 20 runs)")
    plt.ylabel(f"{plotting_y} Scores")
    plt.savefig(f"{trace}.png")
    plt.clf()  
    

