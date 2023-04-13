
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl
import functools
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['font.size'] = 8
plt.rcParams["legend.labelcolor"] = "black"
plt.rcParams["legend.edgecolor"] = "black"
# plt.fig.subplots_adjust() 
plt.figure(figsize=(10, 10))

plotting_y = "f1"
plotting_x = "runtime"
plotting_size = "accuracy"

THRESHOLD = 0.02
TRACE_FILES = 'gcc_trace.txt', 'jpeg_trace.txt', 'perl_trace.txt'
REPETITIONS = 20
baselines = "benchmarks5.csv"
names=["trace_file", "predictor", "args_string", "misprediction_rate", "accuracy",
"precision", "recall", "f1", "runtime", "true_positive", 
"true_negative", "false_positive", "false_negative", 'Size']

colors = ["blue", "green", "orange", "red", "purple", "cyan", "lime", "violet", "gold", "peru", "orangered", "yellow", "dodgerblue", "black", "navy", "deeppink", "fuchsia"]
metrics = ["accuracy", "misprediction_rate", 'f1', 'runtime', "args_string"]


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



results = pd.read_csv(baselines, header=None, names=names)
Runtime =  {}     
for trace in TRACE_FILES:
    print(f"\n\n\n {trace} \n\n\n")
    Runtime[trace] = {}
    trace_results = results[results["trace_file"] == trace]
    algorithms = trace_results.predictor.unique()
    for algo in algorithms:
        print(f"\n\n {algo} \n\n")
        vals_to_be_plotted = []  
        Runtime[trace][algo] = {}
        algorithms_trace = trace_results[trace_results.predictor == algo]
        assert (algorithms_trace.groupby(['args_string']).count().false_negative == REPETITIONS).all()
        filtered_df = algorithms_trace[metrics].groupby(['args_string']).mean()
        print(filtered_df)
        indices = sorted(filtered_df.index, key=functools.cmp_to_key(compare))
        print(indices)
        for args in indices:
            vals = filtered_df[filtered_df.index == args].to_dict(orient='list')
            curr = vals[plotting_y][0]
            if vals_to_be_plotted != []:
                closest_acc = vals_to_be_plotted[min(range(len(vals_to_be_plotted)), key = lambda i: abs(vals_to_be_plotted[i]-curr))]
                diff = abs(closest_acc - curr)
                if diff < THRESHOLD:
                    continue
            vals_to_be_plotted.append(curr)
            Runtime[trace][algo][args] = vals

            
            
            
            
            


for trace in TRACE_FILES:
    plt.grid(alpha=0.5)
    plt.rcParams['font.size'] = 8
    for i,algo in enumerate(Runtime[trace].keys()):
        Y = []
        X = []
        Z = []
        for args in Runtime[trace][algo].keys():
            Y.append(Runtime[trace][algo][args][plotting_y][0])
            X.append(Runtime[trace][algo][args][plotting_x][0])
            Z.append(Runtime[trace][algo][args][plotting_size][0])
            label = algo + " " + args
            # plt.annotate(label, xy=(X[-1], Y[-1]), xycoords='data',)
        Z = np.array(Z)
        area = (500 * Z**2)  # 0 to 15 point radii        
        # plt.scatter(X, Y, s=area, c=colors, alpha=0.2)
        plt.scatter(X, Y, s=area, c=colors[i], alpha=0.5, label=algo, edgecolors='black')
    plt.rcParams['font.size'] = 16
    plt.title(f"BenchMarking:{trace}, Threshold for skipping {THRESHOLD}")
    plt.legend()
    plt.xlabel("Runtime (Avg of 20 runs)")
    plt.ylabel(f"{plotting_y} Scores")
    plt.savefig(f"{trace}.png")
    plt.clf()  
    

