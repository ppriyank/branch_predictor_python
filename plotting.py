
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import functools
import matplotlib.patches as patches

FONTSIZE = 50
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams["legend.labelspacing"] = 0.4
plt.rcParams["legend.labelcolor"] = "black"
plt.rcParams["legend.edgecolor"] = "black"
plt.figure(figsize=(55, 30))

# plotting_y = "F1"
plotting_y = "Accuracy"
plotting_x = "Runtime"
plotting_size = "Accuracy"

EPSILON = 0.01
OPACITY = 0.90
THRESHOLD = 0.02
TRACE_FILES = 'gcc_trace.txt', 'jpeg_trace.txt', 'perl_trace.txt'
REPETITIONS = 20
THRESHOLD_TIME = 10
IGNORE_ALL = False
# IGNORE_ALL = True
baselines = ["benchmarks5.csv", "benchmarks6.csv", "benchmarks11.csv", "benchmarks13.csv", "benchmarks12.csv"]

Ignored_algorithms = ["GShare_ML-running_mean", "GShare_ML-nearest_pattern2", "GShare_ML-logistic2",
    "Tournament", "GShare_ML-logistic", "PShare"  ]

columns=["Tracefile", "Predictor", "Predictor Arguments", "Misprediction Rate", "Accuracy",
"Precision", "Recall", "F1", "Runtime", "TP", 
"TN", "FP", "FN", 'Size']

colors = ["blue", "green", "teal", "red", "purple", "violet", "navy", 
    "cyan", "deeppink", "dodgerblue", "peru", "lime", 
    "black", "gold", "aqua", "fuchsia", "crimson", "orangered", "yellow", "orange"]

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
            elif x[2] != y[2]:
                return int(x[2]) - int(y[2])
            else:
                if x[3] != y[3]:
                    return int(x[3]) - int(y[3])
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
    if IGNORE_ALL:
        if len(Runtime) > 1:
            break 
    Weights[trace] = {}
    Runtime[trace] = {}
    trace_results = results[results["Tracefile"] == trace]
    algorithms = trace_results.Predictor.unique()
    for algo in algorithms:
        if algo in Ignored_algorithms:
            continue 
        if IGNORE_ALL:
            if len(Runtime[trace]) > 1:
                break 

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
                diff = abs(closest_y - curr_y) 
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
            count = 0 
            for args in Runtime[trace][algo].keys():
                count += 1
                y = Runtime[trace][algo][args][plotting_y][0]
                x = Runtime[trace][algo][args][plotting_x][0]
                Z = Runtime[trace][algo][args][plotting_size][0]
                X.append(x)
                Y.append(y)
                opacity = indicator(args)
                label = algo + " " + args
                area = max((3000 * Z**2), 200)
                if fading:
                    plt.scatter(x, y, s=area, c=colors[i], alpha=min(Weights[trace][algo] / opacity, 1), edgecolors='black')    
                else:
                    plt.scatter(x, y, s=area, c=colors[i], alpha=OPACITY, edgecolors='black')    
            if count != 0:
                labels.append(algo + f' ({max(Y):.2f}) ')
                h = Line2D([0], [0], marker='o', markersize=np.sqrt(100), color=colors[i], linestyle='None', alpha=OPACITY, markeredgecolor='black')
                legends.append(h)
                plt.plot(X, Y, '-', alpha=OPACITY /2, color=colors[i])
        # plt.legend(legends, labels, loc="lower right", markerscale=2, scatterpoints=0, fontsize=20)
        plt.legend(legends, labels, loc="lower right", markerscale=5, scatterpoints=0, fontsize=FONTSIZE, bbox_to_anchor=(1.44, 0))
        
        plt.title("BenchMarking: " + r"$\bf{" + str(trace.replace("_", "\_")) + "}$" + f", Thres. skip {THRESHOLD}, t <{THRESHOLD_TIME}s")
        plt.xlabel("Runtime (seconds) (Avg of 20 runs)")
        plt.ylabel(f"{plotting_y} Scores")
        plt.subplots_adjust(right=0.7, left=0.05, top=0.9)
        # plt.tight_layout()
        plt.savefig(f"{trace}_F={fading}.png")
        plt.clf()  
        

def line_eq(x_0, y_0, x_1, y_1):
    if x_0 == x_1:
        return lambda x: (y_0 + y_1) / 2 + EPSILON
    else:        
        m  = (y_1 - y_0) / (x_1 - x_0)
        b = y_1 - m * x_1
        return lambda x: m * x + b

def add_arrow(line, position=None, direction='right', size=15, color=None, width=None, headlength=None, headwidth=None):
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()
    
    delta_x = (xdata[1] - xdata[0]) / 4
    delta_y = (ydata[1] - ydata[0]) / 4

    start_x = xdata[0] + delta_x
    start_y = ydata[0] + delta_y
    # start_x = xdata[0]
    # start_y = ydata[0]

    end_x = xdata[0] + 3 * delta_x
    end_y = ydata[0] + 3 * delta_y

    # start_ind = 0
    # if direction == 'right':
    #     end_ind = start_ind + 1
    # else:
    #     end_ind = start_ind - 1
    # line.axes.annotate('',
    #     xytext=(xdata[start_ind], ydata[start_ind]),
    #     xy=(xdata[end_ind], ydata[end_ind]),
    #     # arrowprops=dict(arrowstyle="->", color=color, width=width),
    #     arrowprops=dict(color=color, width=width, headlength=headlength, headwidth=headwidth),
    #     size=size
    # )
    # line.axes.annotate('',
    #     xytext=(xdata[start_ind], ydata[start_ind]),
    #     xy=(xdata.mean(), ydata.mean()),
    #     # arrowprops=dict(arrowstyle="->", color=color, width=width),
    #     arrowprops=dict(color=color, width=width, headlength=headlength, headwidth=headwidth, alpha=OPACITY),
    #     size=size
    # )
    line.axes.annotate('',
        xytext=(start_x, start_y),
        xy=(end_x, end_y),
        # arrowprops=dict(arrowstyle="->", color=color, width=width),
        arrowprops=dict(color=color, width=width, headlength=headlength, headwidth=headwidth, alpha=OPACITY),
        size=size
    )

def plotting3(fading=False):
    for trace in TRACE_FILES:
        plt.grid(alpha=0.5)
        # plt.rcParams['font.size'] = 8
        legends = []
        labels = []
        for i,algo in enumerate(Runtime[trace].keys()):
            Y = []
            X = []
            Z = []
            count = 0 
            for args in Runtime[trace][algo].keys():
                count += 1
                y = Runtime[trace][algo][args][plotting_y][0]
                x = Runtime[trace][algo][args][plotting_x][0]
                X.append(x)
                Y.append(y)
                label = algo + " " + args
                area = 3000
                plt.scatter(x, y, s=area, c=colors[i], alpha=OPACITY, edgecolors='black')    
            if count != 0:
                labels.append(algo + f' ({max(Y):.2f}) ')
                h = Line2D([0], [0], marker='o', markersize=np.sqrt(100), color=colors[i], linestyle='None', alpha=OPACITY, markeredgecolor='black')
                legends.append(h)
                # plt.plot(X, Y, '-', alpha=OPACITY /2, color=colors[i])
                for x_0, x_1, y_0, y_1 in zip(X[:-1], X[1:], Y[:-1], Y[1:]): 
                    # line = line_eq(x_0, y_0, x_1, y_1)
                    # mid_x = (x_0 + x_1) / 2
                    # mid_y = (y_0 + y_1) / 2
                    line = plt.plot([x_0, x_1], [y_0, y_1], '-', alpha=OPACITY /2, color=colors[i])[0]
                    size = 25
                    add_arrow(line, color=colors[i], size=900, width=0, headlength=size, headwidth=size)

        # plt.legend(legends, labels, loc="lower right", markerscale=2, scatterpoints=0, fontsize=20)
        plt.legend(legends, labels, loc="lower right", markerscale=5, scatterpoints=0, fontsize=FONTSIZE, bbox_to_anchor=(1.44, 0))
        
        plt.title("BenchMarking: " + r"$\bf{" + str(trace.replace("_", "\_")) + "}$" + f", Thres. skip {THRESHOLD}, t <{THRESHOLD_TIME}s")
        plt.xlabel("Runtime (seconds) (Avg of 20 runs)")
        plt.ylabel(f"{plotting_y} Scores")
        plt.subplots_adjust(right=0.7, left=0.05, top=0.95, bottom=0.09)
        # plt.tight_layout()
        plt.savefig(f"{trace}_F={fading}.png")
        plt.clf()  
        if IGNORE_ALL:
            break 

# plotting2(fading=False)
# plotting2(fading=True)
plotting3(fading=False)

# conda activate bert
# python plotting.py