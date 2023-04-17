
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import functools
import matplotlib.patches as patches
import math

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
# THRESHOLD = 0.02
THRESHOLD = 0 
SIZE_THRESHOLD = 0.05
TRACE_FILES = 'gcc_trace.txt', 'jpeg_trace.txt', 'perl_trace.txt'
REPETITIONS = 20
THRESHOLD_TIME = 30
IGNORE_ALL = False
# IGNORE_ALL = True
# read_default_data = True
read_default_data = False
baselines = ["benchmarks5.csv", "benchmarks6.csv", "benchmarks11.csv", "benchmarks13.csv", "benchmarks12.csv"]

Ignored_algorithms = ["GShare_ML-running_mean", "GShare_ML-nearest_pattern2", "GShare_ML-logistic2",
    "Tournament", "GShare_ML-logistic", "PShare", "S_Clustering-skmean" ]

columns=["Tracefile", "Predictor", "Predictor Arguments", "Misprediction Rate", "Accuracy",
"Precision", "Recall", "F1", "Runtime", "TP", 
"TN", "FP", "FN", 'Size']

colors = ["blue", "green", "teal", "red", "purple", "violet", "navy", 
    "cyan", "crimson", "dodgerblue", "peru", "lime", 
    "black", "gold", "yellow", "fuchsia", "aqua", "orangered", "deeppink", "orange"]

metrics = ["Accuracy", "Misprediction Rate", 'F1', 'Runtime', "Predictor Arguments", 'Size']
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

def calc_size(x):
    if x["Predictor"]  in Ignored_algorithms:
        x["Size"] = -1
        return x
    if x["Predictor"] == "Smith":
        m = int(x["Predictor Arguments"])
        x["Size"] = m 
        return x
    if x["Predictor"] == "Bimodal":
        m = int(x["Predictor Arguments"])
        x["Size"] =  3 * 2**m
        return x
    if x["Predictor"] == "Tage":
        x["Size"] = -1
        return x
    if x["Predictor"] == "YehPatt":
        m,n = [int(e) for e in x["Predictor Arguments"].split(",")]
        x["Size"] = (n * 2**m) + (2 * 2**n)
        return x
    if x["Predictor"] == "GShare":
        m,n = [int(e) for e in x["Predictor Arguments"].split(",")]
        x["Size"] = n + (3 * 2**m)
        return x
    if x["Predictor"] == "PShare2":
        m,n = [int(e) for e in x["Predictor Arguments"].split(",")]
        x["Size"] = n + (((n+1) + 2 * 2**m) * 2**n)
        return x
    if x["Predictor"] == "Tournament" or x["Predictor"] == "Tournament2":
        m,n = [int(e) for e in x["Predictor Arguments"].split(",")]
        x["Size"] = (n + 3 * 2**m) + (35 + 16 * 2**m) + 2**n
        return x
    if x["Predictor"] == "GShare_ML-running_mean2":
        m,n = [int(e) for e in x["Predictor Arguments"].split(",")]
        x["Size"] = n + ( (3 + 32)* 2**m)
        return x
    if x["Predictor"] == "GShare_ML-nearest_pattern":
        m,n = [int(e) for e in x["Predictor Arguments"].split(",")]
        size = n + ( (2**2 + 2**3 + 3) * 2**m)
        x["Size"] = size
        return x
    if x["Predictor"] == "PShare2":
        m,n = [int(e) for e in x["Predictor Arguments"].split(",")]
        x["Size"] = n + (((n+1) + 2 * 2**m) * 2**n)
        return x
    if x["Predictor"] == "Hybrid":
        k, g, n, b = [int(e) for e in x["Predictor Arguments"].split(",")]
        x["Size"] = (3 * 2**b) + (n + (3 * 2**g)) + (2 * 2**k)
        return x
    if x["Predictor"] == "Running_logistic" or x["Predictor"] == "Running_Perceptron":
        m = int(x["Predictor Arguments"])
        x["Size"] = (1 + 32) * m
        return x
    if x["Predictor"] == "GShare_Perceptron":
        m,n = [int(e) for e in x["Predictor Arguments"].split(",")]
        x["Size"] = n + ( (2 * n * 32 ) * 2**m)
        return x
    
# GShare Perceptron (m, n) = n + (n * 2**m)
if read_default_data:
    results = results.apply(handle_running_algorithm, axis=1)
    results = results.apply(handle_ml, axis=1)
    results = results.apply(calc_size, axis=1)
    FINAL_DF = None
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
            filtered_df["Tracefile"] = trace
            filtered_df["Predictor"] = algo
            print(filtered_df)
            if FINAL_DF is None:
                FINAL_DF = filtered_df
            else:
                FINAL_DF = pd.concat([FINAL_DF, filtered_df], ignore_index=True)    
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

    line.axes.annotate('',
        xytext=(start_x, start_y),
        xy=(end_x, end_y),
        # arrowprops=dict(arrowstyle="->", color=color, width=width),
        arrowprops=dict(color=color, width=width, headlength=headlength, headwidth=headwidth, alpha=OPACITY),
        size=size
    )

def renaming_name(x):
    if x == "Tournament2":     
        return "Tournament"
    elif x == "PShare2":
        return "PShare"
    elif "running_mean2" in x:
        return x.replace("running_mean2", "running_mean")
    elif x == "Running_logistic":
        return "Online Logistic"
    elif x == "Running_Perceptron":
        return "Online Perceptron"
    return x
    
def plotting3(draw_arrow=False):
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
                algo = renaming_name(algo)
                labels.append(algo + f' ({max(Y):.2f}) ')
                h = Line2D([0], [0], marker='o', markersize=np.sqrt(100), color=colors[i], linestyle='None', alpha=OPACITY, markeredgecolor='black')
                legends.append(h)
                if draw_arrow:
                    for x_0, x_1, y_0, y_1 in zip(X[:-1], X[1:], Y[:-1], Y[1:]): 
                        line = plt.plot([x_0, x_1], [y_0, y_1], '-', alpha=OPACITY /2, color=colors[i])[0]
                        size = 25
                        add_arrow(line, color=colors[i], size=900, width=0, headlength=size, headwidth=size)
                else:
                    plt.plot(X, Y, '-', alpha=OPACITY / 2, color=colors[i])
        # plt.legend(legends, labels, loc="lower right", markerscale=2, scatterpoints=0, fontsize=20)
        plt.legend(legends, labels, loc="lower right", markerscale=5, scatterpoints=0, fontsize=FONTSIZE, bbox_to_anchor=(1.44, 0))
        
        plt.title("BenchMarking: " + r"$\bf{" + str(trace.replace("_", "\_")) + "}$" + f", Thres. skip {THRESHOLD}, t <{THRESHOLD_TIME}s")
        plt.xlabel("Runtime (seconds) (Avg of 20 runs)")
        plt.ylabel(f"{plotting_y} Scores")
        plt.subplots_adjust(right=0.7, left=0.05, top=0.95, bottom=0.09)
        # plt.tight_layout()
        plt.savefig(f"{trace}.png")
        plt.clf()  
        if IGNORE_ALL:
            break 

def plotting4():
    Ratios = {}
    for trace in TRACE_FILES:
        plt.grid(alpha=0.5)
        legends = []
        labels = []
        Ratios[trace] = {}
        max_ratio = -1
        max_ratio_algo = -1
        for i,algo in enumerate(Runtime[trace].keys()):
            count = 0 
            Ratios[trace][algo] = {}
            for args in Runtime[trace][algo].keys():
                count += 1
                y = Runtime[trace][algo][args][plotting_y][0]
                x = Runtime[trace][algo][args][plotting_x][0]
                ratio = y / x
                Ratios[trace][algo][args] = ratio 
                if ratio > max_ratio:
                    max_ratio_algo = (algo, args)
        print(max_ratio_algo)
    return Ratios

def plotting5(FINAL_DF, y_axis_label="acc"):
    plt.figure(figsize=(35, 30))
    for trace in TRACE_FILES:
        plt.grid(alpha=0.5)
        legends = []
        labels = []
        trace_results = FINAL_DF[FINAL_DF["Tracefile"] == trace]
        algorithms = trace_results.Predictor.unique()
        for i,algo in enumerate(algorithms):
            algorithms_trace = trace_results[trace_results.Predictor == algo]
            X = algorithms_trace.Size
            Y = algorithms_trace[plotting_y]
            Z = algorithms_trace[plotting_y] / algorithms_trace.Runtime
            X_clean = []
            Y_clean = []
            Z_clean = []
            for x,y,a,b in zip(X[:-1], X[1:], Y[:-1], Z[:-1]):
                if y- x < SIZE_THRESHOLD:
                    continue
                else:
                    X_clean.append(x)
                    Y_clean.append(a)
                    Z_clean.append(b)
            
            # plt.plot(X, Y, '-o', alpha=OPACITY / 2, color=colors[i])
            algo = renaming_name(algo)
            if y_axis_label == "acc":
                plt.plot(X_clean, Y_clean, 'o--', color=colors[i], label=algo, markersize=20, linewidth =5, markeredgecolor='black')
            else:
                plt.plot(X_clean, Z_clean, 'o--', color=colors[i], label=algo, markersize=20, linewidth =5, markeredgecolor='black')

        plt.legend()
        # plt.legend(legends, labels, loc="lower right", markerscale=5, scatterpoints=0, fontsize=FONTSIZE, bbox_to_anchor=(1.44, 0))
        
        plt.title("BenchMarking: " + r"$\bf{" + str(trace.replace("_", "\_")) + "}$" + f", Thres. skip {SIZE_THRESHOLD}, t <{THRESHOLD_TIME}s")
        plt.xlabel("Log2 (Size)")
        if y_axis_label == "acc":
            plt.ylabel(f"{plotting_y} Scores")
        else:
            plt.ylabel("Accuracy / Time Ratio")
        plt.tight_layout()
        # plt.subplots_adjust(right=0.7, left=0.05, top=0.95, bottom=0.09)
        plt.savefig(f"SIZE_{trace}.png")
        plt.clf()  
        



# plotting2(fading=False)
# plotting2(fading=True)
# plotting3(draw_arrow=False)

# conda activate bert
# python plotting.py

# ratios = plotting4()
# 
if read_default_data:
    FINAL_DF.to_csv("size.csv", index=False)


FINAL_DF = pd.read_csv("size.csv")
FINAL_DF = FINAL_DF.sort_values(by=['Size'])
FINAL_DF.Size = np.log(FINAL_DF.Size) / np.log(2)
FINAL_DF = FINAL_DF[FINAL_DF.Size < THRESHOLD_TIME ]

plotting5(FINAL_DF, y_axis_label="ratio")
# plotting5(FINAL_DF)
