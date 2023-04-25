# import sys
from baseline_predictors import Smith, run_predictor, Bimodal, GShare, Hybrid
while True: 
    input_string = input("\nType q | Q to exit\n")
    if input_string == "q" or input_string == "Q":
        quit()
    algorithm = input_string.split(" ")[1]
    if algorithm == "smith":
        write_file = False
        if len(input_string.split(" ")) == 6 or len(input_string.split(">")) == 2:
            write_file = True
            output_file = input_string.split(" ")[-1]
        _, algorithm, param, trace = input_string.split(" ")[:4]
        counter_bits = int(param)
        trace_file = trace
        predictor = Smith(counter_bits)
        num_predictions, num_mispredictions = run_predictor(predictor, trace_file)
        misprediction_rate = 100 * num_mispredictions / num_predictions
        strings = "COMMAND\n"
        strings += f"./sim smith {counter_bits} {trace_file}\n"
        strings += "OUTPUT\n"
        strings += f"number of predictions:\t\t{num_predictions}\n"
        strings += f"number of mispredictions:\t{num_mispredictions}\n"
        strings += f"misprediction rate:\t\t{misprediction_rate:.2f}%\n"
        strings += f"FINAL COUNTER CONTENT:\t\t{predictor.get_counter()}"
        if write_file:
            with open(output_file, "w") as f:
                f.write(strings)
        else:
            print(strings, end='')
    elif algorithm == "bimodal":
        write_file = False
        if len(input_string.split(" ")) == 6 or len(input_string.split(">")) == 2:
            write_file = True
            output_file = input_string.split(" ")[-1]
        _, algorithm, param, trace = input_string.split(" ")[:4]
        m = int(param)
        trace_file = trace
        predictor = Bimodal(m)
        num_predictions, num_mispredictions = run_predictor(predictor, trace_file)
        misprediction_rate = 100 * num_mispredictions / num_predictions
        strings ="COMMAND\n"
        strings +=f"./sim bimodal {m} {trace_file}\n"
        strings +="OUTPUT\n"
        strings +=f"number of predictions:\t\t{num_predictions}\n"
        strings +=f"number of mispredictions:\t{num_mispredictions}\n"
        strings +=f"misprediction rate:\t\t{misprediction_rate:.2f}%\n"
        strings +="FINAL BIMODAL CONTENTS\n"
        for i, counter in enumerate(predictor.prediction_table):
            strings +=f"{i}\t{counter}\n"
        if write_file:
            with open(output_file, "w") as f:
                f.write(strings)
        else:
            print(strings, end='')
    elif algorithm == "gshare":
        write_file = False
        if len(input_string.split(" ")) == 7 or len(input_string.split(">")) == 2:
            write_file = True
            output_file = input_string.split(" ")[-1]
        _, algorithm, m, n, trace = input_string.split(" ")[:5]
        trace_file = trace
        m = int(m)
        n = int(n)
        predictor = GShare(m, n)
        num_predictions, num_mispredictions = run_predictor(predictor, trace_file)
        misprediction_rate = 100 * num_mispredictions / num_predictions
        strings = "COMMAND\n"
        strings += f"./sim gshare {m} {n} {trace_file}\n"
        strings += "OUTPUT\n"
        strings += f"number of predictions:\t\t{num_predictions}\n"
        strings += f"number of mispredictions:\t{num_mispredictions}\n"
        strings += f"misprediction rate:\t\t{misprediction_rate:.2f}%\n"
        strings += "FINAL GSHARE CONTENTS\n"
        for i, counter in enumerate(predictor.prediction_table):
            strings += f"{i}\t{counter}\n"
        if write_file:
            with open(output_file, "w") as f:
                f.write(strings)
        else:
            print(strings, end='')  
    elif algorithm == "hybrid":    
        write_file = False
        if len(input_string.split(" ")) == 9 or len(input_string.split(">")) == 2:
            write_file = True
            output_file = input_string.split(" ")[-1]
        _, algorithm, k, m_gshare, n, m_bimodal, trace = input_string.split(" ")[:7]
        trace_file = trace
        k = int(k)
        m_gshare = int(m_gshare)
        n = int(n)
        m_bimodal = int(m_bimodal)
        predictor = Hybrid(k, m_gshare, n, m_bimodal)
        num_predictions, num_mispredictions = run_predictor(predictor, trace_file)
        misprediction_rate = 100 * num_mispredictions / num_predictions
        strings = "COMMAND\n"
        strings += f"./sim hybrid {k} {m_gshare} {n} {m_bimodal} {trace_file}\n"
        strings += "OUTPUT\n"
        strings += f"number of predictions:\t\t{num_predictions}\n"
        strings += f"number of mispredictions:\t{num_mispredictions}\n"
        strings += f"misprediction rate:\t\t{misprediction_rate:.2f}%\n"
        strings += "FINAL CHOOSER CONTENTS\n"
        for i, chooser in enumerate(predictor.chooser_table):
            strings += f"{i}\t{chooser}\n"
        strings += "FINAL GSHARE CONTENTS\n"
        for i, counter in enumerate(predictor.gshare.prediction_table):
            strings += f"{i}\t{counter}\n"
        strings += "FINAL BIMODAL CONTENTS\n"
        for i, counter in enumerate(predictor.bimodal.prediction_table):
            strings += f"{i}\t{counter}\n"
        if write_file:
            with open(output_file, "w") as f:
                f.write(strings)
        else:
            print(strings, end='')                  








