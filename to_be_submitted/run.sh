make

sim smith 2 traces/gcc_trace.txt
sim bimodal 5 traces/gcc_trace.txt

sim gshare 8 6 traces/gcc_trace.txt

sim hybrid 5 4 4 4 traces/gcc_trace.txt


sim smith 2 traces/gcc_trace.txt > chikka.txt
sim bimodal 5 traces/gcc_trace.txt > chikka.txt
sim gshare 8 6 traces/gcc_trace.txt > chikka.txt
sim hybrid 5 4 4 4 traces/gcc_trace.txt > chikka.txt



sim bimodal 6 gcc_trace.txt >> gcc_trace_6.txt
sim bimodal 12 gcc_trace.txt >> gcc_trace_12.txt
sim bimodal 4 jpeg_trace.txt >> jpeg_trace_4.txt
sim gshare 9 3 gcc_trace.txt >> gcc_trace_9_3.txt
sim gshare 14 8 gcc_trace.txt >> gcc_trace_14_8.txt

diff gcc_trace_6.txt validation_runs/val_bimodal_1.txt
diff gcc_trace_12.txt validation_runs/val_bimodal_2.txt
diff jpeg_trace_4.txt validation_runs/val_bimodal_3.txt
diff gcc_trace_9_3.txt validation_runs/val_gshare_1.txt
diff gcc_trace_14_8.txt validation_runs/val_gshare_2.txt

sim gshare 11 5 jpeg_trace.txt >> gcc_trace_11_5.txt
sim hybrid 8 14 10 5 gcc_trace.txt >> gcc_trace_8_14_10_5.txt
sim smith 3 gcc_trace.txt >> gcc_trace_3.txt
sim smith 1 jpeg_trace.txt >> jpeg_trace_1.txt
sim smith 4 perl_trace.txt >> perl_trace_3.txt

diff gcc_trace_11_5.txt validation_runs/val_gshare_3.txt
diff gcc_trace_8_14_10_5.txt validation_runs/val_hybrid_1.txt
diff gcc_trace_3.txt validation_runs/val_smith_1.txt
diff jpeg_trace_1.txt validation_runs/val_smith_2.txt
diff perl_trace_3.txt validation_runs/val_smith_3.txt
