# branch_predictor_python
Branch Prediction : Python 


## Smith n-bit predictor

```
python smith.py 3 traces/gcc_trace.txt
python smith.py 1 traces/jpeg_trace.txt
python smith.py 4 traces/perl_trace.txt
```

## Bimodal n-bit predictor

```
python bimodal.py 6 traces/gcc_trace.txt
python bimodal.py 12 traces/gcc_trace.txt
python bimodal.py 4 traces/jpeg_trace.txt
```

## Gshare (m,n) predictor

```
python gshare.py 9 3 traces/gcc_trace.txt
python gshare.py 14 8 traces/gcc_trace.txt
python gshare.py 11 5 traces/jpeg_trace.txt
```

