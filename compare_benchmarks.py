import pandas as pd
import numpy as np

df = pd.read_csv('benchmarks_complete.csv')

ypdf = df[df['Predictor'] == 'YehPatt']
gsdf = df[df['Predictor'] == 'GShare']
common_arguments = np.intersect1d(ypdf['Predictor Arguments'].unique(), gsdf['Predictor Arguments'].unique())
ypdf = ypdf[ypdf['Predictor Arguments'].isin(common_arguments)]
gsdf = gsdf[gsdf['Predictor Arguments'].isin(common_arguments)]

ypdf.sort_values(by=['Predictor Arguments'])
gsdf.sort_values(by=['Predictor Arguments'])


col_to_compare = 'F1'  # 'Accuracy'

for tracefile in df['Tracefile'].unique():
    yp_f1 = ypdf[ypdf['Tracefile'] == tracefile][col_to_compare].to_numpy()
    gs_f1 = gsdf[gsdf['Tracefile'] == tracefile][col_to_compare].to_numpy()
    f1_diff = yp_f1 - gs_f1
    mean_f1_diff = np.mean(f1_diff)
    winner = 'YehPatt' if mean_f1_diff > 0 else 'GShare'
    print(f'Mean {col_to_compare} for {tracefile}: {winner} + {np.abs(mean_f1_diff):.4f}')