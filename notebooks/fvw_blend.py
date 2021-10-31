import pandas as pd

q1 = pd.read_csv('f_model_5_folds.csv')
q2 = pd.read_csv('v_model_5_folds.csv')
q3 = pd.read_csv('w_model_5_folds.csv')
q1['TST'] = (q1['TST']+q2['TST'])*0.5*0.9 + q3['TST']*0.1
q1['C'] = (q1['C']+q2['C'])*0.5*0.9 + q3['C']*0.1
q1.to_csv('fvw_blend.csv', index=False)