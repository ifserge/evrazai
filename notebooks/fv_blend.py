import pandas as pd

q1 = pd.read_csv('f_model_5_folds.csv')
q2 = pd.read_csv('v_model_5_folds.csv')
q1['TST'] = (q1['TST']+q2['TST'])*0.5
q1['C'] = (q1['C']+q2['C'])*0.5
q1.to_csv('fv_blend.csv', index=False)