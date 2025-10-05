import numpy as np
from scipy import stats
from math import sqrt
def simulate_ab_test(n_A=1000, p_A=0.05, n_B=1000, p_B=0.065, seed=42):
    rng = np.random.RandomState(seed); data_A = rng.binomial(1,p_A,n_A); data_B=rng.binomial(1,p_B,n_B); return data_A, data_B
def analyze_ab_test(data_A, data_B):
    conv_A=data_A.mean(); conv_B=data_B.mean(); p_pool=(data_A.sum()+data_B.sum())/(len(data_A)+len(data_B))
    se = sqrt(p_pool*(1-p_pool)*(1/len(data_A)+1/len(data_B))); z=(conv_B-conv_A)/se if se>0 else 0.0; p_value=1-stats.norm.cdf(abs(z))
    from scipy.stats import chi2_contingency; table=[[int(data_A.sum()), len(data_A)-int(data_A.sum())],[int(data_B.sum()), len(data_B)-int(data_B.sum())]]
    chi2,p_chi,_,_=chi2_contingency(table); return {"conv_A":conv_A,"conv_B":conv_B,"z":float(z),"p_value_z":float(2*p_value),"p_chi2":float(p_chi)}
