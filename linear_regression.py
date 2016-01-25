import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

loansData['Interest.Rate.Clean'] = map(lambda x: float(x[:-1])/100, loansData['Interest.Rate'])
loansData['Loan.Length.Clean'] = map(lambda x: x[:-7], loansData['Loan.Length'])
loansData['FICO.Score'] = map(lambda x: float(x[: -(1 + x.find('-'))]), loansData['FICO.Range']) 

intrate = loansData['Interest.Rate.Clean']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

y = np.matrix(intrate).transpose()
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()
x = np.column_stack([x1,x2])
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

print(f.summary())







