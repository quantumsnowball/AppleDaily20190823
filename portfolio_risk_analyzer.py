import argparse
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

'''
Example usage:
python portfolio_risk_analyzer.py \
    --tickers 0823.HK 2382.HK 0700.HK 2313.HK 0003.HK 0002.HK 0027.HK 0669.HK 0011.HK 1299.HK \
    --weights 0.55 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05
'''

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tickers', type=str, nargs='+', help='Tickers to be included in your portfolio', required=True)
parser.add_argument('-w', '--weights', type=float, nargs='+', help='Weights for each tickers correspondingly', required=True)
parser.add_argument('-s', '--start', type=str, default='20100101', help='The start date of all time series in YYYYMMDD format')
parser.add_argument('-e', '--end', type=str, default=None, help='The end date of all time series in YYYYMMDD format')
sysargs = parser.parse_args()

weights = pd.Series(sysargs.weights, index=sysargs.tickers)
assert round(weights.sum(), 2)==1.00, 'Pleaee make sure all you weights sum to 1.0'

def get_close(ticker):    
    df = pd.read_csv(f'{ticker}.csv', index_col=0, parse_dates=True)
    df = df.dropna()
    cl = df['Adj Close']
    cl.name = ticker
    return cl

def porf_mu(mu, weights):
    return np.average(list(mu.values()), weights=weights)

def porf_std(weights, sd, corr):
    terms = (weights[rT]*weights[cT]*corr.loc[rT,cT]*sd[rT]*sd[cT]
                for rT in weights.index
                    for cT in weights.index)
    return np.sqrt(sum(terms))

tickers = weights.index
cl = {t: get_close(t).loc[sysargs.start:] for t in tickers}
r = {t: np.log(cl[t]).diff() for t in tickers}
mu = {t: r[t].mean()*250 for t in tickers}
sd = {t: r[t].std()*np.sqrt(250) for t in tickers}
r_df = pd.concat(r.values(), axis=1)
corr = r_df.corr()    
pMu, pStd = porf_mu(mu, weights), porf_std(weights, sd, corr)
pSharpe = pMu/pStd

fig, axis = plt.subplots(1,1, figsize=(10,7))
axis.set_xlim(-0.05,+0.7); axis.set_ylim(-0.1,+0.7)
axis.set_xlabel('Standard Deviation'); axis.set_ylabel('Expected Return')
axis.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0%}'))
axis.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0%}'))
axis.axhline(0, c='k'); axis.axvline(0, c='k')
axis.set_title((f'Portfolio Risk Analyzer\n'
                f'Time: ({sysargs.start} - {sysargs.end})\n'
                f'Expected Return: {pMu:.1%}, Standard Deviation: {pStd:.1%}\n'
                f'Sharpe Ratio: {pSharpe:.1%}'))
for t in tickers:
    axis.scatter(sd[t], mu[t], c='b')
    axis.annotate(f'{t}: {weights[t]:.1%}', (sd[t], mu[t]))
axis.scatter(pStd, pMu, c='r', marker='X', s=200)
axis.annotate('Portfolio', (pStd, pMu))
plt.show()