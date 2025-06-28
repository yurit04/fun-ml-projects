#
# DISCLAIMER OF WARRANTIES AND LIMITATION OF LIABILITY
#
# YOU ACKNOWLEDGE AND AGREE THAT THIS SOFTWARE IS PROVIDED TO YOU ON AN "AS IS" BASIS.
# IT IS THE PROPERTY OF JB QUANTITATIVE SOLUTIONS, LLC., THEREAFTER
# "THE LICENSOR". THE LICENSOR DISCLAIMS ANY AND ALL REPRESENTATIONS AND WARRANTIES, EXPRESS
# OR IMPLIED INCLUDING (WITHOUT LIMITATION) ANY IMPLIED WARRANTIES OF MERCHANTABILITY, OR HARDWARE
# OR SOFTWARE COMPATIBILITY, OR FITNESS FOR A PARTICULAR PURPOSE OR USE, INCLUDING YOUR PARTICULAR
# BUSINESS OR INTENDED USE, OR OF THE SOFTWARE'S RELIABILITY, PERFORMANCE OR CONTINUED AVAILABILITY.
# THE LICENSOR DOES NOT REPRESENT OR WARRANT THAT THE SOFTWARE OR CALCULATIONS OR PRINTS OR EXPORT
# DATA MADE THEREOF WILL BE FREE FROM VIRUSES OR MALWARE. YOU AGREE THAT YOU ARE SOLELY RESPONSIBLE
# FOR ALL COSTS AND EXPENSES ASSOCIATED WITH RECTIFICATION, REPAIR OR DAMAGE CAUSED BY SUCH DEFECTS,
# ERRORS OR INTERRUPTIONS. FURTHER, THE LICENSOR DOES NOT REPRESENT AND WARRANT THAT THE SOFTWARE
# DOES NOT INFRINGE THE INTELLECTUAL PROPERTY RIGHT OF ANY OTHER PERSON. YOU ACCEPT RESPONSIBILITY
# TO VERIFY THAT THE SOFTWARE MEETS YOUR SPECIFIC REQUIREMENTS.
# THE LICENSOR HEREBY STATES THAT THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSES ONLY
# AND THAT ANY OTHER USE, COMMERCIAL OR PERSONAL, IS NOT ALLOWED UNDER THE PRESENT TERMS.
# IN PARTICULAR, ANY USE OF THIS SOFTWARE FOR THE PURPOSE OF DETERMINING THE PRICE, BUYING, SELLING,
# TRADING, OR MAKING MARKETS IN ANY SECURITY TRADED ON ANY EXCHANGE, IS EXPLICITLY FORBIDDEN UNDER
# THE PRESENT TERMS.
# THIS SOFTWARE IS PROVIDED FOR INFORMATIONAL PURPOSES ONLY AND YOU SHOULD NOT CONSTRUE ANY
# SUCH INFORMATION OR OTHER MATERIAL AS LEGAL TAX, INVESTMENT, FINANCIAL, OR OTHER, ADVICE.
# NOTHING IN THIS SOFTWARE OR ANY OF ITS SUPPORTING MATERIAL CONSTITUTES A SOLICITATION, RECOMMENDATION,
# OR OFFER TO BUY OR SELL ANY SECURITIES OR OTHER FINANCIAL INSTRUMENTS IN ANY JURISDICTION.
# IN NO EVENT SHALL THE LICENSOR BE LIABLE TO YOU OR ANY THIRD PARTY UNDER THIS AGREEMENT OR OTHERWISE,
# WHETHER BY WAY OF INDEMNIFICATION OR OTHERWISE, UNDER ANY THEORY OF LIABILITY WHATSOEVER (INCLUDING,
# BUT NOT LIMITED TO, NEGLIGENCE AND STRICT LIABILITY) FOR ANY DIRECT OR INDIRECT, INCIDENTAL,
# CONSEQUENTIAL, SPECIAL, PUNITIVE OR EXEMPLARY DAMAGES OR REVENUE, LOST PROFITS OR EXPECTED
# BENEFIT NOT ACHIEVED, WHETHER FORESEEABLE OR NOT, WHETHER IN AN ACTION IN CONTRACT, TORT, PRODUCT
# LIABILITY OR STATUTE OR OTHERWISE, EVEN IF THE LICENSOR HAS BEEN ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE, RELATING TO THE SOFTWARE OR YOUR USE THEREOF, OR INABILITY TO USE THE SOFTWARE
# WHETHER OR EVEN IF THE LICENSOR HAS BEEN ADVISED, KNEW OR SHOULD HAVE KNOWN OF THE POSSIBILITY
# OF SUCH LOSS OR DAMAGES AND WITHOUT REGARD AS TO WHETHER SUCH LOSS OR DAMAGE WAS FORESEEABLE
# OR NOT. WITHOUT LIMITING THE GENERALITY OF THE FOREGOING, THE LICENSOR HAS NO OBLIGATION TO PROVIDE
# AND YOU SHALL HAVE NO RIGHT TO SEEK ANY REMEDY FOR ANY DEFECT, ERROR OR FAILURE OF THE SOFTWARE.
#
# BY USING THE CODE BELOW, YOU EXPLICITLY ACKNOWLEDGE YOU UNDERSTAND THE TERMS ABOVE, WILL ABIDE
# BY THEM, AND EXONERATE THE LICENSOR, JB QUANTITATIVE SOLUTIONS, LLC, OF ANY
# LIABILITY. YOU ALSO ACKNOWLEDGE THAT THIS DISCLAIMER IS AN INTEGRAL PART OF, AND SHOULD REMAIN
# ATTACHED TO, THE CODE BELOW.
# YOU ACKNOWLEDGE THAT YOU UNDERSTAND AND AGREE TO THE DISCLAIMER OF WARRANTIES AND THE LIMITATIONS
# ON LIABILITY AND REMEDIES CONTAINED IN THIS AGREEMENT. YOU FURTHER ACKNOWLEDGE THAT THE SOFTWARE IS
# BEING PROVIDED TO YOU WITHOUT A FEE OR WITH A REASONABLE FEE, THAT THE DISCLAIMERS AND LIMITATIONS
# ARE MATERIAL PROVISIONS OF THIS AGREEMENT AND THAT THE LICENSOR WOULD NOT MAKE THE SOFTWARE AVAILABLE
# TO YOU IF SUCH DISCLAIMERS AND LIMITATIONS WERE DELETED OR MODIFIED TO BE MORE FAVORABLE TO YOU.

from __future__ import print_function
import datetime as dt
import numpy as np
import pickle
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DisplayPlots = True

# data
symbol = 'INTC'
day = '2015-01-14'
DataFileName = symbol + day + '.ts_bbo.dat'
print("Loading data from file %s..." % (DataFileName), end='')
file = open(DataFileName, 'rb')
Data = pickle.load(file)
file.close()
ts = Data['ts']
T = len(ts)
bbo = Data['bbo'] # best bid/offer
trades = Data['trades'] # side, price, size
TradeIndices = [t for t in range(T) if abs(trades[t,0]) > 0 ]
print("Found %d trades; T=%d"%(len(TradeIndices),T))
# filling missing data
for ind in [0,1]:
    t = 0
    while np.isnan(bbo[t,ind]):
        t += 1
    bbo[:t,ind] = bbo[t,ind]
for t in range(1,T):
    if np.isnan(bbo[t,0]):
        bbo[t,0] = bbo[t-1,0]
    if np.isnan(bbo[t,1]):
        bbo[t,1] = bbo[t-1,1]
trades_tt = trades[TradeIndices,:] # tt = trade time
trades_tt[:,0] = -trades_tt[:,0]# with proper convention
trades_tt[:,1] *= 1e-4
mid_tt = np.zeros(len(TradeIndices))
for k in range(len(TradeIndices)):
    t = TradeIndices[k]
    if not ( np.isnan(bbo[max(t-1,0),0]) or np.isnan(bbo[max(t-1,0),1]) ):
        mid_tt[k] = 1e-4 * 0.5*(bbo[max(t-1,0),0] + bbo[max(t-1,0),1])
    else:
        mid_tt[k] = mid_tt[k-1]
# timestamps
TimeDiff = np.zeros(len(TradeIndices))
for k in range(1,len(TradeIndices)):
    ts_diff = ts[TradeIndices[k]] - ts[TradeIndices[k-1]]
    TimeDiff[k] = ts_diff.total_seconds() * 1000 # in milliseconds
#print("mean time between trades=%1.2f ms"%(np.mean(TimeDiff)))
#print("median time between trades=%1.2f ms"%(np.median(TimeDiff)))

# C epsilon: autocorrelation of trades
AllTaus = range(1,50)
C = np.zeros(len(AllTaus))
p = 1.
n_rho = 10
for k in range(len(AllTaus)):
    tau = AllTaus[k]
    tmp = np.corrcoef(trades_tt[tau:,0],trades_tt[:-tau,0])
    C[k] = tmp[0,1]
    if k < n_rho:
        p *= C[k]
rho = p**(2/(n_rho * (n_rho + 1)))
print("\nStats:")
print("Trade sign autocorrel at one lag %1.2f%%"%(100. * rho))
rho_vals = np.zeros(len(AllTaus))
for k in range(len(AllTaus)):
    rho_vals[k] = rho**(k + 1)
# I mkt impact function
I = np.zeros(len(AllTaus))
for k in range(len(AllTaus)):
    tau = AllTaus[k]
    ret = np.zeros(len(TradeIndices))
    ret[:-tau] = mid_tt[tau:] - mid_tt[:-tau]
    I[k] = np.mean(np.multiply(ret,trades_tt[:,0]))
print("Mkt impact I[infinity] = %1.2f cents"%(100. * I[-1]))
spreads = 1e-4 * 100. * (bbo[:, 1] - bbo[:, 0])
print("mean spread=%1.2f cent(s)"%(np.mean(spreads)))
winsorize_threshold_spread = np.percentile(spreads,99)
print("Winsorized mean spread=%1.2f cent(s)"%(np.mean(np.minimum(spreads,winsorize_threshold_spread))))
print("median spread=%1.2f cent(s)"%(np.median(spreads)))

# spread histogram
Bins = list(np.linspace(0., 3., 30))
SpreadHistogram = np.array(np.histogram(spreads, bins=Bins)[0]).astype('float')
dvSpread = Bins[1] - Bins[0]
SpreadHistogram /= (dvSpread * np.sum(SpreadHistogram)) #frequencies
ClassCentersSpread = np.zeros(len(Bins) - 1)
ClassCentersSpread = 0.5 * (np.array(Bins[1:]).astype('float') + np.array(Bins[:-1]).astype('float'))

# trade size histogram
Bins = list(np.linspace(0, 500,50))
TradeSizeHistogram = np.array(np.histogram(trades_tt[:,2], bins=Bins)[0]).astype('float')
dvTrade = Bins[1] - Bins[0]
TradeSizeHistogram /= (dvTrade * np.sum(TradeSizeHistogram)) #frequencies
ClassCentersTrade = np.zeros(len(Bins) - 1)
ClassCentersTrade = 0.5 * (np.array(Bins[1:]).astype('float') + np.array(Bins[:-1]).astype('float'))

if DisplayPlots:
    t0 = 0
    t1 = T

    plt.figure()
    ax = plt.gca()
    ax.ticklabel_format(axis='y', useOffset=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.plot_date(ts[t0:t1], 1e-4 * bbo[t0:t1,:], '-')
    plt.title('best bid/offer')

    plt.figure()
    plt.bar(ClassCentersTrade, TradeSizeHistogram, width=dvTrade/ 2, color='green')
    plt.title('Trade size histogram (#stocks)')

    plt.figure()
    plt.plot(C)
    plt.plot(rho_vals)
    plt.xlabel(r'number of lags $\tau$')
    plt.legend(['data',r'theoretical $\rho^\tau$'])
    plt.title('trade sign autocorrelation')

    plt.figure()
    plt.bar(ClassCentersSpread, SpreadHistogram, width=dvSpread / 2, color='green')
    plt.title('spread histogram (cents)')

    plt.figure()
    plt.plot(100. * I)
    plt.xlabel(r'number of lags $\tau$')
    plt.title(r'Market impact function I($\tau$)')


    plt.show()














