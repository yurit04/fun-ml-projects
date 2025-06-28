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
# TRADING, OR MARKING MARKETS IN ANY SECURITY TRADED ON ANY EXCHANGE, IS EXPLICITLY FORBIDDEN UNDER
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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DisplayPlots = True
SaveData = False
RecomputeData = False
Jump = False
AdverseSelection = False
tag = '' #'.jump' '.adverse'
DataFileName = 'Lesson2' + tag + '.dat'
# parameters ------
np.random.seed(10)
ndays = 1
deltat = 1 # seconds
StartDate = dt.datetime(2023,9,15,9,30,0) #fake date
date_list = [StartDate + dt.timedelta(seconds=x * deltat) for x in range(0, int(6.5*3600))]
date_text= [x.strftime('%Y-%m-%d %H:%M:%S') for x in date_list]
n = len(date_list)
DeltaT = n
A = 0.05
s0 = 20.
sigma = s0 * 0.02 / np.sqrt(n)
k = np.log(2.) / 0.01
q0 = 100.
gamma = 1e-2 / q0
theta0 = 0.5
h = 15 * 60
AllDeltas = [0.0025,0.005,0.0075,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.06,0.07,0.08,0.09,0.1,np.inf]
#AllDeltas = [0.0025,np.inf]
# ---------------------------

def PrintStats(delta0,Stats):
    if np.isinf(delta0):
        print("Av-St, %1.0f, %1.0f, %1.0f,%1.1f"
              % (Stats[delta0]['mu'], Stats[delta0]['sigma'], Stats[delta0]['q_std'],
                 Stats[delta0]['Sharpe']))
    else:
        print("%1.4f, %1.0f, %1.0f, %1.0f,%1.1f"
              % (delta0, Stats[delta0]['mu'], Stats[delta0]['sigma'], Stats[delta0]['q_std'],
                 Stats[delta0]['Sharpe']))

if RecomputeData:
    DailyPnls = {}
    IntradayPnls = {}
    MidPrice = {}
    Pos = {}
    Stats = {}
    print("1/2 spread,mean pnl,std pnl, std q, Sharpe")
    for delta0 in AllDeltas:
        delta_b = None
        delta_a = None
        DailyPnls[delta0] = np.zeros(ndays)
        IntradayPnls[delta0] = np.zeros(n)
        Pos[delta0] = np.zeros(n)
        MidPrice[delta0] = np.zeros(n)
        q_sum_sq = 0.
        q_sum = 0.
        for ind in range(ndays):
            u_b = np.random.uniform(0., 1., n)
            u_a = np.random.uniform(0., 1., n)
            dN_b = np.zeros(n)
            dN_a = np.zeros(n)
            N_b = np.zeros(n)
            N_a = np.zeros(n)
            q = np.zeros(n)
            dW = np.array(np.random.normal(0., 1., n))
            ds = sigma * dW
            # jump
            if Jump:
                s = s0 + np.cumsum(ds)
                ds[int(n/2)] = -0.05 * s[int(n/2)]
            s = s0 + np.cumsum(ds)

            for t in range(1, n):

                if np.isinf(delta0): #Avellaneda-Stoikov model
                    delta_b = max(0.5 * gamma * sigma * sigma * DeltaT + (1. / gamma) * np.log(
                        1. + gamma / k) + gamma * sigma * sigma * DeltaT * q[t - 1], 0.)
                    delta_a = max(0.5 * gamma * sigma * sigma * DeltaT + (1. / gamma) * np.log(
                        1. + gamma / k) - gamma * sigma * sigma * DeltaT * q[t - 1], 0.)
                else: #constant spread model
                    delta_b = delta0
                    delta_a = delta0

                if AdverseSelection:
                    # adverse selection model
                    theta = theta0 * (s[min(t + h, n - 1)] - s[t]) / (sigma * np.sqrt(h))
                    adv_factor = 1. / (1. + np.exp(-theta))
                    lambda_b = A * np.exp(-k * delta_b) * (1. - adv_factor)
                    lambda_a = A * np.exp(-k * delta_a) * adv_factor
                else:
                    lambda_b = A * np.exp(-k * delta_b)
                    lambda_a = A * np.exp(-k * delta_a)

                if u_b[t] < lambda_b * deltat:
                    dN_b[t] = 1.
                else:
                    dN_b[t] = 0.
                if u_a[t] < lambda_a * deltat:
                    dN_a[t] = 1.
                else:
                    dN_a[t] = 0.
                N_b[t] = N_b[t-1] + dN_b[t]
                N_a[t] = N_a[t-1] + dN_a[t]
                q[t] = q0 * (N_b[t] - N_a[t])

            p_b = s - delta_b
            p_a = s + delta_a
            dx = q0 * (np.multiply(p_a,dN_a) - np.multiply(p_b,dN_b))
            x = np.cumsum(dx)
            pnl = x + np.multiply(q,s)

            DailyPnls[delta0][ind] = pnl[-1] #cumulated daily pnl
            IntradayPnls[delta0] = pnl
            Pos[delta0] = q
            MidPrice[delta0] = s
            q_sum_sq += np.sum(np.multiply(q,q))
            q_sum += np.sum(q)
        q_var = (1./(ndays * n)) * q_sum_sq - ((1./(ndays * n))  * q_sum)**2
        Stats[delta0] = {}
        Stats[delta0]['mu'] = np.mean(DailyPnls[delta0])
        Stats[delta0]['sigma'] = np.std(DailyPnls[delta0])
        Stats[delta0]['Sharpe'] = Stats[delta0]['mu'] / Stats[delta0]['sigma']
        Stats[delta0]['q_std'] = np.sqrt(q_var)
        PrintStats(delta0,Stats)
    if SaveData:
        # save data ------------------------------------
        DataToSave = {}
        DataToSave['AllDeltas'] = AllDeltas
        DataToSave['DailyPnls'] = DailyPnls
        DataToSave['IntradayPnls'] = IntradayPnls
        DataToSave['Pos'] = Pos
        DataToSave['MidPrice'] = MidPrice
        DataToSave['Stats'] = Stats
        print("Saving data to file %s..." % (DataFileName), end='')
        file = open(DataFileName, 'wb')
        pickle.dump(DataToSave, file)
        file.close()
        print("done.")
        # -----------------------------------------------
else:
    # load data ------------------------------------
    print("Loading data from file %s..." % (DataFileName), end='')
    file = open(DataFileName, 'rb')
    Data = pickle.load(file)
    file.close()
    print("Done.")
    AllDeltas = Data['AllDeltas']
    DailyPnls = Data['DailyPnls']
    IntradayPnls = Data['IntradayPnls']
    Pos = Data['Pos']
    MidPrice = Data['MidPrice']
    Stats = Data['Stats']
    print("1/2 spread,mean pnl,std pnl, std q, Sharpe")
    for delta0 in AllDeltas:
        PrintStats(delta0, Stats)
    # -----------------------------------------------

if DisplayPlots:
    delta0 = np.inf
    mu_vals = []
    sigma_vals = []
    q_std_vals = []
    Sharpe_vals = []
    for delta in AllDeltas:
        mu_vals.append(Stats[delta]['mu'])
        sigma_vals.append(Stats[delta]['sigma'])
        q_std_vals.append(Stats[delta]['q_std'])
        Sharpe_vals.append(Stats[delta]['Sharpe'])

    '''plt.figure()
    ax = plt.gca()
    ax.ticklabel_format(axis='y', useOffset=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.plot_date(date_list, MidPrice[delta0], '-')
    plt.title('intraday mid price')

    plt.figure()
    ax = plt.gca()
    ax.ticklabel_format(axis='y', useOffset=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.plot_date(date_list, Pos[delta0], '-')
    plt.title('intraday stock inventory q_t')

    plt.figure()
    ax = plt.gca()
    ax.ticklabel_format(axis='y', useOffset=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.plot_date(date_list, IntradayPnls[delta0], '-')
    plt.title('intraday market-maker pnl')'''

    plt.figure()
    plt.plot(AllDeltas, mu_vals, '-')
    plt.xlabel('delta (half-spread)')
    plt.title('Average daily market-maker Pnl')

    plt.figure()
    plt.xlabel('delta (half-spread)')
    plt.plot(AllDeltas, sigma_vals, '-')
    plt.title('Std of daily market-maker Pnl')

    plt.figure()
    plt.xlabel('delta (half-spread)')
    plt.plot(AllDeltas, q_std_vals, '-')
    plt.title('Std of market-maker positions')

    plt.figure()
    plt.xlabel('delta (half-spread)')
    plt.plot(AllDeltas, Sharpe_vals, '-')
    plt.title('Market-maker\'s Sharpe ratio')

    plt.show()
















