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
import numpy as np
import pickle

np.random.seed(200)
Log = False
np.set_printoptions(precision=2)
T = 200000
tick = 0.01
mu = 0.6 # cancel
nu = 0.2 # # exec
lam = 0.6 # submit
rho = 0.#0.0
v0 = 10
PreviousTrade = 1
OrderSizeUnit = 100000
MaxDepthSubmit = 5
p0 = 20
MaxOrderSize = {"limit":10,"market":10}
Gammas = {"limit":1.6,"market":1.6}
p_order = {}
for type in ["limit", "market"]:
    MaxSize = MaxOrderSize[type]
    gamma = Gammas[type]
    p_order[MaxSize] = np.zeros(MaxSize)
    sum_p = 0.
    for k in range(MaxSize):
        sum_p += 1. / pow(k + 1., gamma)
    inv_sum_p = 1. / sum_p
    for k in range(MaxSize):
        p_order[MaxSize][k] = inv_sum_p / pow(k + 1., gamma)


def RandomOrderSize(MaxSize):
    s = np.random.uniform(0., 1.)
    cumul = 0.
    for k in range(MaxSize):
        cumul += p_order[MaxSize][k]
        if s < cumul:
            ind = k
            break
    return ind + 1

def RoundTick(price):
    return tick*round((1./tick) * price)
def SortBook(BidBook,OfferBook):
    BidBook = sorted(BidBook, key=lambda x: x[0], reverse=True)
    OfferBook = sorted(OfferBook, key=lambda x: x[0])
    return BidBook,OfferBook
BidBook = [ [] for _ in range(T) ]
OfferBook = [ [] for _ in range(T) ]
BestBidSize_ts = np.zeros(T)
BestOfferSize_ts = np.zeros(T)
# book initialization
for k in range(v0):
    RandIndList = np.random.randint(1, high=4, size=1)
    level = []
    for l in range(RandIndList[0]):
        LimitOrderSize = RandomOrderSize(MaxOrderSize["limit"])
        level.append(LimitOrderSize)
    BidBook[0].append([p0 - 5 * tick - k * tick, level])
for k in range(v0):
    RandIndList = np.random.randint(1, high=4, size=1)
    level = []
    for l in range(RandIndList[0]):
        LimitOrderSize = RandomOrderSize(MaxOrderSize["limit"])
        level.append(LimitOrderSize)
    OfferBook[0].append([p0 + 5 * tick + k * tick, level])

BidBook[0] = sorted(BidBook[0] , key=lambda x: x[0],reverse=True)
OfferBook[0] = sorted(OfferBook[0] , key=lambda x: x[0])
BestBid_ts = np.zeros(T)
BestOffer_ts = np.zeros(T)
BestBid_ts[0] = BidBook[0][0][0]
BestOffer_ts[0] = OfferBook[0][0][0]
nlevels = 10
SummaryBidSizeBook = np.zeros((T,nlevels))
SummaryOfferSizeBook = np.zeros((T,nlevels))
SummaryBidBook = np.zeros((T,nlevels))
SummaryOfferBook = np.zeros((T,nlevels))

# summary stats -- initialization
t = 0
BestBid_ts[t] = BidBook[t][0][0]
BestOffer_ts[t] = OfferBook[t][0][0]
for k in range(len(BidBook[t])):
    BestBidSize_ts[t] += sum(map(abs, BidBook[t][k][1]))
for k in range(len(OfferBook[t])):
    BestOfferSize_ts[t] += sum(map(abs, OfferBook[t][k][1]))
for k in range(min(nlevels, len(BidBook[t]))):
    SummaryBidSizeBook[t, k] = sum(BidBook[t][k][1])
    SummaryBidBook[t, k] = BidBook[t][k][0]
for k in range(min(nlevels, len(OfferBook[t]))):
    SummaryOfferSizeBook[t, k] = sum(OfferBook[t][k][1])
    SummaryOfferBook[t, k] = OfferBook[t][k][0]

BuyVolume = np.zeros(T)
SellVolume = np.zeros(T)
CancelBid = np.zeros(T)
CancelOffer = np.zeros(T)
AddBid = np.zeros(T)
AddOffer = np.zeros(T)
BookDisplayWindow = 50
AllTrades_ts = []

def DisplayBook(bid,offer):
    print("------------------------------------------")
    print("Offers:")
    for k in range(len(offer) - 1, -1, -1):
        print("\t\t\t%1.2f " % (offer[k][0]), end='')
        for l in range(len(offer[k][1])):
            if offer[k][1][l] >= 0:
                print("%d "%(offer[k][1][l]),end='')
            else:
                print("*%d " % (abs(offer[k][1][l])), end='')
        print()
    print("------------------------------------------")
    print("Bids:")
    for k in range(len(bid)):
        print("\t\t\t%1.2f " % (bid[k][0]), end='')
        for l in range(len(bid[k][1])):
            if bid[k][1][l] >= 0:
                print("%d "%(bid[k][1][l]),end='')
            else:
                print("*%d " % (abs(bid[k][1][l])), end='')
        print()
    print("------------------------------------------")
    print("OfferBook=",offer)
    print("BidBook=",bid)

def SortBook(BidBook,OfferBook):
    BidBook = sorted(BidBook, key=lambda x: x[0], reverse=True)
    OfferBook = sorted(OfferBook, key=lambda x: x[0])
    return BidBook,OfferBook

def Distribution(MaxIndex,NIndices):
    # uniform case
    indices = np.random.randint(0, high=MaxIndex, size=NIndices)
    return indices

def SubmissionIndices(MaxIndex,NIndices):
    indices = Distribution(MaxIndex,NIndices)
    if NIndices == 1:
        return indices[0]
    else:
        return list(set(indices))

def CancellationIndices(MaxIndex,NIndices):
    indices = Distribution(MaxIndex,NIndices)
    if NIndices == 1:
        return indices[0]
    else:
        return indices
if Log:
    DisplayBook(BidBook[0],OfferBook[0])

for t in range(1,T):
    BidBook[t] = BidBook[t-1].copy()
    OfferBook[t] = OfferBook[t - 1].copy()
    mid = 0.5 * ( BidBook[t][0][0] + OfferBook[t][0][0] )
    if Log:
        print()
        print("t=%d ========================================================================="%(t))
        print()
        DisplayBook(BidBook[t],OfferBook[t])

    # execution --------------------------------------------
    eps = np.random.uniform(0., 1.)
    if eps < nu:
        # first, determine if buy or sell
        u = np.random.uniform(0., 1.)
        if u >= 0.5 - 0.5 * rho * PreviousTrade:
            CurrentTrade = 1
        else:
            CurrentTrade = -1
        AllTrades_ts.append(CurrentTrade)
        PreviousTrade = CurrentTrade

        if CurrentTrade == 1: #BUY
            MktOrderSize = RandomOrderSize(MaxOrderSize["market"])
            CumulExec = 0
            for level in range(len(OfferBook[t])):
                for depth in range(len(OfferBook[t][level][1])):
                    if CumulExec == MktOrderSize:
                        break
                    if CumulExec + abs(OfferBook[t][level][1][0]) <= MktOrderSize:
                        CumulExec += abs(OfferBook[t][level][1][0])
                        OfferBook[t][level][1].pop(0)
                    else:
                        # MM orders
                        sgn = np.sign(OfferBook[t][level][1][0])
                        OfferBook[t][level][1][0] = sgn * ( abs(OfferBook[t][level][1][0]) - (MktOrderSize - CumulExec) )
                        CumulExec += MktOrderSize - CumulExec

            #clean-up empty levels
            CopyOfferBook = OfferBook[t].copy()
            OfferBook[t] = []
            for level in CopyOfferBook:
                if len(level[1]) > 0:
                    OfferBook[t].append(level)
            BuyVolume[t] += CumulExec
            BidBook[t], OfferBook[t] = SortBook(BidBook[t], OfferBook[t])
            if Log:
                print()
                print("====> buy market order size %d; exec size %d"%(MktOrderSize,CumulExec))
                DisplayBook(BidBook[t],OfferBook[t])

        else: # SELL
            MktOrderSize = RandomOrderSize(MaxOrderSize["market"])
            CumulExec = 0
            for level in range(len(BidBook[t])):
                for depth in range(len(BidBook[t][level][1])):
                    if CumulExec == MktOrderSize:
                        break
                    if CumulExec + abs(BidBook[t][level][1][0]) <= MktOrderSize:
                        CumulExec += abs(BidBook[t][level][1][0])
                        BidBook[t][level][1].pop(0)
                    else:
                        # MM orders
                        sgn = np.sign(BidBook[t][level][1][0])
                        BidBook[t][level][1][0] = sgn * (abs(BidBook[t][level][1][0]) - (MktOrderSize - CumulExec))
                        CumulExec += MktOrderSize - CumulExec
            #clean-up empty levels
            CopyBidBook = BidBook[t].copy()
            BidBook[t] = []
            for level in CopyBidBook:
                if len(level[1]) > 0:
                    BidBook[t].append(level)
            SellVolume[t] += CumulExec
            BidBook[t], OfferBook[t] = SortBook(BidBook[t], OfferBook[t])
            if Log:
                print()
                print("====> sell market order size %d; exec size %d"%(MktOrderSize,CumulExec))
                DisplayBook(BidBook[t],OfferBook[t])
    # cancellation --------------------------------------------
    eps = np.random.uniform(0., 1.)
    if len(OfferBook[t]) > 0 and eps < mu:
        LevelIndex = CancellationIndices(len(OfferBook[t]),1)
        OrderIndex = CancellationIndices(len(OfferBook[t][LevelIndex][1]),1)
        limit = OfferBook[t][LevelIndex][0]
        CanceledOfferSize = OfferBook[t][LevelIndex][1][OrderIndex]
        if CanceledOfferSize > 0:# don't cancel MM orders
            CancelOffer[t] += CanceledOfferSize
            del OfferBook[t][LevelIndex][1][OrderIndex]
            if len(OfferBook[t][LevelIndex][1]) == 0:
                del OfferBook[t][LevelIndex]
            BidBook[t], OfferBook[t] = SortBook(BidBook[t], OfferBook[t])
            if Log:
                print()
                print("====> cancel offer; %1.2f, %d"%(limit,CanceledOfferSize))
                DisplayBook(BidBook[t], OfferBook[t])
    eps = np.random.uniform(0., 1.)
    if len(BidBook[t]) > 0 and eps < mu:
        LevelIndex = CancellationIndices(len(BidBook[t]),1)
        OrderIndex = CancellationIndices(len(BidBook[t][LevelIndex][1]),1)
        limit = BidBook[t][LevelIndex][0]
        CanceledBidSize = BidBook[t][LevelIndex][1][OrderIndex]
        if CanceledBidSize > 0:
            CancelBid[t] += CanceledBidSize
            del BidBook[t][LevelIndex][1][OrderIndex]
            if len(BidBook[t][LevelIndex][1]) == 0:
                del BidBook[t][LevelIndex]
            BidBook[t], OfferBook[t] = SortBook(BidBook[t], OfferBook[t])
            if Log:
                print()
                print("====> cancel bid; %1.2f, %d"%(limit,CanceledBidSize))
                DisplayBook(BidBook[t], OfferBook[t])
    # addition --------------------------------------------
    eps = np.random.uniform(0., 1.)
    if eps  < lam:
        LimitOrderSize = RandomOrderSize(MaxOrderSize["limit"])
        nticks = SubmissionIndices(MaxDepthSubmit, 1)
        limit = tick*round((1./tick)*( mid + nticks * tick ))
        IsLimitInBook = False
        for k in range(len(OfferBook[t])):
            if np.abs(limit - OfferBook[t][k][0])< 1e-8:
                IsLimitInBook = True
                OfferBook[t][k][1].append(LimitOrderSize)
                break
        if not IsLimitInBook:
            OfferBook[t].append([limit,[LimitOrderSize]])
            OfferBook[t] = sorted(OfferBook[t], key=lambda x: x[0])
        AddOffer[t] += LimitOrderSize
        BidBook[t], OfferBook[t] = SortBook(BidBook[t], OfferBook[t])
        if Log:
            print()
            print("====> add offer, limit %1.2f, size %d"%(limit,LimitOrderSize))
            DisplayBook(BidBook[t], OfferBook[t])
    eps = np.random.uniform(0., 1.)
    if eps  < lam:
        LimitOrderSize = RandomOrderSize(MaxOrderSize["limit"])
        nticks = SubmissionIndices(MaxDepthSubmit, 1)
        limit = tick*round((1./tick)*( mid - nticks * tick ))
        IsLimitInBook = False
        for k in range(len(BidBook[t])):
            if np.abs(limit - BidBook[t][k][0])< 1e-8:
                IsLimitInBook = True
                BidBook[t][k][1].append(LimitOrderSize)
                break
        if not IsLimitInBook:
            BidBook[t].append([limit,[LimitOrderSize]])
            BidBook[t] = sorted(BidBook[t], key=lambda x: x[0])
        AddBid[t] += LimitOrderSize
        BidBook[t], OfferBook[t] = SortBook(BidBook[t], OfferBook[t])
        if Log:
            print()
            print("====> add bid, limit %1.2f, size %d"%(limit,LimitOrderSize))
            DisplayBook(BidBook[t], OfferBook[t])
    # recreation --------------------------------------------
    if len(OfferBook[t]) == 0:
        for k in range(v0):
            RandIndList = np.random.randint(1, high=4, size=1)
            level = []
            for l in range(RandIndList[0]):
                LimitOrderSize = RandomOrderSize(MaxOrderSize["limit"])
                level.append(LimitOrderSize)
            OfferBook[t].append([mid + 0.05 + k * tick, level])
        BidBook[t], OfferBook[t] = SortBook(BidBook[t], OfferBook[t])
        if Log:
            print()
            print("====> Offer queue empty -- recreated")
            DisplayBook(BidBook[t], OfferBook[t])
    if len(BidBook[t]) == 0:
        for k in range(v0):
            RandIndList = np.random.randint(1, high=4, size=1)
            level = []
            for l in range(RandIndList[0]):
                LimitOrderSize = RandomOrderSize(MaxOrderSize["limit"])
                level.append(LimitOrderSize)
            BidBook[t].append([mid - 0.05 - k * tick, level])
        BidBook[t], OfferBook[t] = SortBook(BidBook[t], OfferBook[t])
        if Log:
            print()
            print("====> Bid queue empty -- recreated")
            DisplayBook(BidBook[t], OfferBook[t])
    # stats -----------------------------------------------------
    BestBid_ts[t] = BidBook[t][0][0]
    BestOffer_ts[t] = OfferBook[t][0][0]
    for k in range(len(BidBook[t])):
        BestBidSize_ts[t] += sum(map(abs,BidBook[t][k][1]))
    for k in range(len(OfferBook[t])):
        BestOfferSize_ts[t] += sum(map(abs,OfferBook[t][k][1]))

    if BestBidSize_ts[t] == 0 or BestOfferSize_ts[t] == 0:
        print("----------------------------------")
        print("ZERO QUEUE SIZE")
        DisplayBook(BidBook[t],OfferBook[t])
        print("----------------------------------")
        exit(-1)

    mid = 0.5 * ( BestBid_ts[t] + BestOffer_ts[t] )
    # summary stats
    BestBid_ts[t] = BidBook[t][0][0]
    BestOffer_ts[t] = OfferBook[t][0][0]
    for k in range(len(BidBook[t])):
        BestBidSize_ts[t] += sum(map(abs, BidBook[t][k][1]))
    for k in range(len(OfferBook[t])):
        BestOfferSize_ts[t] += sum(map(abs, OfferBook[t][k][1]))
    for k in range(min(nlevels,len(BidBook[t]))):
        SummaryBidSizeBook[t,k] = sum(BidBook[t][k][1])
        SummaryBidBook[t,k] = BidBook[t][k][0]
    for k in range(min(nlevels, len(OfferBook[t]))):
        SummaryOfferSizeBook[t,k] = sum(OfferBook[t][k][1])
        SummaryOfferBook[t,k] = OfferBook[t][k][0]

# save data ------------------------------------
DataToSave = {}
DataToSave['BidBook'] = SummaryBidBook
DataToSave['OfferBook'] = SummaryOfferBook
DataToSave['BidSizeBook'] = 1e3 * SummaryBidSizeBook
DataToSave['OfferSizeBook'] = 1e3 * SummaryOfferSizeBook
symbol = 'SYNTHETIC_DATA'
day = '2015-01-29'
AggDataFileName = symbol + day + '.aggbook.dat'
print("Saving data to file %s..." % (AggDataFileName), end='')
file = open(AggDataFileName, 'wb')
pickle.dump(DataToSave,file)
file.close()
print("done. T=%d, nlevels=%d" % (T,nlevels))
#-----------------------------------------------
