import os, os.path
import win32com.client
import pandas as pd
import xlwings as xw
import datetime as dt
import dateutil.relativedelta as relD
import time

#-------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------

def runKDSQuery(spreadsheet):
    if os.path.exists(spreadsheet):
        xl = win32com.client.DispatchEx("Excel.Application")
        wb = xl.Workbooks.Open(os.path.abspath(spreadsheet))
        xl.Application.Run("'" + spreadsheet + "'!Module1.run_query")
        wb.Close(True)
        xl.Application.Quit()  
        del xl
    else:
        print("Couldn't fine spreadsheet named " + spreadsheet)
        
def editQuery(newQuery,spreadsheet):
    app = xw.App(visible=False)
    wb = xw.Book(spreadsheet)
    inputSheet = wb.sheets['Input']
    inputSheet.range('Query').value = newQuery
    wb.save()
    wb.close()
    app.kill()

def pickupResults(spreadsheet,outCSV='out.csv',mode='w',outputSheet="Output",resultsLabel=""):
    xlsx    = pd.ExcelFile(spreadsheet)
    output  = pd.read_excel(xlsx,sheet_name=outputSheet)
    output['Label'] = resultsLabel
    if os.path.exists(outCSV) == False:
        output.to_csv(outCSV,mode='w',index=False)
    else:
        if mode == 'w':
            output.to_csv(outCSV,mode='w',index=False)
        elif mode == 'a':
            output.to_csv(outCSV,mode='a',index=False,header=False)
        else:
            print("unknown to_csv writing mode: " + mode)            

#-------------------------------------------------------------------------------------------------------------------------------------------------
# Type=PoolByPool,x=poolnumber, NonBlank=yes/y=cusip: StatePoolPctAK: StatePoolPctAL: StatePoolPctAR: StatePoolPctAZ: StatePoolPctCA: StatePoolPctCO: StatePoolPctCT: StatePoolPctDC: StatePoolPctDE: StatePoolPctFL: StatePoolPctGA: StatePoolPctGU: StatePoolPctHI: StatePoolPctIA: StatePoolPctID: StatePoolPctIL: StatePoolPctIN: StatePoolPctKS: StatePoolPctKY: StatePoolPctLA: StatePoolPctMA: StatePoolPctMD: StatePoolPctME: StatePoolPctMI: StatePoolPctMN: StatePoolPctMO: StatePoolPctMS: StatePoolPctMT: StatePoolPctNC: StatePoolPctND: StatePoolPctNE: StatePoolPctNH: StatePoolPctNJ: StatePoolPctNM: StatePoolPctNV: StatePoolPctNY: StatePoolPctOH: StatePoolPctOK: StatePoolPctOR: StatePoolPctPA: StatePoolPctPR: StatePoolPctRI: StatePoolPctSC: StatePoolPctSD: StatePoolPctTN: StatePoolPctTX: StatePoolPctUT: StatePoolPctVA: StatePoolPctVI: StatePoolPctVT: StatePoolPctWA: StatePoolPctWI: StatePoolPctWV: StatePoolPctWY, Agency=fn, MortgageType=fix, Program=30, DateWindowPeriod=201001: current, Issuance=201001, poolType=bottom,
def geoPctQuery(observationWindow,issueWindow):
    print(observationWindow + ' | ' + issueWindow)
    return "Type=PoolByPool,x=poolnumber, NonBlank=yes/y=cusip: StatePoolPctAK: StatePoolPctAL: StatePoolPctAR: StatePoolPctAZ: StatePoolPctCA: StatePoolPctCO: StatePoolPctCT: StatePoolPctDC: StatePoolPctDE: StatePoolPctFL: StatePoolPctGA: StatePoolPctGU: StatePoolPctHI: StatePoolPctIA: StatePoolPctID: StatePoolPctIL: StatePoolPctIN: StatePoolPctKS: StatePoolPctKY: StatePoolPctLA: StatePoolPctMA: StatePoolPctMD: StatePoolPctME: StatePoolPctMI: StatePoolPctMN: StatePoolPctMO: StatePoolPctMS: StatePoolPctMT: StatePoolPctNC: StatePoolPctND: StatePoolPctNE: StatePoolPctNH: StatePoolPctNJ: StatePoolPctNM: StatePoolPctNV: StatePoolPctNY: StatePoolPctOH: StatePoolPctOK: StatePoolPctOR: StatePoolPctPA: StatePoolPctPR: StatePoolPctRI: StatePoolPctSC: StatePoolPctSD: StatePoolPctTN: StatePoolPctTX: StatePoolPctUT: StatePoolPctVA: StatePoolPctVI: StatePoolPctVT: StatePoolPctWA: StatePoolPctWI: StatePoolPctWV: StatePoolPctWY, Agency=fn, MortgageType=fix, Program=30, DateWindowPeriod=" + observationWindow + ", Issuance=" + issueWindow + ", poolType=bottom,"
#-------------------------------------------------------------------------------------------------------------------------------------------------
# Type=PoolByPool,x=poolnumber, NonBlank=yes/y=cpr1: cpr3: cpr6: cpr12: cpr24: cprlife: smm: DayCount: origb: currb: prevb: paydown: Prepay: factor: ocoupon: coupon: owac: wac: wam: age: aols: waols: onloans: cnloans: pcnloans: ppnloans: osato: csato: oltv: cltv: %cltv_80: %cltv_105: %cltv_125: %ccltv_80: %ccltv_105: %ccltv_125: fico: spread: %FedHold: %CMOHold: ODTI: CODTI: %CashWindow: %Majors: cusip: prefix: FnBrk_OCLTV: FnBrk_CCLTV: PurpPctpurchase: PurpPctrefi: ChannelPctBroker: ChannelPctCorr: ChannelPctRetail: OccPctinvestor: OccPctowner: PropUnitsPct2-4, Agency=fn, MortgageType=fix, Program=30, DateWindowPeriod=201901: 201912, Issuance=201901, poolType=bottom,
def poolAttributesQuery(observationWindow,issueWindow):
    print(observationWindow + ' | ' + issueWindow)
    return "Type=PoolByPool,x=poolnumber, NonBlank=yes/y=cusip: prefix: spread: cpr1: cpr3: cpr6: cpr12: cpr24: cprlife: smm: DayCount: origb: currb: prevb: paydown: Prepay: factor: ocoupon: coupon: owac: wac: wam: age: aols: waols: onloans: cnloans: pcnloans: ppnloans: osato: csato: oltv: cltv: %cltv_80: %cltv_105: %cltv_125: %ccltv_80: %ccltv_105: %ccltv_125: fico: %FedHold: %CMOHold: ODTI: CODTI: %CashWindow: %Majors: FnBrk_OCLTV: FnBrk_CCLTV: PurpPctpurchase: PurpPctrefi: ChannelPctBroker: ChannelPctCorr: ChannelPctRetail: OccPctinvestor: OccPctowner: PropUnitsPct2-4, Agency=fn, MortgageType=fix, Program=30, DateWindowPeriod=" + observationWindow + ", Issuance=" + issueWindow + ", poolType=bottom,"
#-------------------------------------------------------------------------------------------------------------------------------------------------
# Type=PoolByPool,x=poolnumber, NonBlank=yes/y=SellerPctAMRHT: SellerPctALS: SellerPctCAFULL: SellerPctCNTL: SellerPctCITIZ: SellerPct53: SellerPctFIR: SellerPctFRDOM: SellerPctGUILD: SellerPctCHASE: SellerPctLLSL: SellerPctMATRX: SellerPctNCM: SellerPctNATIONSTAR: SellerPctNRESM: SellerPctPNYMAC: SellerPctPILOSI: SellerPctQUICK: SellerPctREG: SellerPctRMSC: SellerPctUNSHFI: SellerPctWFHM: cusip: prefix, Agency=fn, MortgageType=fix, Program=30, DateWindowPeriod=current, Issuance=202001,  poolType=bottom,
def poolSellerPctQuery(observationWindow,issueWindow):
    print(observationWindow + ' | ' + issueWindow)
    return "Type=PoolByPool,x=poolnumber, NonBlank=yes/y=SellerPctAMRHT: SellerPctALS: SellerPctCAFULL: SellerPctCNTL: SellerPctCITIZ: SellerPct53: SellerPctFIR: SellerPctFRDOM: SellerPctGUILD: SellerPctCHASE: SellerPctLLSL: SellerPctMATRX: SellerPctNCM: SellerPctNATIONSTAR: SellerPctNRESM: SellerPctPNYMAC: SellerPctPILOSI: SellerPctQUICK: SellerPctREG: SellerPctRMSC: SellerPctUNSHFI: SellerPctWFHM: cusip: prefix, Agency=fn, MortgageType=fix, Program=30, DateWindowPeriod=" + observationWindow + ", Issuance=" + issueWindow + ", poolType=bottom,"

#-------------------------------------------------------------------------------------------------------------------------------------------------    

def fullQueryRun(queryBuilder,observationWindows,issueWindow,outCSV):

    starttime = time.perf_counter()
    
    if os.path.exists(outCSV):
        print("Removing ",outCSV)
        os.remove(outCSV)
        
    for observation_window in observationWindows:
        print('--------------------------------------------------------------------------------------')
        print(dt.datetime.now().strftime("%A, %B %d, %Y, %I:%M%p"))
    
        query = queryBuilder(observation_window,issueWindow)
        print(query)
        
        editQuery(query,spreadsheet);
        print('                    :  query edited, running')
        runKDSQuery(spreadsheet)
        print('                    :  query finished, collecting')
        pickupResults(spreadsheet,outCSV,mode='a',resultsLabel=issueWindow)
        print('                    :  query results appended')
        print('--------------------------------------------------------------------------------------')
    
    endtime = time.perf_counter()
    print('elapsed time        : ',"%1.1f" % (endtime - starttime),'sec /',"%1.2f" % ((endtime - starttime)/60),'min')
    print('--------------------------------------------------------------------------------------')

#------------------------------------------------------------------------------------------------------
    
#------------------------------------------------------------------------------------------------------
#---------------------------------- MAIN --------------------------------------------------------------
#------------------------------------------------------------------------------------------------------    
    
if __name__ == "__main__":
    
    spreadsheet = "Z:/Python Scripts/cpr-cdr runners/pools dataset/KDS/kds_macro.xlsm"
    data_dir    = "Z:/Python Scripts/cpr-cdr runners/pools dataset/data"
    
    todaysDate = dt.datetime.now()
        
    issue_periods = list(pd.Series(pd.date_range('2010-01-01',todaysDate,freq='M'),name='Issue Dates').apply(lambda x: x.strftime('%Y%m')))
    
    observation_periods = list(map(lambda x: str(x) + '01: ' + str(x) + '12',range(2010,2020,1)))
    observation_periods.append('current: current')

    #------------------------------------------------------------------------------------------------------  
    # download pool geographical data
    for i in range(len(issue_periods)):
        if i>107:
            issPeriod = issue_periods[i]
            year_of_issue = issPeriod[0:4]
            this_observ_periods = list()
            for ob_period in observation_periods:
                ob_period_start = ob_period.split(":")[0]
                if ob_period_start != 'current':
                    ob_period_year = ob_period_start[0:4]
                    if int(year_of_issue) <= int(ob_period_year):
                        this_observ_periods.append(ob_period)
            this_observ_periods.append(observation_periods[-1])
            #------------------------------------------------------------------------------------------------------
            fullQueryRun(geoPctQuery,this_observ_periods,issPeriod,data_dir + "/geo pct/pools_geo_pct_issued_" + issPeriod + ".csv")
            #------------------------------------------------------------------------------------------------------            
    #------------------------------------------------------------------------------------------------------
    # download pool attributes data
    for i in range(len(issue_periods)):
        if i<1e6:
            issPeriod = issue_periods[i]
            year_of_issue = issPeriod[0:4]
            this_observ_periods = list()
            for ob_period in observation_periods:
                ob_period_start = ob_period.split(":")[0]
                if ob_period_start != 'current':
                    ob_period_year = ob_period_start[0:4]
                    if int(year_of_issue) <= int(ob_period_year):
                        this_observ_periods.append(ob_period)
            this_observ_periods.append(observation_periods[-1])
            #------------------------------------------------------------------------------------------------------
            fullQueryRun(poolAttributesQuery,this_observ_periods,issPeriod,data_dir + "/pools attributes/pools_attributes_issued_" + issPeriod + ".csv")
            #------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------
    # download pool seller pct data
    for i in range(len(issue_periods)):
        if i>7:
            issPeriod = issue_periods[i]
            year_of_issue = issPeriod[0:4]
            this_observ_periods = list()
            for ob_period in observation_periods:
                ob_period_start = ob_period.split(":")[0]
                if ob_period_start != 'current':
                    ob_period_year = ob_period_start[0:4]
                    if int(year_of_issue) <= int(ob_period_year):
                        this_observ_periods.append(ob_period)
            this_observ_periods.append(observation_periods[-1])
            #------------------------------------------------------------------------------------------------------
            fullQueryRun(poolSellerPctQuery,this_observ_periods,issPeriod,data_dir + "/seller pct/pools_attributes_issued_" + issPeriod + ".csv")
            #------------------------------------------------------------------------------------------------------
    
    #------------------------------------------------------------------------------------------------------













