
# coding: utf-8

# # Set up

# In[2]:


import numpy as np
import pandas as pd
import calendar
import matplotlib.pyplot as plt
import os
from IPython.display import display


# # Common functions

# In[3]:


def forecast_window(start_month, start_year, num_months):
    '''
    Generate list of months in the forecasting windows and number of days in each month
    :Parameters:
    start_month, start year: the first month of the forecasting window
    num_months: the number of months in the forecasting window
    '''
    month_names, days_in_month, year = list(), list(), list()
    for period in range(num_months):
        if start_month > 12:
            start_month = 1
            start_year += 1
        month_name = calendar.month_name[start_month]   
        month_length= calendar.monthrange(start_year,start_month)[1]
        month_names.append(month_name+' '+str(start_year))
        days_in_month.append(month_length)
        start_month += 1
    return month_names, days_in_month  

def generate_timeline(file):
    get_start_month = file.parse(sheetname='get_start_month', header=None).loc[1]
    start_month = get_start_month[0]
    start_year = get_start_month[1]
    num_months = get_start_month[2]
    months = list(range(num_months))
    months_extended = [-5, -4, -3, -2, -1] + months
    window = forecast_window(start_month, start_year, num_months)
    months_names = pd.Series(window[0])
    days_in_month = pd.Series(window[1])
    return {'num_months': num_months,
            'months': months,
            'months_extended': months_extended,
            'months_names': months_names,
            'days_in_month': days_in_month}   

def printmd(string):
    '''
    Print strings formatted in markdown
    '''
    dp(Markdown(string))
    
def add_nim_actual(table):
    '''
    Add actual NIM to the output table
    :params:
    table: dataframe containing NII and ADB
    '''
    nim_actual = table['nii']/table['adb']
    table['nim_actual'] = nim_actual
    return table

def add_provision_to_toi(table):
    '''
    Add provision to TOI ratio to the output table
    :praram:
    table: dataframe containing TOI and provision
    '''
    provision_to_toi = table['provision']/table['toi']
    table['provision/toi'] = provision_to_toi
    return table    


# # Loan functions

# In[241]:


def inputs_loan(file, book):
    # Read other inputs
    other_inputs = file.parse(sheetname='other_inputs_'+book).fillna(value=0)
    other_inputs = other_inputs.fillna(value=0)
    inputs = {index: other_inputs.loc[index] for index in other_inputs.index.tolist()}
    inputs['disbursement'] = pd.to_numeric(inputs['disbursement'])
    # Payment method
    inputs['payment_method'] = str(file.parse(sheetname='payment_method', header=None)[0][0])
    # Read flow rates
    flow_rate = file.parse(sheetname='flow_rate').fillna(value=0)
    flow_rate = flow_rate.set_index(['from', 'to'])
    inputs['flow_rate'] = flow_rate * (-1)
    # Read paid-off rate
    inputs['paid_off_rate'] = file.parse(sheetname='paid_off_rate').fillna(value=0)
    # Read historical ENR
    inputs['enr_historical'] = file.parse(sheetname='enr_historical_'+book).fillna(value=0)
    # Read netflow rate
    inputs['netflow_rate'] = file.parse(sheetname='netflow_rate').fillna(value=0)
    # Read recovery rate
    inputs['recovery_rate'] = file.parse(sheetname='recovery_rate').fillna(value=0)
    # Read promo assumptions
    inputs['cii_rate_promo'] = file.parse(sheetname='cii_rate_promo').fillna(value=0)
    inputs['promo_projection'] = file.parse(sheetname='promo_projection').fillna(value=0)
    inputs['promo_proportion'] = file.parse(sheetname='promo_proportion').fillna(value=0)
    return inputs
    
def outputs_loan_by_book(file, book):

    '''Load inputs'''
    timeline = generate_timeline(file)
    inputs = inputs_loan(file, book)
    good_bank_buckets = ['B0', 'B1A', 'B1B', 'B2', 'B3']
    overdue_buckets = ['B1A', 'B1B', 'B2', 'B3']
    overdue_10days_buckets = ['B1B', 'B2', 'B3']
    all_buckets = good_bank_buckets + ['B4+', 'other', 'new_sale']

    '''Calculations'''

    # Create tables to fill in later
    enr = pd.DataFrame(np.zeros([len(all_buckets), len(timeline['months_extended'])]),
                      index=all_buckets, columns=timeline['months_extended'])
    enr[-1] = inputs['enr_historical'][-1]
    enr[-2] = inputs['enr_historical'][-2]

    ppmt_all_bkts = pd.DataFrame(np.zeros([len(all_buckets), len(timeline['months_extended'])]),
                      index=all_buckets, columns=timeline['months_extended'])

    prepayment = pd.Series(np.zeros(len(timeline['months_extended'])), index=timeline['months_extended'])

    ppmt_impact_prepayment = pd.Series(np.zeros(len(timeline['months_extended'])), index=timeline['months_extended'])

    ppmt_impact_disbursement = pd.Series(np.zeros(len(timeline['months_extended'])), index=timeline['months_extended'])

    ppmt_impact_outflow = pd.Series(np.zeros(len(timeline['months_extended'])), index=timeline['months_extended'])

    ppmt_impact_inflow = pd.Series(np.zeros(len(timeline['months_extended'])), index=timeline['months_extended'])

    bkt_flow_matrix = inputs['flow_rate'] * 0.0
    bkt_flow_matrix.sort_index(level=[0,1], inplace=True)

    bkt_to_n0 = pd.DataFrame(np.zeros([len(all_buckets), len(timeline['months_extended'])]),
                             index=all_buckets, columns=timeline['months_extended'])

    bkt_to_ovd = pd.DataFrame(np.zeros([len(all_buckets), len(timeline['months_extended'])]),
                      index=all_buckets, columns=timeline['months_extended'])

    # ---------BALANCE SHEET---------
    
    # Accumulative disbursement    
    accum_disbursement = inputs['disbursement'].cumsum(axis=0) 

    # New sales to bad bank
    newsale_to_bb = inputs['newsale_to_bb_rate'] * inputs['disbursement']
    
    ## New sales to buckets
    newsale_to_bkt = inputs['disbursement'] * inputs['flow_rate'].loc['new_sale']
    newsale_to_bkt.loc['B0'] = -(inputs['disbursement'] + newsale_to_bkt.loc['B1A':'B3'].sum() + newsale_to_bb)

    # New sales to B1B+
    newsale_to_ovd = newsale_to_bkt.loc['B1B':'B3'].sum()
    
    # New sales to B0, B1A
    newsale_to_n0 = inputs['disbursement'] + newsale_to_bb + newsale_to_ovd
    newsale_to_n0 = newsale_to_n0.fillna(value=0)

    # Loop through months
    for month in timeline['months']:

        ## Bucket flow
        bkt_flow_matrix[month] = inputs['flow_rate'][month] * pd.concat([enr[month-1]]*len(all_buckets), axis=0).sort_index().values            
        bkt_outflow = bkt_flow_matrix.groupby('from').sum()
        bkt_outflow.index = all_buckets

        bkt_inflow = bkt_flow_matrix.groupby('to').sum()
        bkt_inflow.index = all_buckets

        bkt_netflow = bkt_outflow - bkt_inflow

        ## Net payment from all buckets
        prepayment[month] = enr[month-1]['B0'] * inputs['early_termination_rate'][month]

        ppmt_impact_prepayment[month] = prepayment[month-1]/inputs['tenor'][month] + ppmt_impact_prepayment[month-1]

        if book == 'new':
            if inputs['payment_method'].lower() == 'eqp':
                ppmt_impact_disbursement[month] = (inputs['disbursement'][month-1]/inputs['tenor'][month] + 
                                                               ppmt_impact_disbursement[month-1])
            else:
                ppmt_impact_disbursement[month] = np.ppmt(rate=inputs['lifetime_interest'][month], 
                                                                      per=month/2, 
                                                                      nper=inputs['tenor'][month], 
                                                                      pv=accum_disbursement[month])

        ppmt_impact_outflow[month] = bkt_outflow[month]['B0']/inputs['tenor'][month] + ppmt_impact_outflow[month-1]

        ppmt_impact_inflow[month] = bkt_inflow.loc['B0'][month]/inputs['tenor'][month] + ppmt_impact_inflow[month-1]


        ppmt_impact_net = (-ppmt_impact_inflow + ppmt_impact_outflow
                       + ppmt_impact_disbursement - ppmt_impact_prepayment)

        ppmt_scheduled = inputs['system_payment_forecast'] + ppmt_impact_net

        ppmt_b0 = - ppmt_scheduled * (1 - inputs['unpaid_by_schedule_rate']) - prepayment

        ### ppmt_all_bkts
        ppmt_all_bkts[month] = inputs['paid_off_rate'][month] * enr[month-1][overdue_buckets]
        ppmt_all_bkts.loc['B0'] = ppmt_b0

        ## ENR
        enr[month] = enr[month-1] + bkt_outflow[month] - bkt_inflow[month] + ppmt_all_bkts[month] - newsale_to_bkt[month]

    # Bucket stay
    bkt_stay = enr.T.shift(1).T + ppmt_all_bkts + bkt_outflow

    # EOP balance    
    eop = enr.loc[good_bank_buckets].sum(axis=0)

    # ADB N0
    bkt_to_n0.loc['B0'] = bkt_stay.loc['B0'] - bkt_flow_matrix.loc['B0','B1A']
    bkt_to_n0.loc['B1A'] = bkt_stay.loc['B1A'] - bkt_flow_matrix.loc['B1A','B0']
    for bucket in overdue_10days_buckets:
        bkt_to_n0.loc[bucket] = -bkt_flow_matrix.loc[bucket,'B0'] - bkt_flow_matrix.loc[bucket,'B1A']

    adb_n0_n0 = bkt_to_n0.loc['B0':'B1A'].sum() - ppmt_all_bkts.loc['B0':'B1A'].sum()/2 

    # ADB to overdue
    for bucket in all_buckets:
        bkt_to_ovd.loc[bucket] = - bkt_flow_matrix.loc[bucket].loc['B1B':'B3'].sum()
    for bucket in overdue_10days_buckets:
        bkt_to_ovd.loc[bucket] += bkt_stay.loc[bucket]

    adb_n0_ovd = bkt_to_ovd.loc['B0':'B1A'].sum()

    # ADB overdue to paid
    adb_ovd_paid = bkt_to_ovd.loc[overdue_10days_buckets].sum() - ppmt_all_bkts.loc[overdue_10days_buckets].sum()/2
    adb_ovd_paid_upflow = (bkt_stay.loc['B1B':'B3'].sum() - 
                           bkt_flow_matrix.loc['B3'].loc['B1B':'B2'].sum() -
                           bkt_flow_matrix.loc['B2','B1B'] -
                           ppmt_all_bkts.loc['B1B':'B3'].sum())

    # ADB overdue to NO
    adb_ovd_n0 = bkt_to_n0.loc['B1B':'B3'].sum()

    #  ADB to bad bank
    enr_to_bb = -bkt_inflow.loc['B4+'] - newsale_to_bb
    enr_to_bb[-1] = inputs['enr_to_bb_historical'][-1]

    adb_to_bb = enr_to_bb/2

    # ADB new sale
    adb_newsale = inputs['disbursement']/2 + newsale_to_bb/4
    
    ## Total ADB
    adb = adb_n0_n0 + adb_n0_ovd + adb_ovd_paid + adb_ovd_n0 + adb_to_bb + adb_newsale
    
    # ---------TOI---------
    
    # CII
    
    ## CII from B0, B1A ENR 
    if book == 'new':
        cii_rate_increase = inputs['cii_rate_promo'] * 0.0

        list_of_promo = inputs['promo_projection'].index.tolist()

        disbursement_by_promo = inputs['promo_proportion'] * inputs['disbursement']

        for promo in list_of_promo:
            for month in timeline['months']:
                if month < inputs['promo_projection']['promo_time'][promo] or month == 0:
                    cii_rate_increase[month][promo] = 0
                else:
                    origination_month = month-inputs['promo_projection']['promo_time'][promo]
                    cii_rate_increase[month][promo] = ((disbursement_by_promo[origination_month][promo]
                                                       / accum_disbursement[month-1]) * 
                                                       (inputs['promo_projection']['postpromo'][promo] - inputs['cii_rate_promo'][month][promo])) 


        cii_rate_newsale = (inputs['cii_rate_promo'] * disbursement_by_promo).sum()/inputs['disbursement']
        cii_rate_newsale = cii_rate_newsale.fillna(value=0)

        cii_rate_n0_n0 = pd.Series(np.zeros(len(timeline['months_extended'])), index=timeline['months_extended'])
        cii_rate_n0_n0[0] = 0
        for month in range(1,len(timeline['months'])):
            cii_rate_n0_n0[month] = ((bkt_to_n0[month-1]['B0':'B3'].sum() *
                                     (cii_rate_n0_n0[month-1] + cii_rate_increase[month].sum()) + 
                                     newsale_to_n0[month-1]*cii_rate_newsale[month-1]) /
                                     enr[month-1]['B0':'B1A'].sum()) 

        cii_n0_n0 = adb_n0_n0 * cii_rate_n0_n0 / 360 * timeline['days_in_month']
    
    else:
        cii_rate_n0_n0 = inputs['cii_rate_n0_n0_oldbook']
        cii_n0_n0 = adb_n0_n0 * inputs['cii_rate_n0_n0_oldbook']/360 * timeline['days_in_month']

    ## CII from ENR moving to B1B+
    cii_n0_ovd = adb_n0_ovd * inputs['cii_rate_n0_ovd']/360 * timeline['days_in_month']
    
    ## CII from new disbursement
    if book == 'new':
        cii_newsale = adb_newsale * cii_rate_newsale/360 * timeline['days_in_month']
    else:
        cii_newsale = 0

    ## CII from paid ENR in B1B+
    cii_ovd_paid = adb_ovd_paid_upflow * inputs['cii_rate_ovd_paid_adjust'] * cii_rate_n0_n0/360 * timeline['days_in_month']

    ## CII from ENR flowing upwards from B1B+
    cii_ovd_n0 = adb_ovd_n0 * inputs['cii_rate_ovd_n0_adjust'] * cii_rate_n0_n0/360 * timeline['days_in_month']

    ## CII from ENR flowing to bad bank
    cii_ovd_bb = adb_to_bb * inputs['cii_rate_ovd_bb']/360 * timeline['days_in_month']

    ## Total CII
    cii = cii_n0_n0 + cii_n0_ovd + cii_newsale + cii_ovd_paid + cii_ovd_n0 + cii_ovd_bb
    
    # FTP
    
    ## FTP for B0, B1A ENR
    if book == 'new':
        ftp_rate_increase = cii_rate_increase * 0.0

        for promo in list_of_promo:
            for month in timeline['months']:
                if month < inputs['promo_projection']['promo_time'][promo] or month == 0:
                    ftp_rate_increase[month][promo] = 0
                else:
                    origination_month = month-inputs['promo_projection']['promo_time'][promo]
                    ftp_rate_increase[month][promo] = ((disbursement_by_promo[origination_month][promo]
                                                       / accum_disbursement[month-1]) * 
                                                       (-inputs['promo_projection']['ftp_rate_postpromo'][promo] + inputs['promo_projection']['ftp_rate_promo'][promo])) 

        ftp_rate_newsale = -disbursement_by_promo.multiply(inputs['promo_projection']['ftp_rate_promo'], axis='index').sum()/inputs['disbursement']
        ftp_rate_newsale = ftp_rate_newsale.fillna(value=0)

        ftp_rate_n0_n0 = pd.Series(np.zeros(len(timeline['months_extended'])), index=timeline['months_extended'])
        ftp_rate_n0_n0[0] = 0

        for month in range(1,len(timeline['months'])):
            ftp_rate_n0_n0[month] = ((bkt_to_n0[month-1]['B0':'B3'].sum() *
                                     (ftp_rate_n0_n0[month-1] + ftp_rate_increase[month].sum()) + 
                                     newsale_to_n0[month-1]*ftp_rate_newsale[month-1]) /
                                     enr[month-1]['B0':'B1A'].sum()) 

        ftp_n0_n0 = adb_n0_n0 * ftp_rate_n0_n0 / 360 * timeline['days_in_month']
        
    else:
        ftp_n0_n0 = adb_n0_n0 * inputs['ftp_rate_n0_n0_oldbook']/360 * timeline['days_in_month']

    ## FTP for ENR moving to B1B+
    ftp_n0_ovd = adb_n0_ovd * inputs['ftp_rate_n0_ovd']/360 * timeline['days_in_month']

    ## FTP for new disbursement
    if book == 'new':
        ftp_newsale = adb_newsale * inputs['ftp_newsale_adjust'] * ftp_rate_newsale/360 * timeline['days_in_month']
    else:
        ftp_newsale = 0

    ## FTP for paid ENR in B1B+
    ftp_ovd_paid = adb_ovd_paid * inputs['ftp_rate_ovd_paid']/360 * timeline['days_in_month']

    ## FTP for ENR flowing upwards from B1B+
    ftp_ovd_n0 = adb_ovd_n0 * inputs['ftp_rate_ovd_n0']/360 * timeline['days_in_month']

    ## FTP for ENR flowing to bad bank
    ftp_ovd_bb = adb_to_bb * inputs['ftp_rate_ovd_bb']/360 * timeline['days_in_month']

    ## Total FTP
    ftp = ftp_n0_n0 + ftp_n0_ovd + ftp_newsale + ftp_ovd_paid + ftp_ovd_n0 + ftp_ovd_bb
    
    # Other NII
    
    ## NII from prepayment fee
    nii_prepayment_fee = inputs['prepayment_fee_rate'] * prepayment

    ## NII impact from commissions
    nii_commission = inputs['disbursement'] * inputs['nii_commission_rate']

    ## Total other NII
    nii_other = nii_prepayment_fee + nii_commission
    
    ## Total NII
    nii = cii + ftp + nii_other
    
    ## NFI
    nfi = (1/3*inputs['disbursement'].shift(1) + 2/3*inputs['disbursement']) * inputs['nfi_to_disbursement']
    '''TBD'''
    
    ## TOI
    toi = nii + nfi
    
    # ---------PROVISION---------

    ## Average net flow rate
    avg_netflow = inputs['netflow_rate'].rolling(window=12, min_periods=1, axis=1).mean()

    ## Probability of default
    default_prob = avg_netflow.iloc[::-1].iloc[-13:]
    default_prob = default_prob.rolling(window=13, min_periods=1, axis=0).apply(np.prod)
    default_prob = default_prob.iloc[::-1].loc[:,-5:]

    ## Provision rate
    loss_rate = 1- inputs['recovery_rate']
    provision_rate = default_prob * loss_rate.product(axis=0)
    provision_rate_GB = provision_rate.loc[good_bank_buckets]
    provision_rate_GB.loc['B1A'] = provision_rate.loc['B1']
    provision_rate_GB.loc['B1B'] = provision_rate.loc['B1']

    ## Provision in good bank
    provision_GB = enr.loc[good_bank_buckets].T.shift(1).T * provision_rate_GB.T.shift(1).T

    provision_expense_GB = provision_GB - provision_GB.shift(1,axis=1)  

    ## Provision to bad bank
    provision_expense_bb = enr_to_bb.shift(1) * provision_rate.loc['B6'].shift(1)

    ## Total provision
    provision = provision_expense_GB.sum(axis=0) + provision_expense_bb

    return {'eop': eop,
            'provision': provision,
            'adb': adb,
            'nii': nii,
            'nfi': nfi,
            'toi': toi,
            'disbursement': inputs['disbursement']}

def outputs_loan(file):
    outputs_new = outputs_loan_by_book(file, 'new')
    outputs_old = outputs_loan_by_book(file,'old')
    timeline = generate_timeline(file)
    outputs = {'month': timeline['months_names']}
    indicator_list = ['eop', 'adb', 'toi', 'nii', 'nfi', 'provision', 'disbursement']
    for indicator in indicator_list:
        outputs[indicator] = outputs_new[indicator] + outputs_old[indicator]                                                         
    return outputs


# ## Overdraft function

# In[344]:


def inputs_od(file):
    # Read other inputs
    other_inputs = file.parse(sheetname='other_inputs').fillna(value=0)
    other_inputs = other_inputs.fillna(value=0)
    inputs = {index: other_inputs.loc[index] for index in other_inputs.index.tolist()}
    inputs['disbursement'] = pd.to_numeric(inputs['disbursement'])
    # Read flow rates
    flow_rate = file.parse(sheetname='flow_rate').fillna(value=0)
    flow_rate = flow_rate.set_index(['from', 'to'])
    inputs['flow_rate'] = flow_rate
    # Read paid-off rate
    inputs['paid_off_rate'] = file.parse(sheetname='paid_off_rate').fillna(value=0)
    # Read historical ENR
    inputs['enr_historical'] = file.parse(sheetname='enr_historical').fillna(value=0)
    # Read netflow rate
    inputs['netflow_rate'] = file.parse(sheetname='netflow_rate').fillna(value=0)
    # Read recovery rate
    inputs['recovery_rate'] = file.parse(sheetname='recovery_rate').fillna(value=0)
    # Read increase in month rate
    inputs['increase_in_month_rate'] = file.parse(sheetname='increase_in_month_rate').fillna(value=0)
    return inputs
    
def outputs_od(file):

    '''Load inputs'''
    timeline = generate_timeline(file)
    inputs = inputs_od(file)
    good_bank_buckets = ['B0', 'B1A', 'B1B', 'B2', 'B3']
    overdue_buckets = ['B1A', 'B1B', 'B2', 'B3']
    overdue_10days_buckets = ['B1B', 'B2', 'B3']
    all_buckets = good_bank_buckets + ['B4+', 'other', 'new_sale']

    '''Calculations'''

    # Create tables to fill in later
    enr = pd.DataFrame(np.zeros([len(all_buckets), len(timeline['months_extended'])]),
                      index=all_buckets, columns=timeline['months_extended'])
    enr[-1] = inputs['enr_historical'][-1]
    enr[-2] = inputs['enr_historical'][-2]

    ppmt_all_bkts = pd.DataFrame(np.zeros([len(all_buckets), len(timeline['months_extended'])]),
                      index=all_buckets, columns=timeline['months_extended'])

    bkt_flow_matrix = inputs['flow_rate'] * 0.0
    bkt_flow_matrix.sort_index(level=[0,1], inplace=True)

    bkt_to_n0 = pd.DataFrame(np.zeros([len(all_buckets), len(timeline['months_extended'])]),
                             index=all_buckets, columns=timeline['months_extended'])

    bkt_to_ovd = pd.DataFrame(np.zeros([len(all_buckets), len(timeline['months_extended'])]),
                             index=all_buckets, columns=timeline['months_extended'])
    
    bkt_increase_in_month = pd.DataFrame(np.zeros([len(all_buckets), len(timeline['months_extended'])]),
                             index=all_buckets, columns=timeline['months_extended'])

    # ---------BALANCE SHEET---------
    
    # New sales principal payment
    ppmt_newsale = inputs['disbursement'] * inputs['newsale_rate_closed_in_month']
    
    ## New sales to buckets
    newsale_to_bkt = inputs['disbursement'] * inputs['flow_rate'].loc['new_sale']
    newsale_to_bkt.loc['B0'] = -(inputs['disbursement'] + newsale_to_bkt.loc['B1A':'other'].sum() + inputs['disbursement'] * inputs['paid_off_rate'].loc['new_sale'] + ppmt_newsale)

    # New sales to bad bank
    newsale_to_bb = inputs['newsale_to_bb_rate'] * inputs['disbursement']
    
    # Loop through months
    for month in timeline['months']:

        ## Bucket flow
        bkt_flow_matrix[month] = inputs['flow_rate'][month] * pd.concat([enr[month-1]]*len(all_buckets), axis=0).sort_index().values            
        bkt_outflow = bkt_flow_matrix.groupby('from').sum()
        bkt_outflow.index = all_buckets

        bkt_inflow = bkt_flow_matrix.groupby('to').sum()
        bkt_inflow.index = all_buckets

        bkt_netflow = bkt_outflow - bkt_inflow

        ### ppmt_all_bkts
        ppmt_all_bkts[month] = inputs['paid_off_rate'][month] * enr[month-1]
        
        ### Bucket increase in month
        bkt_increase_in_month[month] = inputs['increase_in_month_rate'][month] * enr[month-1]
        bkt_increase_in_month[month]['B0'] = inputs['disbursement'][month]
        
        ## ENR
        enr[month] = enr[month-1] + bkt_outflow[month] - bkt_inflow[month] + ppmt_all_bkts[month] + newsale_to_bkt[month] + bkt_increase_in_month[month]

    # EOP balance    
    eop = enr.loc[good_bank_buckets].sum(axis=0)
    
    # ADB
    adb = (eop.shift(1) + eop)/(inputs['adb_factor'].fillna(2))
    
    # ENR to bad bank
    enr_to_bb = -bkt_inflow.loc['B4+'] - newsale_to_bb
    enr_to_bb[-1] = inputs['enr_to_bb_historical'][-1]
    
    # ---------TOI--------------
    ## NII
    nii = adb * (inputs['ftp_rate'] + inputs['cii_rate'])/360 * timeline['days_in_month'] + inputs['other_nii']
    
    ## NFI
    nfi = inputs['account_number'] * inputs['fee_per_account']
    
    ## TOI
    toi = nii + nfi
    
    # ---------PROVISION---------

    ## Average net flow rate
    avg_netflow = inputs['netflow_rate'].rolling(window=12, min_periods=1, axis=1).mean()

    ## Probability of default
    default_prob = avg_netflow.iloc[::-1].iloc[-13:]
    default_prob = default_prob.rolling(window=13, min_periods=1, axis=0).apply(np.prod)
    default_prob = default_prob.iloc[::-1].loc[:,-5:]

    ## Provision rate
    loss_rate = 1- inputs['recovery_rate']
    provision_rate = default_prob * loss_rate.product(axis=0)
    provision_rate_GB = provision_rate.loc[good_bank_buckets]
    provision_rate_GB.loc['B1A'] = provision_rate.loc['B1']
    provision_rate_GB.loc['B1B'] = provision_rate.loc['B1']

    ## Provision in good bank
    provision_GB = enr.loc[good_bank_buckets].T.shift(1).T * provision_rate_GB.T.shift(1).T

    provision_expense_GB = provision_GB - provision_GB.shift(1,axis=1)  

    ## Provision to bad bank
    provision_expense_bb = enr_to_bb.shift(1) * provision_rate.loc['B6'].shift(1)

    ## Total provision
    provision = provision_expense_GB.sum(axis=0) + provision_expense_bb
    
    return {'month': timeline['months_names'],
            'eop': eop,
            'provision': provision,
            'adb': adb,
            'nii': nii,
            'nfi': nfi,
            'toi': toi}


# ## Credit Card functions

# In[348]:


def inputs_cc(file):
    # Read other inputs
    other_inputs = file.parse(sheetname='other_inputs')
    other_inputs = other_inputs.fillna(value=0)
    inputs = {index: other_inputs.loc[index] for index in other_inputs.index.tolist()}
    # Read ENR historical
    inputs['enr_historical'] = file.parse(sheetname='enr_historical').fillna(value=0)
    # Read netflow
    inputs['netflow_rate'] = file.parse(sheetname='netflow_rate').fillna(value=0)
    # Read recovery rate
    inputs['recovery_rate'] = file.parse(sheetname='recovery_rate').fillna(value=0)    
    return inputs

def outputs_cc(file):
    '''Load inputs'''
    timeline = generate_timeline(file)
    inputs = inputs_cc(file)
    overdue_buckets = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B24', 'B36']
    all_buckets = ['B0'] + overdue_buckets
    good_bank_buckets = ['B0', 'B1', 'B2', 'B3']
    
    '''Calculations'''
    
    # Create tables to fill in later
    total_issued = pd.Series(np.zeros(len(timeline['months_extended'])), index=timeline['months_extended']) 
    total_issued[-1] = inputs['total_issued_historical'][-1]
    
    enr = pd.DataFrame(np.zeros([len(all_buckets), len(timeline['months_extended'])]),
                      index=all_buckets, columns=timeline['months_extended'])
    enr[-1] = inputs['enr_historical'][-1]*1.0
        
    # ---------BALANCE SHEET---------  
    
    # Total (accumulative) issued cards
    for month in timeline['months']:
        total_issued[month] = total_issued[month-1]*(1-inputs['attrition_rate'][month]) + inputs['monthly_issued'][month]
    
    # Total (accumulative) activated cards
    total_activated = total_issued * inputs['activation_rate']
    
    # Spend
    retail_spend = total_activated * inputs['retail_spend_per_activated']
    onl_spend = total_activated * inputs['onl_spend_per_activated']
    cash_spend = total_activated * inputs['cash_spend_per_activated']
    total_spend = retail_spend + onl_spend + cash_spend
    
    ## Average net flow rate
    avg_netflow = inputs['netflow_rate'].rolling(window=12, min_periods=1, axis=1).mean()
     
    # ENR
    for month in timeline['months']:
        enr[month]['B0'] = (enr[month-1]['B0'] + total_spend[month]) * inputs['revolving_rate'][month]
        for i in range(1,len(all_buckets)):
             enr[month][all_buckets[i]] = enr[month-1][all_buckets[i-1]] * avg_netflow[month][all_buckets[i-1]]
    
    # EOP
    eop = enr.loc['B0':'B3'].sum()
    
    # ---------TOI---------  
    # ANR
    adb = (eop.shift(1) * inputs['billing_day'] + eop *(timeline['days_in_month'] - inputs['billing_day'])) / timeline['days_in_month']
    
    # CII
    cii_normal = (adb * inputs['cii_rate']/360 * timeline['days_in_month'] +
                  (enr['B1':'B3'].shift(1).sum() * inputs['billing_day'] + 
                   enr['B1':'B3'].sum() * (timeline['days_in_month'] - inputs['billing_day'])) / 
                  timeline['days_in_month'] *
                  inputs['cii_rate']/2/360 * timeline['days_in_month'] +
                  cash_spend * (1 - inputs['revolving_rate']) * inputs['cii_rate']/360 * timeline['days_in_month'])
    
    cii_adjusted = - cii_normal*(1-inputs['revolving_rate'])
    
    cii = cii_normal + cii_adjusted
    
    # FTP
    ftp = adb * inputs['ftp_rate']/360 * timeline['days_in_month']
    
    # NII
    nii = cii - ftp
    
    # NFI
    ## Gross fee income
    interchange_fee = (retail_spend + onl_spend) * inputs['interchange_rate']
    annual_fee = interchange_fee * inputs['annual_fee_to_gross'] / inputs['interchange_to_gross']
    late_payment_fee = interchange_fee * inputs['late_payment_to_gross'] / inputs['interchange_to_gross']
    fx_fee = interchange_fee * inputs['fx_fee_to_gross'] / inputs['interchange_to_gross']
    cash_advance_fee = interchange_fee * inputs['cash_advance_to_gross'] / inputs['interchange_to_gross']
    installment_fee = interchange_fee * inputs['installment_to_gross'] / inputs['interchange_to_gross']
    misc = interchange_fee * inputs['misc_to_gross'] / inputs['interchange_to_gross']
    gross_fee = interchange_fee + annual_fee + late_payment_fee + fx_fee + cash_advance_fee + installment_fee + misc
    ## Fee expense
    mc_expense = -interchange_fee * inputs['mc_expense_to_interchange']
    mc_guarantee = mc_expense / inputs['mc_expense_rate'] * inputs['mc_guarantee_rate']
    sms_expense = mc_expense / inputs['mc_expense_rate'] * inputs['sms_banking_rate']
    #customer_reward_expense = - gross_fee
    
    ## Net fee income
    '''To be changed'''
    nfi = interchange_fee *100/59
    
    # TOI
    toi = nii + nfi
    
    #---------PROVISION---------

    ## Probability of default
    default_prob = avg_netflow.iloc[::-1].iloc[-13:]
    default_prob = default_prob.rolling(window=13, min_periods=1, axis=0).apply(np.prod)
    default_prob = default_prob.iloc[::-1].loc[:,-5:]

    ## Provision rate
    loss_rate = 1- inputs['recovery_rate']
    provision_rate = default_prob * loss_rate.product(axis=0)
    provision_rate_GB = provision_rate.loc[good_bank_buckets]

    ## Provision in good bank
    provision_GB = enr.loc[good_bank_buckets].T.shift(1).T * provision_rate_GB.T.shift(1).T

    provision_expense_GB = provision_GB - provision_GB.shift(1,axis=1)  

    ## Provision to bad bank
    provision_expense_bb = enr.loc['B6'].shift(1) * provision_rate.loc['B6'].shift(1)

    ## Total provision
    provision = provision_expense_GB.sum(axis=0) + provision_expense_bb
    
    return {'month': timeline['months_names'],
            'total_activated': total_activated,
            'total_spend': total_spend,
            'eop': eop,
            'adb': adb,
            'nii': nii,
            'nfi': nfi,
            'toi': toi,
            'monthly_issued': inputs['monthly_issued'],
            'provision': provision}


# # TD & CASA

# In[349]:


def inputs_deposit(file):
    other_inputs = file.parse(sheetname='other_inputs')
    other_inputs = other_inputs.fillna(value=0)
    inputs = {index: other_inputs.loc[index] for index in other_inputs.index.tolist()}
    
    inputs['composition'] = file.parse(sheetname='composition').fillna(value=0)
    inputs['cii_rate'] = file.parse(sheetname='cii_rate').fillna(value=0)
    inputs['ftp_rate'] = file.parse(sheetname='ftp_rate').fillna(value=0)
    
    return inputs

def outputs_deposit(file):
    '''Load inputs'''
    timeline = generate_timeline(file)
    inputs = inputs_deposit(file)
    
    '''Calculations'''
    # Balance
    eop_by_type = inputs['eop'] * inputs['composition'] 
    adb_by_type = (eop_by_type.shift(1, axis=1) + eop_by_type)/2
    adb = (inputs['eop'].shift(1) + inputs['eop'])/2
    
    # NII
    cii_by_type = adb_by_type * inputs['cii_rate']/12
    ftp_by_type = adb_by_type * inputs['ftp_rate']/360 * timeline['days_in_month'] 
    nii_bond = inputs['bond_par_value'] * inputs['bond_nii_rate'] * inputs['bond_maturity'] / 360
    nii_by_type = ftp_by_type - cii_by_type
    nii = nii_by_type.sum() + nii_bond
    
    # NFI
    debit_card_fee = inputs['debit_cards_num'] * inputs['fee_per_card']
    other_fees = inputs['other_fees_per_active'] * inputs['active_customers']
    remittance = inputs['remittance_historical'] * (inputs['remittance_growth_rate']+1).cumprod()
    nfi = debit_card_fee + other_fees + remittance + inputs['fe_credit']
    
    # TOI
    toi = nfi + nii
    
    
    return {'month': timeline['months_names'],
            'eop': inputs['eop'],
            'adb': adb,
            'nii': nii,
            'nfi': nfi,
            'toi': toi}
    


# In[350]:


def inputs_investment(file):
    inputs = {}
    inputs['to_nii'] = file.parse(sheetname='to_nii')
    inputs['to_nfi'] = file.parse(sheetname='to_nfi')
    return inputs
    
def outputs_investment(file):    
    '''Load inputs'''
    timeline = generate_timeline(file)
    inputs = inputs_investment(file)
    
    '''Calculations'''
    nfi = (inputs['to_nfi'].loc['sales'].shift(1)*1/3 + inputs['to_nfi'].loc['sales']*2/3) * inputs['to_nfi'].loc['rate'] * inputs['to_nfi'].loc['maturity']/360
    nii = (inputs['to_nii'].loc['sales'].shift(1)*1/3 + inputs['to_nii'].loc['sales']*2/3) * inputs['to_nii'].loc['rate'] * inputs['to_nii'].loc['maturity']/360
    toi = nfi + nii
    
    return {'month': timeline['months_names'],
            'nii': nii,
            'nfi': nfi,
            'toi': toi}
    


# ## Run

# In[363]:


def get_files_and_paths(folder):
    files = []
    paths = []
    for root, directories, filenames in os.walk(folder):
        for filename in filenames:
            if filename.endswith('.xlsx') and not filename.startswith('~$'):
                    path = os.path.join(root, filename)
                    paths.append(path)
                    files.append(pd.ExcelFile(path))
    return files, paths

def transform(string):
    return ''.join(string.lower().split())

def read_type(file):
    return file.parse(sheetname='classify').loc[0].apply(transform)

model_dict = {'loan': outputs_loan, 'overdraft': outputs_od, 'creditcard': outputs_cc, 
              'deposit': outputs_deposit,'investment': outputs_investment}

def add_dfs(list_of_dfs):
    # Add data frames in a list
    sum_df = list_of_dfs[0]
    if len(list_of_dfs) > 1:
        for df in list_of_dfs[1:]:
            sum_df = sum_df.add(df, fill_value=0).fillna(0)         
    # Add provision to TOI and NIM actual
    if 'provision' in sum_df.columns:
        sum_df = add_nim_actual(add_provision_to_toi(sum_df))        
    return sum_df

def aggregate(outputs_map):
    # will be used to aggregate outputs by product classes, products, and sub_products
    for key, output_list in outputs_map.items():
        outputs_map[key] = add_dfs(output_list)
    return outputs_map   

def all_outputs(folder, year=None):
    files = get_files_and_paths(folder)[0]
    # Run model on sub products
    outputs = list()
    index = 0
    class_to_out = dict()
    prod_to_out = dict()
    subprod_to_out = dict()
    for file in files:
        # Read meta information
        product_class, product, subproduct = read_type(file)[0], read_type(file)[1], read_type(file)[2]
        # Calculate outputs
        output = model_dict[product_class](file)
        # Change index to timestamp, drop historical months
        output = pd.DataFrame.from_dict(output, orient='columns', dtype=None).loc[0:]
        output.index = pd.to_datetime(output['month'])
        output = output.drop('month', axis=1)
        # Filter by year
        if year is not None:
            output = output.loc[output.index.year == year]
        outputs.append(output)
        # Map outputs to product class, product, and subproduct
        class_to_out[product_class] = class_to_out.get(product_class, [])
        class_to_out[product_class].append(output)
        prod_to_out[product] = prod_to_out.get(product, [])
        prod_to_out[product].append(output)
        subprod_to_out[subproduct] = subprod_to_out.get(subproduct, [])
        subprod_to_out[subproduct].append(output)
    # Aggregate outputs by product classes, products, sub_products    
    class_to_out = aggregate(class_to_out)
    prod_to_out = aggregate(prod_to_out)
    subprod_to_out_agg = aggregate(subprod_to_out)
    total = add_dfs(list(class_to_out.values()))
    return total, class_to_out, prod_to_out, subprod_to_out

def visualize(df, size):
    print('### Summary ###')
    if 'eop' in df.columns:
        print('EOP balance, last month: {0:.0f}'.format(df['eop'].iloc[-1]))
        df.plot(y=['eop', 'adb'], figsize=size, ylim = (0,None))
        plt.show()
    print('TOI - total: {0:.1f}'.format(df['toi'].sum()))
    print('TOI - monthly_average: {0:.1f}'.format(df['toi'].mean()))
    print('NII - total: {0:.1f}'.format(df['nii'].sum()))
    print('NII - monthly_average: {0:.1f}'.format(df['nii'].mean()))
    print('NFI - total: {0:.1f}'.format(df['nfi'].sum()))
    print('NFI - monthly_average: {0:.1f}'.format(df['nfi'].mean()))
    if 'provision' in df.columns:
        print('Provision - total: {0:.0f}'.format(df['provision'].sum()))
        print('Provision - monthly average: {0:.0f}'.format(df['provision'].mean()))
        print('Provision as % of TOI: {0:.1f}%'.format(df['provision'].sum() / df['toi'].sum() * 100))
        df.plot(y=['nii', 'nfi','toi', 'provision'], figsize = size, ylim = (0,None))
        plt.show()
    else: 
        df.plot(y=['nii', 'nfi','toi'], figsize=size, ylim = (0,None))
        plt.show()
    if 'disbursement' in df.columns:
        print('Disbursement - total: {0:0f}'.format(df['disbursement'].sum()))
        print('Disbursement - monthly average: {0:0f}'.format(df['disbursement'].mean()))
        df.plot(y=['disbursement'], figsize=size, ylim = (0,None))
        plt.show()
    if 'monthly_issued' in df.columns:
        print('Card spend - total: {0:.0f}'.format(df['total_spend'].sum()))
        print('Card spend - monthly average: {0:.0f}'.format(df['total_spend'].mean()))
        df.plot(y=['total_spend'], figsize=size, ylim = (0,None))
        plt.show()
        print('Cards issuance - total: {0:.0f}'.format(df['monthly_issued'].sum()))
        print('Cards issuance - monthly average: {0:.0f}'.format(df['monthly_issued'].mean()))
        df.plot(y=['monthly_issued'], figsize=size, ylim = (0,None))
        plt.show()
    print()
    print('### Full Table ###')
    display(df)
    

def run_model(folder, size, year=None):
    total, class_to_out, prod_to_out, subprod_to_out = all_outputs(folder, year=year)
    print('------------------------------------------------------------------------------------------------------')
    print('ALL PRODUCTS')
    print()
    visualize(total, size)
    print()
    print('------------------------------------------------------------------------------------------------------')
    print('BY PRODUCT CLASS')
    print()
    for key, value in class_to_out.items():
        print('___{}________________________________________'.format(key.upper()))
        visualize(value, size)
        print()
    print()
    print('------------------------------------------------------------------------------------------------------')
    print('BY PRODUCT')
    print()
    for key, value in prod_to_out.items():
        print('___{}________________________________________'.format(key.upper()))
        visualize(value, size)
        print()
    print()
    print('------------------------------------------------------------------------------------------------------')
    print('BY SUBPRODUCT')
    for key, value in subprod_to_out.items():
        print('___{}________________________________________'.format(key.upper()))
        visualize(value, size) 
        print()
    return None
    

