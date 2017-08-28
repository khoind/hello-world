
# coding: utf-8

# # Set up

# In[1]:


from __future__ import print_function
import numpy as np
import pandas as pd
import calendar
import matplotlib.pyplot as plt
import os
from IPython.display import Markdown
from IPython.display import display as dp
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from pixiedust.display import *


# In[2]:


def printmd(string):
    '''
    Print strings formatted in markdown
    '''
    dp(Markdown(string))


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
        year.append(start_year)
        start_month += 1
    return month_names, days_in_month, year        


# In[4]:


def adb(eop, eop_month_minus_1):
    '''
    Calculate month average balance from month end of period balance
    :Parameters:
    eop: Series, end of period balance of each month
    eop_month_minus_1: number, end of period balance of latest historical month
    '''
    eop_previous_month = eop.shift(1) # EOP balance of the previous month
    eop_previous_month[0] = eop_month_minus_1 # EOP balance of month zero
    return (eop+eop_previous_month)/2


# In[5]:


def add_nim_actual(table):
    '''
    Add actual NIM to the output table
    :params:
    table: dataframe containing NII and ADB
    '''
    nim_actual = table['nii']/table['total_balance_adb']
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


# # Product classes

# In[6]:


class Product():
    '''
    Class of all products that generate interest income
    '''

    def get_start_month(self):
        return self.file.parse(sheetname='get_start_month', header=None).loc[1]
    
    def start_month(self):
        return self.get_start_month()[0]
    
    def start_year(self):
        return self.get_start_month()[1]
    
    def num_months(self):
        return self.get_start_month()[2]
    
    def months(self):
        return list(range(self.num_months()))
    
    def months_extended(self):
        return [-5, -4, -3, -2, -1] + self.months()
    
    def months_names(self):
        names = forecast_window(self.start_month(), self.start_year(), self.num_months())[0]
        return pd.Series(names)
    
    def days_in_month(self):
        return forecast_window(self.start_month(), self.start_year(), self.num_months())[1]
    
    def year(self):
        return forecast_window(self.start_month(), self.start_year(), self.num_months())[2]
        
    def cii_rate(self):    
        # Interest rate
        return self.file.parse(sheetname='cii_rate', header=None).loc[1]


# In[7]:


class Lending(Product):
    '''
    Class of all lendind products and sub-products including loans, credit cards, and overdrafts
    '''       
    def cof_rate(self):    
        # cost of fund rate
        return self.file.parse(sheetname='cof_rate', header=None).loc[1]
   
    def flow_rate(self):
        # Flow rate
        flow_rate = self.file.parse(sheetname='flow_rate')
        flow_rate.index = self.months()
        return flow_rate
        
    def net_flow(self):
        # Forecast net flow rate
        net_flow = self.file.parse(sheetname='net_flow')
        net_flow.index = self.months()
        net_flow.columns = list(range(14))
        # Historical net flow rate
        net_flow_pre_month_0 = self.file.parse(sheetname='net_flow_pre_month_0')
        net_flow_pre_month_0.index = [-5, -4, -3, -2, -1]
        net_flow_pre_month_0.columns = list(range(14))
        # Join historical net flow and forecast net flow
        net_flow = pd.concat([net_flow_pre_month_0, net_flow])
        return net_flow
        
    def enr_pre_month_0(self):    
        # Historical ENR
        enr_pre_month_0 = self.file.parse(sheetname='enr_pre_month_0')
        enr_pre_month_0.index = [-5, -4, -3, -2, -1]
        enr_pre_month_0.columns = list(range(14))
        return enr_pre_month_0
        
    def provision_for_fraud_rate(self):    
        # Provision for fraud
        return self.file.parse(sheetname='provision_for_fraud_rate', header=None).loc[1]
        
    def recovery_rate(self):    
        # Recovery rate & loss_rate
        recovery_rate = self.file.parse(sheetname='recovery_rate')
        recovery_rate.index = self.months()
        recovery_rate.columns = ['B13-23', 'B24-36', 'B36+']
        return recovery_rate
    
    def loss_rate(self):
        return 1-self.recovery_rate()
        
    def provision_rate_pre_month_0(self):
        # Historical provision rate
        provision_rate_pre_month_0 = self.file.parse(sheetname='provision_rate_pre_month_0')
        provision_rate_pre_month_0.index = [-5, -4, -3, -2, -1]
        provision_rate_pre_month_0.columns = list(range(14))
        return provision_rate_pre_month_0
    
    def enr_by_bucket(self):
        return  self.enr_by_bucket_extended().loc[0:,:]
    
    def total_balance_eop(self):
        # EOP balance, good bank only  
        return self.enr_by_bucket().loc[:,0:3].sum(axis=1)
    
    def total_balance_eop_month_minus_1(self):
        # Total good bank balance at the end of latest historical month
        return self.enr_pre_month_0().loc[-1,0:3].sum()
    
    def total_balance_adb(self):
        # Average balance, good bank
        return adb(self.total_balance_eop(), self.total_balance_eop_month_minus_1())
    
    def cof(self):
        # Cost of fund
        return self.total_balance_adb()*self.cof_rate()/365*self.days_in_month()
    
    def nii(self):
        # Net interest income
        return self.cii() - self.cof()
    
    def toi(self):
        # TOI
        return self.nii() + self.nfi()  
    
    def default_prob(self):
        net_flow = self.net_flow()
        loss_rate = self.loss_rate()
        provision_rate_pre_month_0 = self.provision_rate_pre_month_0()
        ## Probability of Default
        # Create an empty probability of default table
        default_prob = pd.DataFrame(np.zeros([self.num_months()+5, 14]), index=self.months_extended(), columns=list(range(0,14)))
        # Fill in the probability of default table
        for month in self.months_extended():
            for bucket in range(14):
                default_prob[bucket][month] = np.product([net_flow[x][month] for x in range(bucket, 14)])
        return default_prob
        
    def provision_rate(self):
        net_flow = self.net_flow()
        loss_rate = self.loss_rate()
        provision_rate_pre_month_0 = self.provision_rate_pre_month_0()
        default_prob = self.default_prob()
        ## Provision rate        
        # Create a table for provision by bucket and month
        provision_rate = pd.DataFrame(np.zeros([self.num_months()+5, 14]), index=self.months_extended(), columns=list(range(0,14)))
        # Fill in provision rate table
        for month in self.months():
            for bucket in range(14):
                provision_rate[bucket][month] = (np.average([default_prob[bucket][month+x] for x in range(-5,1)])*
                                                 pd.Series.product(loss_rate.loc[month]))       
        provision_rate.loc[-5, -4, -3, -2, -1] =  provision_rate_pre_month_0
        return provision_rate
    
    def provision(self):
        provision_rate = self.provision_rate()
        enr_by_bucket_extended = self.enr_by_bucket_extended()
        provision_for_fraud_rate = self.provision_for_fraud_rate()
        ## Provision
        # Create a table for provision
        provision = pd.Series(np.zeros([self.num_months()]), index = self.months())
        # Fill in the provision table
        for month in self.months():
            sumproduct1 = 0
            sumproduct2 = 0
            for bucket in range(5):
                sumproduct1 += provision_rate[bucket][month-1]*enr_by_bucket_extended[bucket][month-1]
            for bucket in range(4):
                sumproduct2 += provision_rate[bucket][month-2]*enr_by_bucket_extended[bucket][month-2] 
            provision[month] = (sumproduct1-sumproduct2)*(1+provision_for_fraud_rate[month])
        return provision    
    
    def outputs(self):
        out = self.out()
        return add_provision_to_toi(add_nim_actual(out))


# In[8]:


class Loan(Lending):
    '''
    Class of all lending products and sub-products
    '''
        
    def disbursement(self):    
        # Forecasted monthly disbursement in VND bn
        return self.file.parse(sheetname='disbursement', header=None).loc[1]
    
    def disbursement_month_minus_1(self):
        # Disbursement in latest historical month
        return self.file.parse(sheetname='disbursement_month_minus_1', header=None)[0][1]
        
    def n0_to_gb(self):     
        # Percentage of group 1 loan (N0) in good bank
        return self.file.parse(sheetname='n0_to_gb', header=None).loc[1]
    
    def overdue_to_gb(self):
        # Percent of group 2 loan in good bank
        return self.file.parse(sheetname='overdue_to_gb', header=None).loc[1]
    
    def n0_eop_month_minus_1(self):
        # Good bank group 1 balance in latest historical month
        return self.file.parse(sheetname='n0_eop_month_minus_1', header=None)[0][1]
    
    def overdue_eop_month_minus_1(self):
        # Good bank group 2 balance in latest historical month
        return self.file.parse(sheetname='overdue_eop_month_minus_1', header=None)[0][1]
    
    def nfi_to_disbursement(self):
        # Net fee income from insurance as percentage of disbursement
        return self.file.parse(sheetname='nfi_to_disbursement', header=None).loc[1]
    
    def enr_by_bucket_extended(self):
        flow_rate = self.flow_rate()
        net_flow = self.net_flow()
        disbursement = self.disbursement()
        # Create a table for end of period balance, both good bank and bad bank (ENR)
        enr_by_bucket = pd.DataFrame(np.zeros([self.num_months(), 14]), index=self.months(), columns=list(range(0,14)))
        # Join the ENR table with historical ENR
        enr_by_bucket_extended = pd.concat([self.enr_pre_month_0(), enr_by_bucket])  
        # Fill in the ENR table
        for month in self.months():
            for bucket in range(14):
                if bucket == 0:
                    enr_by_bucket_extended[bucket][month] = (enr_by_bucket_extended[bucket][month-1]*flow_rate['B0-B0'][month] +
                                                        enr_by_bucket_extended[bucket+1][month-1]*flow_rate['B1-B0'][month] +
                                                        disbursement[month])
                else:
                    enr_by_bucket_extended[bucket][month] = enr_by_bucket_extended[bucket-1][month-1]*net_flow[bucket-1][month]                 
        return  enr_by_bucket_extended

    def n0_eop(self):
        # EOP N0 (group 1 loan) balance
        return self.total_balance_eop()*self.n0_to_gb()
    
    def cii_accrual(self):
        # Interest payment from group 1 loan (N0)
        return adb(self.n0_eop(), self.n0_eop_month_minus_1())*self.cii_rate()/365*self.days_in_month()
    
    def overdue_eop(self):
        # EOP group 2 balance of a given month
        return self.total_balance_eop()*self.overdue_to_gb()
    
    def cii_reinstated(self):
        # Reinstated interest payment in group 2 loan
        return adb(self.overdue_eop(), self.overdue_eop_month_minus_1())*self.cii_rate()/365*self.days_in_month()
    
    def cii(self):
        # Total interest payment
        return self.cii_accrual()+self.cii_reinstated()
    
    def nfi(self):
        # NFI from insurance
        disbursement = self.disbursement()
        disbursement_previous_month = disbursement.shift(1)
        disbursement_previous_month[0] = self.disbursement_month_minus_1()
        insurance_charged_amount = disbursement*2/3 + disbursement_previous_month*1/3
        return insurance_charged_amount*self.nfi_to_disbursement()      

    def out(self):
        out = pd.concat([self.months_names(), self.disbursement(), self.total_balance_eop(), self.total_balance_adb(), self.nii(), self.nfi(), self.toi(), self.provision()], axis=1)
        out.columns = ['month', 'disbursement','total_balance_eop', 'total_balance_adb', 'nii', 'nfi', 'toi', 'provision']
        out = out.apply(pd.to_numeric, errors='ignore')
        return out


# In[9]:


class UPL(Loan):
    def __init__(self, master_input_folder, input_file_name):        
        self.path = master_input_folder+'/UPL/'+input_file_name+'.xlsx'
        self.file = pd.ExcelFile(self.path)
        
class AutoLoan(Loan):
    def __init__(self, master_input_folder, input_file_name):       
        self.path = master_input_folder+'/AutoLoan/'+input_file_name+'.xlsx'
        self.file = pd.ExcelFile(self.path)
        
class ConsumptionLoan(Loan):
    def __init__(self, master_input_folder, input_file_name):        
        self.path = master_input_folder+'/ConsumptionLoan/'+input_file_name+'.xlsx'
        self.file = pd.ExcelFile(self.path)
        
class BusinessLoan(Loan):
    def __init__(self, master_input_folder, input_file_name):       
        self.path = master_input_folder+'/BusinessLoan/'+input_file_name+'.xlsx'
        self.file = pd.ExcelFile(self.path)
        
class HomeLoan(Loan):
    def __init__(self, master_input_folder, input_file_name):       
        self.path = master_input_folder+'/HomeLoan/'+input_file_name+'.xlsx' 
        self.file = pd.ExcelFile(self.path)


# In[10]:


class CreditCard(Lending):
    '''
    Class of all credit card products and sub-products
    '''
    def __init__(self, master_input_folder, input_file_name):
        self.path = master_input_folder+'/CreditCard/'+input_file_name+'.xlsx'
        self.file = pd.ExcelFile(self.path)
        
    def retail_spend_per_activated(self):    
        # Retail spend per activated card in VND mn
        return self.file.parse(sheetname='retail_spend_per_activated', header=None).loc[1]
        
    def onl_spend_per_activated(self):  
        # Online spend per activated card in VND mn
        return self.file.parse(sheetname='onl_spend_per_activated', header=None).loc[1]
    
    def cash_spend_per_activated(self):
        # Cash spend per activated card in VND mn
        return self.file.parse(sheetname='cash_spend_per_activated', header=None).loc[1]
    
    def monthly_issued(self):
        # Number of cards issued each month
        return self.file.parse(sheetname='monthly_issued', header=None).loc[1]
    
    def total_issued_month_minus_1(self):
        # Total number of issued cards at latest historical month
        return self.file.parse(sheetname='total_issued_month_minus_1', header=None)[0][1]
    
    def activation_rate(self):
        # Activation rate (% activated cards in total issued cards)
        return self.file.parse(sheetname='activation_rate', header=None).loc[1]
    
    def attrition_rate(self):
        # Attrition rate (% cards closed each month in total issued cards)
        return self.file.parse(sheetname='attrition_rate', header=None).loc[1]
    
    def revolving_rate(self):
        # Revolving rate
        return self.file.parse(sheetname='revolving_rate', header=None).loc[1]
    
    def annual_fee_amount(self):
        # Annual fee per card in VND mn
        return self.file.parse(sheetname='annual_fee_amount', header=None).loc[1]
    
    def annual_fee_waiver_rate(self):
        # Annual fee waiver %
        return self.file.parse(sheetname='annual_fee_waiver_rate', header=None).loc[1]
    
    def annual_fee_due_rate(self):
        # % of cards with annual fee due in a given month
        return self.file.parse(sheetname='annual_fee_due_rate', header=None).loc[1]
    
    def interchange_rate(self):
        # Interchange rate
        return self.file.parse(sheetname='interchange_rate', header=None).loc[1]
    
    def mastercard_billing_rate(self):
        # MasterCard payment on interchange
        return self.file.parse(sheetname='mastercard_billing_rate', header=None).loc[1]
    
    def cash_advance_fee_rate(self):
        # Cash advance fee rate
        return self.file.parse(sheetname='cash_advance_fee_rate', header=None).loc[1]
    
    def foreign_transaction_rate(self):
        # % of foreign transaction amount in total spend
        return self.file.parse(sheetname='foreign_transaction_rate', header=None).loc[1]
    
    def fx_fee_rate(self):
        # Foreign exchange fee rate on foreign transactions
        return self.file.parse(sheetname='fx_fee_rate', header=None).loc[1]
    
    def late_payment_rate(self):
        # Late payment rate
        return self.file.parse(sheetname='late_payment_rate', header=None).loc[1]
    
    def min_payment_revolver(self):
        # Minimum payment on revolver
        return self.file.parse(sheetname='min_payment_revolver', header=None).loc[1]
    
    def loyalty_rate(self):
        # Loyalty expense as % of spend
        return self.file.parse(sheetname='loyalty_rate', header=None).loc[1]
    
    def cash_back_rate(self):
        # Cashback as % of spend
        return self.file.parse(sheetname='cash_back_rate', header=None).loc[1]
    
    def benefits_purchase_expense(self):
        # Benefits purchase expense (e.g., airlines miles) in VND bn
        return self.file.parse(sheetname='benefits_purchase_expense', header=None).loc[1]
    
    def other_expense_per_card(self):
        # Other expenses per activated card (e.g., SMS fee)
        return self.file.parse(sheetname='other_expense_per_card', header=None).loc[1]     
           
    def total_issued(self):
        ## Total number of issued cards by month
        monthly_issued = self.monthly_issued()
        attrition_rate = self.attrition_rate()
        # Create a table for total issued cards by month
        total_issued = pd.Series(np.zeros([self.num_months()]), index=self.months())
        # Fill in total_issued table
        total_issued[0] = self.total_issued_month_minus_1()*(1-attrition_rate[0]) + monthly_issued[0]
        for month in self.months()[1:]:
                total_issued[month] = total_issued[month-1]*(1-attrition_rate[month]) + monthly_issued[month]  
        return total_issued
    
    def total_activated(self):
        # Total number of activated cards by month
        return self.total_issued()*self.activation_rate()
    
    def retail_spend(self):
        # Total retail spending in VND bn
        return self.total_activated()*self.retail_spend_per_activated()/1000
    
    def onl_spend(self):
        # Total online spending in VND bn
        return self.total_activated()*self.onl_spend_per_activated()/1000
    
    def cash_spend(self):
        # Total cash spending in VND bn
        return self.total_activated()*self.cash_spend_per_activated()/1000
    
    def total_spend(self):
        # Total spending in VND bn
        return self.retail_spend() + self.onl_spend() + self.cash_spend()
    
    def enr_by_bucket_extended(self):
        net_flow = self.net_flow()
        revolving_rate = self.revolving_rate()
        total_spend = self.total_spend()
        # Create a table for end of period balance, both good bank and bad bank (ENR)
        enr_by_bucket = pd.DataFrame(np.zeros([self.num_months(), 14]), index=self.months(), columns=list(range(0,14)))
        # Join the ENR table with historical ENR
        enr_by_bucket_extended = pd.concat([self.enr_pre_month_0(), enr_by_bucket])  
        # Fill in the ENR table
        for month in self.months():
            for bucket in range(14):
                if bucket == 0:
                    enr_by_bucket_extended[bucket][month] = (enr_by_bucket_extended[bucket][month-1] + total_spend[month])*revolving_rate[month]
                else:
                    enr_by_bucket_extended[bucket][month] = enr_by_bucket_extended[bucket-1][month-1]*net_flow[bucket-1][month]                                                                              
        return enr_by_bucket_extended
        
    def cii(self):
        # Interest payment from customers
        enr_by_bucket = self.enr_by_bucket()
        anr_b0_1 = adb(enr_by_bucket.loc[:, 0:1].sum(axis=1), self.enr_pre_month_0().loc[-1,0:1].sum())
        return anr_b0_1*self.cii_rate()/365*self.days_in_month()
    
    def annual_fee(self):
        # Total annual fee
        return (1-self.annual_fee_waiver_rate())*self.annual_fee_amount()*self.annual_fee_due_rate()/1000*self.total_activated()
    
    def interchange_fee(self):
        # Total interchange fee net MasterCard billing
        return (self.retail_spend()+self.onl_spend())*self.interchange_rate()*(1-self.mastercard_billing_rate())
    
    def cash_advance_fee(self):
        # Total cash advance fee
        return self.cash_spend()*self.cash_advance_fee_rate()
    
    def fx_fee(self):
        # Total foreign exchange fee
        return self.foreign_transaction_rate()*self.total_spend()*self.fx_fee_rate()
    
    def late_payment_fee(self):
        # Total late payment fee
        enr_b0_previous_month = self.enr_by_bucket_extended().loc[-1:10,0] # look up ENR bucket 0 of the previous month
        enr_b0_previous_month.index = self.months()
        return self.min_payment_revolver()*(self.revolving_rate()*self.late_payment_rate()*enr_b0_previous_month)
    
    def cash_back_expense(self):
        # Total cash back expense
        return (self.retail_spend()+self.onl_spend())*self.cash_back_rate()
    
    def loyalty_expense(self):
        # Total loyalty expense
        return (self.retail_spend()+self.onl_spend())*self.loyalty_rate()
    
    def other_expenses(self):
        # Other net expenses
        return self.other_expense_per_card()*self.total_activated()/1000

    def nfi(self):
        return self.annual_fee()+self.interchange_fee()+self.cash_advance_fee()+self.fx_fee()+self.late_payment_fee()                 - self.cash_back_expense() - self.loyalty_expense() - self.other_expenses()

    def out(self):
        out = pd.concat([self.months_names(), self.monthly_issued(), self.total_activated(), self.total_balance_eop(), self.total_balance_adb(), self.nii(), self.nfi(), self.toi(), self.provision()], axis=1)
        out.columns = ['month', 'monthly_issued', 'total_activated', 'total_balance_eop', 'total_balance_adb', 'nii', 'nfi', 'toi', 'provision']
        out = out.apply(pd.to_numeric, errors='ignore')
        return out


# In[11]:


class Deposit(Product):
    '''
    Class of all CASA, TD products
    '''
        
    def num_active_customers(self):
        # Number of active customers
        return self.file.parse(sheetname='num_active_customers', header=None).loc[1]
    
    def eop_per_active_customer(self):
        # Monthly EOP balance per active customers
        return self.file.parse(sheetname='eop_per_active_customer', header=None).loc[1]
    
    def eop_month_minus_1(self):
        # EOP balance in latest historical month
        return self.file.parse(sheetname='eop_month_minus_1', header=None)[0][1]
    
    def vof_rate(self):
        # value of fund rate
        return self.file.parse(sheetname='vof_rate', header=None).loc[1]
    
    def nfi_per_active_customer(self):
        # NFI per active customer
        return self.file.parse(sheetname='nfi_per_active_customer', header=None).loc[1]
        
    def total_balance_eop(self):
        return self.num_active_customers()*self.eop_per_active_customer()
    
    def total_balance_adb(self):
        return adb(self.total_balance_eop(), self.eop_month_minus_1())
    
    def nii(self):
        total_balance_adb = self.total_balance_adb()
        return total_balance_adb*(self.vof_rate() - self.cii_rate())/365*self.days_in_month()
    
    def nim_actual(self):
        return self.nii()/self.total_balance_adb()
    
    def nfi(self):
        return self.nfi_per_active_customer()*self.num_active_customers()
    
    def toi(self):
        return self.nii()+self.nfi()
    
    def out(self):
        out = pd.concat([self.months_names(), self.total_balance_eop(), self.total_balance_adb(), self.nii(), self.nfi(), self.toi()], axis=1)
        out.columns = ['month', 'total_balance_eop', 'total_balance_adb', 'nii', 'nfi', 'toi']
        out = out.apply(pd.to_numeric, errors='ignore')
        return out
    
    def outputs(self):
        out = self.out()
        return add_nim_actual(out)


# In[12]:


class CASA(Deposit):
    def __init__(self, master_input_folder, input_file_name):
        self.path = master_input_folder+'/CASA/'+input_file_name+'.xlsx'
        self.file = pd.ExcelFile(self.path)
        
class TD(Deposit):
    def __init__(self, master_input_folder, input_file_name):
        self.path = master_input_folder+'/TD/'+input_file_name+'.xlsx'
        self.file = pd.ExcelFile(self.path)


# # Build aggregations

# In[13]:


def inputs_dir(master_input_folder):
    return {UPL: master_input_folder+'/UPL',
              AutoLoan: master_input_folder+'/AutoLoan',
              HomeLoan: master_input_folder+'/HomeLoan',
              ConsumptionLoan: master_input_folder+'/ConsumptionLoan',
              BusinessLoan: master_input_folder+'/BusinessLoan',
              CreditCard: master_input_folder+'/CreditCard',
              CASA: master_input_folder+'/CASA',
              TD: master_input_folder+'/TD'
            }

product_name_dict = {'UPL': UPL,
                     'AutoLoan': AutoLoan,
                     'HomeLoan': HomeLoan,
                     'ConsumptionLoan': ConsumptionLoan,
                     'BusinessLoan': BusinessLoan,
                     'CreditCard': CreditCard,
                     'CASA': CASA, 
                     'TD': TD}


# In[14]:


def list_all_sub_products(master_input_folder, product_name):
    product_class = product_name_dict[product_name]
    path = inputs_dir(master_input_folder)[product_class]  
    asp = []
    for _,__,files in os.walk(path):
        for file in files:
            if file.endswith('.xlsx'):
                asp.append(file[:-5])
    return asp               

def total(master_input_folder, product_name, *arg):
    '''
    Product a combined output of all sub products
    :params:
    input_folder_name: name of folder containing all input files for all products, default to 'inputs'
    input_sub_folder_name: name of folder containing all input files for a given product
    *arg: name of all input files containing inputs for subproducts to be combined
    '''
    asp=[]
    product_class = product_name_dict[product_name]
    if arg == ():
        asp = list_all_sub_products(master_input_folder, product_name)
    else:       
        asp = [file for file in arg]
    product = product_class(master_input_folder, asp[0]).out()
    months_names = product['month']
    for file in asp[1:]:
        product = product.add(product_class(master_input_folder, file).out())
    product = add_nim_actual(product)        
    if issubclass(product_class, Lending):
        product = add_provision_to_toi(product)
    product['month'] = months_names 
    return product 


# In[15]:


aggregates = ['all deposit', 'all secured lending','all unsecured lending',
              'all lending except cards', 'all lending including cards', 'all products']

# Build a list of indicators applicable to a given product
indicators = {}
indicators['all products'] = ['total_balance_eop', 'total_balance_adb', 'nii', 'nfi', 'toi']
indicators['all deposits'] = indicators['all products']
indicators['all lending including cards'] = indicators['all products'] + ['provision']
indicators['all secured lending'] = indicators['all lending including cards']
indicators['all unsecured lending'] = indicators['all lending including cards']
indicators['all lending except cards'] = indicators['all lending including cards']
indicators['UPL'] = indicators['all lending including cards'] + ['disbursement']
indicators['AutoLoan'] = indicators['UPL']
indicators['HomeLoan'] = indicators['UPL']
indicators['ConsumptionLoan'] = indicators['UPL']
indicators['BusinessLoan'] = indicators['UPL']
indicators['CASA'] = indicators['all deposits']
indicators['TD'] = indicators['all deposits']
indicators['CreditCard'] = indicators['all lending including cards'] + ['monthly_issued', 'total_issued', 'total_activated', 'total_spend']

# Build a product classification
categories = {}
categories['all secured lending'] = ['AutoLoan', 'HomeLoan', 'ConsumptionLoan', 'BusinessLoan']
categories['all unsecured lending'] = ['UPL', 'CreditCard']
categories['all lending except cards'] = categories['all secured lending'] + ['UPL']
categories['all lending including cards'] = categories['all lending except cards'] + ['CreditCard']
categories['all deposits'] = ['CASA', 'TD']
categories['all products'] = categories['all lending including cards'] + categories['all deposits']


# In[16]:


def aggregation(master_input_folder, chosen_product):
    products_list = categories[chosen_product]
    indicators_list = indicators[chosen_product]
    products = []
    # Initialize products
    for product in products_list:
        products.append(total(master_input_folder=master_input_folder, product_name=product)[['month']+indicators_list])
    output = products[0]
    for product_output in products[1:]:
        output += product_output
    output['month'] = products[0][['month']]    
    return output


# # Build interface

# In[17]:


def choose_folder():
    master_input_folder = input()
    return master_input_folder


# In[18]:


def list_product_tree(master_input_folder):
    '''
    Generate a dictionary of subfolders and files given a folder
    '''
    product_tree = {}
    for folder,_,files in os.walk(master_input_folder):
        sub_products = []
        for file in files:
            if file.endswith('.xlsx'):
                sub_products.append(file[:-5])
        if folder != master_input_folder:        
            product_tree[folder[len(master_input_folder)+1:]] = sub_products
    return product_tree


# In[19]:


def choose_product(product_tree):
    '''
    Choose a product given a master input folder
    '''
    product_list = [product for product in product_tree.keys()] + aggregates
    current = interactive(f, product=product_list)
    dp(current)
    return current

def choose_subproduct(chosen_product, product_tree):
    '''
    Choose a subproduct given a product
    '''
    subproduct_list = product_tree[chosen_product]
    current = interactive(f, product=subproduct_list+['all sub-products'])
    dp(current)
    return current

def f(product):
    return product


# In[20]:


def step_1():
    printmd('**Instruction**')
    printmd('1) Type the name of the **input data folder** you want to use for forecasting')
    master_input_folder = choose_folder()
    printmd('2) See above the list of **products and subproducts** contained in the folder')
    product_tree = list_product_tree(master_input_folder)
    interact(f, product=product_tree)
    return master_input_folder, product_tree


# In[21]:


def step_2b(choice_product, product_tree):
    chosen_product = choice_product.result
    printmd('**Instruction**')
    if chosen_product in aggregates:
        printmd('&rarr; Move to step 3')
        return None
    else:
        printmd('Choose a specific sub-product, or you can choose all sub-products')
        choice_sub_product = choose_subproduct(chosen_product, product_tree)
        return choice_sub_product


# In[22]:


def step_3(choice_product, choice_subproduct, master_input_folder):
    chosen_product = choice_product.result
    if chosen_product in aggregates:
        return aggregation(master_input_folder, chosen_product)
    else:
        chosen_subproduct = choice_subproduct.result
        if chosen_subproduct == 'all sub-products':
            return total(master_input_folder, chosen_product)
        else:
            subproduct_name = product_name_dict[chosen_product](master_input_folder, chosen_subproduct)
            return subproduct_name.out()






