# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 23:59:00 2020

@author: Guilherme Mazanti
"""

from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import requests as rq
import scipy.optimize as opt
import warnings as w

# =============================================================================
# Models
# =============================================================================
def model(t, A, a, tau):
  return A * (np.tanh(a*(t - tau)) + 1) / 2

def jac(t, A, a, tau):
  return np.array([np.tanh(a*(t - tau)) + 1,\
    A*(t - tau)/(np.cosh(a*(t - tau))**2),\
    -A*a/(np.cosh(a*(t - tau))**2)]).transpose() / 2
  
# =============================================================================
# Class DailyData
# =============================================================================

class DailyData():
  def __init__(self, myDate):
    self.date = myDate
    self.strDate = myDate.isoformat()
    self.countryNameColumn = "Countries and territories"
    self.plotter = Plots(self)
    
    # If the file does not exist, get it from the internet
    if not os.path.exists(self.strDate + ".xlsx"):
      webData = rq.get("https://www.ecdc.europa.eu/sites/default/files/"
                       "documents/COVID-19-geographic-disbtribution-worldwide-"+
                       self.strDate + ".xlsx")
      with open(self.strDate + ".xlsx", "wb") as file:
        file.write(webData.content)
        
    self.db = pd.read_excel(self.strDate + ".xlsx", keep_default_na = False)
    
    # Column names of the .xlsx files have changed in time. The next lines
    # compensate for these changes
    self.db.columns = [c[0].upper()+c[1:] for c in self.db.columns]
    
    if self.countryNameColumn not in self.db.columns:
      self.countryNameColumn = "CountriesAndTerritories"
      
    # All country codes
    self.countryCodes = set(self.db.GeoId.values)
    
    # Dictionaries indexed by country code which will contain data relevant to
    # that country
    self.names = {} # Name of the country
    self.dates = {} # 1D numpy array with dates
    self.cases = {} # 1D numpy array with reported cases
    self.deaths = {} # 1D numpy array with reported deaths
    self.bestFit = {} # tuple containing the parameters of the best fit
   
  # ---------------------------------------------------------------------------
  # Methods for data collection
  # ---------------------------------------------------------------------------
  
  def getCountryName(self, countryCode):
    if countryCode not in self.names:
      self.names[countryCode] = \
        self.db[self.db.GeoId==countryCode][self.countryNameColumn].values[0]
    return self.names[countryCode]
    
  def getCountryDatesCasesDeaths(self, countryCode):
    """
    Get dates, cases and deaths for a country from the current dataframe.
    Results are returned as numpy arrays and also stored in the class
    dictionaries self.dates, self.cases, self.deaths.
    
    Inputs:
      countryCode: Country code (according to the GeoId column of the
        dataframe).
      
    Outputs:
      dates: 1D numpy array with increasing dates. All dates are necessarily
        subsequent. The first date corresponds necessarily to the firs reported
        case, the last date corresponds to self.date. These constraints do not
        hold if no cases have been reported for that country.
      cases: 1D numpy array with the same shape as dates such that cases[i]
        contains the number of reported cases in day dates[i].
      deaths: same as cases, but with reported deaths instead.
      
    Exceptions:
      Exception: if countryCode does not appear in db.
      
    Warnings:
      UserWarning: if there are no cases reported for the country.
    """
    if countryCode in self.dates:
      if self.cases[countryCode].sum() == 0:
        w.warn("No cases reported in "+countryCode, stacklevel=2)
      return self.dates[countryCode],\
             self.cases[countryCode],\
             self.deaths[countryCode]
    
    dbCountry = self.db[self.db.GeoId==countryCode]
    if dbCountry.empty:
      raise Exception("No data for "+countryCode)
    
    # Data extraction
    dates = zip(dbCountry.Day.values, dbCountry.Month.values,\
                dbCountry.Year.values)
    dates = np.array([date(y, m, d) for d, m, y in dates])
    cases = dbCountry.Cases.values
    deaths = dbCountry.Deaths.values
    
    # Sort by increasing date
    ind = dates.argsort()
    
    dates = dates[ind]
    cases = cases[ind]
    deaths = deaths[ind]
      
    # Erasing all data until the first case
    firstCase = np.flatnonzero(cases)
    firstCase = firstCase[0] if firstCase.size > 0 else None
    
    if firstCase is not None:
      dates = dates[firstCase:]
      cases = cases[firstCase:]
      deaths = deaths[firstCase:]
    else:
      w.warn("No cases reported in "+countryCode, stacklevel=2)
    
    # We guarantee that all days appear until the current date
    N = (self.date - dates[0]).days + 1
    self.dates[countryCode] = np.empty(N, dtype=object)
    self.cases[countryCode] = np.zeros(N, dtype=int)
    self.deaths[countryCode] = np.zeros(N, dtype=int)
    
    self.dates[countryCode][0] = dates[0]
    self.cases[countryCode][0] = cases[0]
    self.deaths[countryCode][0] = deaths[0]
    j = 1
    for i in range(1, N):
      self.dates[countryCode][i] = self.dates[countryCode][i-1] + timedelta(1)
      if j < dates.size and dates[j] == self.dates[countryCode][i]:
        self.cases[countryCode][i] = cases[j]
        self.deaths[countryCode][i] = deaths[j]
        j += 1
    
    return self.dates[countryCode],\
           self.cases[countryCode],\
           self.deaths[countryCode]
  
  # ---------------------------------------------------------------------------
  # Estimations
  # ---------------------------------------------------------------------------
  def getCountryBestFit(self, countryCode):
    """
    Fits a function of the form
    f(t) = A * [tahn(a*(t - tau)) + 1] / 2
    finding the best parameters A, a, tau.
    
    Returns the parameters and store them in self.bestFit
    """
    if countryCode not in self.bestFit:
      _, cases, _ = self.getCountryDatesCasesDeaths(countryCode)
      params, _ = opt.curve_fit(model, np.arange(cases.size), cases.cumsum(),\
        p0 = [cases.sum(), 0.15, 50], check_finite=True, jac = jac,\
        maxfev = 10000)
      self.bestFit[countryCode] = params
    return self.bestFit[countryCode]
  
  def bestFitAll(self, minCases = 100, minDates = 10):
    """
    Finds the best fit of all countries with at least minCases cases and for
    which we have at least minDates days of data.
    """
    for code in self.countryCodes:
      dates, cases, _ = self.getCountryDatesCasesDeaths(code)
      if cases.sum() >= minCases and dates.size >= minDates:
        self.getCountryBestFit(code)
        
  def saveParams(self):
    with open(self.strDate + ".csv", "w") as file:
      file.write("Country code,Country name,Total cases today,"
                 "Estimate of final number of cases,"
                 "Estimate of number of cases tomorrow,"
                 "Estimate of number of cases in 3 days,"
                 "Estimate of number of cases in 7 days,"
                 "Number of days for doubling,Number of days until inflection,"
                 "Inflection date\n")
      
      for code in self.countryCodes:
        if code in self.bestFit:
          countryName = self.getCountryName(code)
          dates, cases, deaths = self.getCountryDatesCasesDeaths(code)
          totalCases = cases.sum()
          date0 = dates[0]
          A, a, tau = self.getCountryBestFit(code)
          nbDouble = np.log(2)/a
          time1Days = (self.date - date0).days + 1
          cases1Days = model(time1Days, A, a, tau)
          time3Days = (self.date - date0).days + 3
          cases3Days = model(time3Days, A, a, tau)
          time7Days = (self.date - date0).days + 7
          cases7Days = model(time7Days, A, a, tau)
          file.write("{:s},{:s},{:d},{:d},{:d},{:d},{:d},{:.2f},{:d},{:s}\n"\
            .format(code, countryName, totalCases,\
                    int(A), int(cases1Days), int(cases3Days), int(cases7Days),\
                    nbDouble, int(tau+0.5),\
                    (date0+timedelta(int(tau+0.5))).isoformat()))
          
# =============================================================================
# Class for plotting
# =============================================================================
class Plots():
  def __init__(self, parent):
    self.parent = parent
    
  def totalCasesByDate(self, listCountryCodes, logScale = False):
    """
    Plots the total number of cases as a function of the day for the countries
    whose codes are in listCountryCodes
    """
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.22, 0.85, 0.68])
    ax.grid(True)
    ax.set_axisbelow(True)
    for countryCode in listCountryCodes:
      dates, cases, _ = self.parent.getCountryDatesCasesDeaths(countryCode)
      ax.plot(dates, cases.cumsum(), label=countryCode)
    for text in ax.get_xticklabels():
      text.set_rotation(45)
    ax.set_xlabel("Date")
    ax.set_ylabel("Total number of cases")
    ax.set_title("Total number of cases by date")
    ax.legend()
    if logScale:
      ax.set_yscale("log")
      
  def totalCasesByDayN(self, listCountryCodes, nbCases = 1, logScale = False):
    """
    Plots the total number of cases a function of the number of days since the
    country first reached nbCases total cases, for all countries whose codes
    are in listCountryCodes
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_axisbelow(True)
    for countryCode in listCountryCodes:
      _, cases, _ = self.parent.getCountryDatesCasesDeaths(countryCode)
      totalCases = cases.cumsum()
      ax.plot(totalCases[totalCases >= nbCases], label=countryCode)
    ax.set_xlabel("Days since case number {:d}".format(nbCases))
    ax.set_ylabel("Total number of cases")
    ax.set_title("Total number of cases by days since case number {:d}"\
                 .format(nbCases))
    ax.legend()
    if logScale:
      ax.set_yscale("log")
      
  def bestFitByDayN(self, listCountryCodes, nbCases = 1, nbDays = 100,\
                     logScale = False):
    """
    Plots the best fit for the total number of cases as a function of the
    number of days since the country first reached nbCases total cases, for all
    countries whose codes are in listCountryCodes, and for a total of nbDays
    after the first day
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_axisbelow(True)
    for countryCode in listCountryCodes:
      #try:
        A, a, tau = self.parent.getCountryBestFit(countryCode)
        dates, cases, _ = self.parent.getCountryDatesCasesDeaths(countryCode)
        t = np.arange(nbDays)
        offset = (cases < nbCases).sum()
        totCases = model(t + offset, A, a, tau)
        ax.plot(totCases, label=countryCode)
      #except RuntimeError:
      #  w.warn("Could not find optimal curve for "+countryCode)
    ax.set_xlabel("Days since case number {:d}".format(nbCases))
    ax.set_ylabel("Total number of cases")
    #ax.set_ylim([0, 100000])
    ax.set_title("Estimate of the total number of cases by days since case"\
                 " number {:d}".format(nbCases))
    ax.legend()
    if logScale:
      ax.set_yscale("log")
      
  def totalCasesAndBestFit(self, code, future = 0, logScale = False):
    """
    Plots both total cases and best fit for the country whose code corresponds
    to the variable code. The best fit is plotted for an extra number of days
    corresponding to the variable future.
    """
    countryName = self.parent.getCountryName(code)
    
    fig = plt.figure()
    ax = fig.add_axes([0.12, 0.22, 0.85, 0.68])
    ax.grid(True)
    ax.set_axisbelow(True)
    dates, cases, _ = self.parent.getCountryDatesCasesDeaths(code)
    A, a, tau = self.parent.getCountryBestFit(code)
    nbDays = (self.parent.date - dates[0]).days + 1 + future
    t = np.arange(nbDays)
    totCases = model(t, A, a, tau)
    ax.plot(dates, cases.cumsum(), label="Observed", lw=2.75)
    ax.plot([dates[0] + timedelta(int(ti)) for ti in t], totCases,\
             label="Estimate", lw=1.25)
    for text in ax.get_xticklabels():
      text.set_rotation(45)
    ax.set_xlabel("Date")
    ax.set_ylabel("Total number of cases")
    ax.set_title("Total number of cases by date for "+countryName)
    ax.legend()
    if logScale:
      ax.set_yscale("log")