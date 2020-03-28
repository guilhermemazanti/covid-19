# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 01:06:07 2020

@author: Guilherme Mazanti
"""

from DailyData import model, DailyData
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATEDELTA = [0, 1, 3, 7]

# =============================================================================
# Class CompareCSV
# =============================================================================
class CompareCSV():
  def __init__(self, baseDate):
    self.dates = {i: baseDate + timedelta(i) for i in DATEDELTA}
    self.csv = {i: pd.read_csv(self.dates[i].isoformat() + ".csv",\
                               index_col = "Country code")\
                   for i in DATEDELTA}
    
    self.names = self.csv[0]["Country name"].to_dict()

  # ---------------------------------------------------------------------------
  # Evaluation
  # ---------------------------------------------------------------------------
  def getErrors(self):
    if not hasattr(self, "errors"):
      forecasts = {i:{} for i in DATEDELTA[1:]}
      realValues = {i:{} for i in DATEDELTA[1:]}
      for k, v in self.csv[0].iterrows():
        forecasts[1][k] = v["Estimate of number of cases tomorrow"]
        forecasts[3][k] = v["Estimate of number of cases in 3 days"]
        forecasts[7][k] = v["Estimate of number of cases in 7 days"]
      for i in DATEDELTA[1:]:
        for k, v in self.csv[i].iterrows():
          realValues[i][k] = v["Total cases today"]
          
      self.errors = {i:{k:(forecasts[i][k] - realValues[i][k])/realValues[i][k]\
                        for k in forecasts[i]} for i in DATEDELTA[1:]}
    return self.errors
  
  # ---------------------------------------------------------------------------
  # Histogram
  # ---------------------------------------------------------------------------
  def errorHistogram(self, days):
    if days in DATEDELTA[1:]:
      fig, ax = plt.subplots()
      ax.grid(True)
      ax.set_axisbelow(True)
      bins = np.arange(-95, 96, 10)
      data = 100*np.array(list(self.getErrors()[days].values()))
      ax.hist(data, bins, ec="black")
      
      ax.bar([-100, 100], [(data<-95).sum(), (data>95).sum()],\
             width=10, ec="black", fc="C1")
      
      ax.set_xlabel("Relative error [%]")
      ax.set_ylabel("Number of countries")
      ax.set_title("Relative error histogram for a {:d}-day forecast".format(days))
      ax.set_ylim([0, 26])

class CompareGraphs():
  def __init__(self, date0, date1):
    self.date0 = date0
    self.date1 = date1
    self.dd0 = DailyData(date0)
    self.dd1 = DailyData(date1)

  def plotCountry(self, code, logScale = False):
    fig = plt.figure()
    ax = fig.add_axes([0.12, 0.22, 0.85, 0.68])
    ax.grid(True)
    ax.set_axisbelow(True)
    
    dates, cases, _ = self.dd1.getCountryDatesCasesDeaths(code)
    dates0, cases0, _ = self.dd0.getCountryDatesCasesDeaths(code)
    assert dates0[0]==dates[0] # Otherwise something's wrong !
    A, a, tau = self.dd0.getCountryBestFit(code)
    
    ax.plot(dates0, cases0.cumsum(), lw=2.75, label="Data used for training")
    
    ax.plot(dates[dates >= self.date0], cases.cumsum()[dates >= self.date0],\
            lw=2.75, label="Observed data")
    
    firstDayPredict = (self.date0 - dates[0]).days
    lastDayPredict = (self.date1 - dates[0]).days
    t = np.arange(firstDayPredict, lastDayPredict+1)
    predCases = model(t, A, a, tau)
    
    ax.plot([dates[0] + timedelta(int(ti)) for ti in t], predCases, lw=1.25,\
            label = "Estimated data")
    for text in ax.get_xticklabels():
      text.set_rotation(45)
    ax.set_xlabel("Date")
    ax.set_ylabel("Total number of cases")
    ax.set_title("Total number of cases by date for "\
                 +self.dd1.getCountryName(code))
    ax.legend()
    if logScale:
      ax.set_yscale("log")
    