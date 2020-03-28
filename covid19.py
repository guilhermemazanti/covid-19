# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:03:36 2020

@author: Guilherme Mazanti
"""
from Compare import CompareCSV, CompareGraphs
from DailyData import DailyData

from datetime import date

import matplotlib.pyplot as plt

#TODAY = date.today()
TODAY = date(2020, 3, 27)

# =============================================================================
# Main
# =============================================================================

if __name__=="__main__":
  plt.close("all")
  dd = DailyData(TODAY)
  
  dd.bestFitAll()
  dd.saveParams()

  #countries = ["CN", "KR", "FR", "DE", "IT", "UK", "AT", "US", "BR", "AR", "AU"]
  countries = ["CN", "FR", "BR", "US", "IT", "KR"]
  dd.plotter.totalCasesByDate(countries, logScale = True)
  dd.plotter.totalCasesByDayN(countries, nbCases = 100, logScale = True)
  dd.plotter.bestFitByDayN(countries, nbCases = 100, nbDays = 50,\
                           logScale = True)
  dd.plotter.totalCasesAndBestFit("BR", future=7, logScale = False)
  dd.plotter.totalCasesAndBestFit("FR", future=7, logScale = False)
  dd.plotter.totalCasesAndBestFit("US", future=7, logScale = False)
  dd.plotter.totalCasesAndBestFit("CN", future=7, logScale = False)
  dd.plotter.totalCasesAndBestFit("IT", future=7, logScale = False)
  dd.plotter.totalCasesAndBestFit("KR", future=7, logScale = False)

  c = CompareCSV(date(2020, 3, 20))
  c.errorHistogram(1)
  c.errorHistogram(3)
  c.errorHistogram(7)

  cp = CompareGraphs(date(2020, 3, 20), date(2020, 3, 27))
  cp.plotCountry("BR")
  cp.plotCountry("FR")
  cp.plotCountry("US")
  cp.plotCountry("CN")
  cp.plotCountry("IT")
  cp.plotCountry("KR")
  
#  cp.plotCountry("AT")
#  cp.plotCountry("CA")
  
#  cp.plotCountry("LU")
#  cp.plotCountry("PE")
#  cp.plotCountry("IE")
  
#  cp.plotCountry("TH")
#  cp.plotCountry("RO")
#  cp.plotCountry("ID")