# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:03:36 2020

@author: Guilherme Mazanti
"""

from datetime import date, timedelta
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests as rq
import scipy.optimize as opt
import warnings as w

#TODAY = date.today()
TODAY = date(2020, 3, 20)

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
# Data extraction
# =============================================================================

def getDataFromCountry(db, countryCode):
  """
  Extrai dados de um país do dataframe db e retorna os resultados como arrays
  numpy.
  
  Entradas:
    db: base de dados pandas no formato fornecido pelo ECDC.
    countryCode: string contendo o código do país.
    
  Saídas:
    dates: array 1D numpy com as datas em ordem crescente. As datas são
      necessariamente contíguas. A primeira data corresponde necessariamente ao
      primeiro caso reportado, a última data corresponde ao dia atual. Essas
      especificações não se aplicam caso nenhum caso tenha sido reportado para
      o país 
    cases: array 1D numpy de mesma shape que dates tal que cases[i] contém a
      quantidade de casos reportados no dia dates[i].
    deaths: idem cases, mas quantidade de mortes reportadas.
    
  Exceções:
    Exception: caso countryCode não apareça na base de dados db.
    
  Warnings:
    UserWarning: caso não haja casos reportados para o país.
  """
  dbCountry = db[db.GeoId==countryCode]
  if dbCountry.empty:
    raise Exception("No data for country with code "+countryCode)
  
  # Extraction des données
  countryName = dbCountry["Countries and territories"].values[0]
  dates = zip(dbCountry.Day.values, dbCountry.Month.values,\
              dbCountry.Year.values)
  dates = np.array([date(y, m, d) for d, m, y in dates])
  cases = dbCountry.Cases.values
  deaths = dbCountry.Deaths.values
  
  # Tri par date croissante
  ind = dates.argsort()
  
  dates = dates[ind]
  cases = cases[ind]
  deaths = deaths[ind]
    
  # Suppression des dates jusqu'au premier cas
  firstCase = np.flatnonzero(cases)
  firstCase = firstCase[0] if firstCase.size > 0 else None
  
  if firstCase is not None:
    dates = dates[firstCase:]
    cases = cases[firstCase:]
    deaths = deaths[firstCase:]
  else:
    w.warn("No cases reported in "+countryName, stacklevel=2)
  
  # On garantit que tous les jours sont présents dès le début
  N = (TODAY - dates[0]).days + 1
  newDates = np.empty(N, dtype=object)
  newCases = np.zeros(N, dtype=int)
  newDeaths = np.zeros(N, dtype=int)
  
  newDates[0] = dates[0]
  newCases[0] = cases[0]
  newDeaths[0] = deaths[0]
  j = 1
  for i in range(1, N):
    newDates[i] = newDates[i-1] + timedelta(1)
    if j < dates.size and dates[j] == newDates[i]:
      newCases[i] = cases[j]
      newDeaths[i] = deaths[j]
      j += 1
  
  return newDates, newCases, newDeaths

# =============================================================================
# Estimation
# =============================================================================
def findBestTanh(db, countryCode):
  """
  Fits a function of the form
  f(t) = A * [tahn(a*(t - tau)) + 1] / 2
  finding the best parameters A, a, tau.
  """
  _, cases, _ = getDataFromCountry(db, countryCode)
#  model = lambda t, A, a, tau: A * (np.tanh(a*(t - tau)) + 1) / 2
#  jac = lambda t, A, a, tau: np.array([np.tanh(a*(t - tau)) + 1,\
#    A*(t - tau)/(np.cosh(a*(t - tau))**2),\
#    -A*a/(np.cosh(a*(t - tau))**2)]).transpose() / 2
  (A, a, tau), _ = opt.curve_fit(model, np.arange(cases.size), cases.cumsum(),\
    p0 = [cases.sum(), 0.15, 50], check_finite=True, jac = jac, maxfev = 10000)
  return A, a, tau

def estimParams(db):
  listCountryCodes = set(db.GeoId.values)
  params = dict()
  for code in listCountryCodes:
    dates, cases, deaths = getDataFromCountry(db, code)
    if cases.sum() >= 100 and dates.size >= 10:
      countryName = db[db.GeoId==code]["Countries and territories"].values[0]
      A, a, tau = findBestTanh(db, code)
      params[code] = (countryName, dates[0], A, a, tau)
  return params

def saveParams(params, filename):
  with open(filename, "w") as file:
    file.write("Country code,Country name,Estimation of total number of cases,"
               "Estimate of number of cases tomorrow,"
               "Estimate of number of cases in 3 days,"
               "Estimate of number of cases in 7 days,"
               "Number of days for doubling,Number of days until inflection,"
               "Inflection date\n")
    for code in params:
      countryName, date0, A, a, tau = params[code]
      nbDouble = np.log(2)/a
      time1Days = (TODAY - date0).days + 1
      cases1Days = model(time1Days, A, a, tau)
      time3Days = (TODAY - date0).days + 3
      cases3Days = model(time3Days, A, a, tau)
      time7Days = (TODAY - date0).days + 7
      cases7Days = model(time7Days, A, a, tau) #A * (np.tanh(a*(time7Days - tau)) + 1) / 2
      file.write("{:s},{:s},{:d},{:d},{:d},{:d},{:.2f},{:d},{:s}\n".format(\
                 code, countryName,\
                 int(A), int(cases1Days), int(cases3Days), int(cases7Days),\
                 nbDouble, int(tau+0.5),\
                 (date0+timedelta(int(tau+0.5))).isoformat()))

# =============================================================================
# Plots
# =============================================================================

def plotTotalCasesByDate(db, listCountryCodes, logScale = False):
  """
  """
  fig = plt.figure()
  ax = fig.add_axes([0.1, 0.22, 0.85, 0.68])
  ax.grid(True)
  ax.set_axisbelow(True)
  for countryCode in listCountryCodes:
    dates, cases, _ = getDataFromCountry(db, countryCode)
    ax.plot(dates, cases.cumsum(), label=countryCode)
  for text in ax.get_xticklabels():
    text.set_rotation(45)
  ax.set_xlabel("Date")
  ax.set_ylabel("Total number of cases")
  ax.set_title("Total number of cases by date")
  ax.legend()
  if logScale:
    ax.set_yscale("log")
    
def plotTotalCasesByFirstDay(db, listCountryCodes, logScale = False):
  """
  """
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.grid(True)
  ax.set_axisbelow(True)
  for countryCode in listCountryCodes:
    _, cases, _ = getDataFromCountry(db, countryCode)
    ax.plot(cases.cumsum(), label=countryCode)
  ax.set_xlabel("Days since first case")
  ax.set_ylabel("Total number of cases")
  ax.set_title("Total number of cases by days since first case")
  ax.legend()
  if logScale:
    ax.set_yscale("log")


def plotBestTanhByFirstDay(db, listCountryCodes, nbDays = 100, logScale = False):
  """
  """
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.grid(True)
  ax.set_axisbelow(True)
  for countryCode in listCountryCodes:
    #try:
      A, a, tau = findBestTanh(db, countryCode)
      t = np.arange(nbDays)
      totCases = model(t, A, a, tau) # A * (np.tanh(a*(t - tau)) + 1) / 2
      ax.plot(totCases, label=countryCode)
    #except RuntimeError:
    #  w.warn("Could not find optimal curve for "+countryCode)
  ax.set_xlabel("Days since first case")
  ax.set_ylabel("Total number of cases")
  #ax.set_ylim([0, 100000])
  ax.set_title("Estimate of the total number of cases by days since first case")
  ax.legend()
  if logScale:
    ax.set_yscale("log")
    
def plotTotalCasesAndBestTanh(db, code, future = 0, logScale = False):
  """
  """
  countryName = db[db.GeoId==code]["Countries and territories"].values[0]
  
  fig = plt.figure()
  ax = fig.add_axes([0.12, 0.22, 0.85, 0.68])
  ax.grid(True)
  ax.set_axisbelow(True)
  dates, cases, _ = getDataFromCountry(db, code)
  A, a, tau = findBestTanh(db, code)
  nbDays = (TODAY - dates[0]).days + 1 + future
  t = np.arange(nbDays)
  totCases = model(t, A, a, tau) # A * (np.tanh(a*(t - tau)) + 1) / 2
  ax.plot(dates, cases.cumsum(), label="Observed", lw=2.75)
  ax.plot([dates[0] + timedelta(int(ti)) for ti in t], totCases,\
           label="Estimation", lw=1.25)
  for text in ax.get_xticklabels():
    text.set_rotation(45)
  ax.set_xlabel("Date")
  ax.set_ylabel("Total number of cases")
  ax.set_title("Total number of cases by date for "+countryName)
  ax.legend()
  if logScale:
    ax.set_yscale("log")

# =============================================================================
# Main
# =============================================================================

if __name__=="__main__":
  plt.close("all")
  filename = TODAY.isoformat()
  if not os.path.exists(filename + ".xlsx"):
    webData = rq.get("https://www.ecdc.europa.eu/sites/default/files/"
                     "documents/COVID-19-geographic-disbtribution-worldwide-"+
                     filename + ".xlsx")
    with open(filename + ".xlsx", "wb") as file:
      file.write(webData.content)
      
  db = pd.read_excel(filename + ".xlsx", keep_default_na = False)
  
  #countries = ["CN", "KR", "FR", "DE", "IT", "UK", "AT", "US", "BR", "AR", "AU"]
  countries = ["CN", "FR", "BR", "US", "IT"]
  plotTotalCasesByDate(db, countries, logScale = True)
  plotTotalCasesByFirstDay(db, countries, logScale = True)
  plotBestTanhByFirstDay(db, countries, logScale = True)
  plotTotalCasesAndBestTanh(db, "BR", future=7, logScale = False)
  plotTotalCasesAndBestTanh(db, "FR", future=7, logScale = False)
  plotTotalCasesAndBestTanh(db, "US", future=7, logScale = False)
  plotTotalCasesAndBestTanh(db, "CN", future=7, logScale = False)
  plotTotalCasesAndBestTanh(db, "IT", future=7, logScale = False)
  params = estimParams(db)
  saveParams(params, filename + ".csv")