#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================================

import os, json

import kaasrc.plugin_collection
import kaasrc.communs
import kaasrc.controles
import numpy as np

# -------------------------------------------------------------------------------
## Plot the metrics and save files
# @param pMetric : metric used for measurement
# @param pMethod : explainability method
# @param pResults : metric results
# @param pFichProd : pathname prefix to save graphic
# @param pColor : colormap to use
# @param pNomData : data name to plot
# @param pAnnotations : flag to to indicate if annortations are plotted
# @param pModeHisto : flag to indicate the plot mode : histogram or dots
def _metricPlotScores(pMetric, pMethod, pResults, pFichProd, pColor, pNomData=None, pAnnotations=True, pModeHisto=False):
    # Graphique
    parametres = kaasrc.graphMatplot.GRAPH_parametresDefaut(pColor)
    parametres["rotation"] = 10

    valeursX = [ei for ei, i in enumerate(pResults) if i != "mean"]
    tics = [i for i in pResults if i != "mean"]
    scores = [round(pResults[i], 3) for i in pResults if i != "mean"]

    grX = {'libelle': pMethod, 'valeurs': valeursX, 'tics': tics}
    grY = [{'couleur': '#8c6d31', 'libelle': "Scores", 'trace': 'p', 'axe': 'y1', 'valeurs': scores}]
    if pAnnotations:
        grY[0]['annotations'] = (-0.05, 0.01, 0, ["%3.3f"%x for x in scores])
    fichierGraphique = "%s--m_%s"%(pFichProd, pMethod)
    titre = pMetric
    if pNomData is not None:
        titre += " on data #%s"%pNomData
    if pModeHisto:
        grY[0]['trace'] = 'l'
        kaasrc.graphMatplot.GRAPH_matplotHisto(fichierGraphique, titre, grX, grY, parametres)
    else:
        kaasrc.graphMatplot.GRAPH_matplot2D(fichierGraphique, titre, grX, grY, None, parametres)
    print("       saved in: %s.png"%fichierGraphique, flush=True)

# ---------------------------------------------------------------------------
## Plot the metrics and save files
# @param pMetric : metric used for measurement
# @param pMethod : explainability method
# @param pResults : metric results
# @param pFichProd : pathname prefix to save graphic
# @param pColor : color map
# @param pNomData : name of the data to treat
def _metricPlotScoresSklearn(pMetric, pMethod, pResults, pFichProd, pColor, pNomData=None):
    # Graphique
    parametres = kaasrc.graphMatplot.GRAPH_parametresDefaut(pColor, 4)
    parametres["rotation"] = 10
    parametres["decalage"] = 0.2
    parametres["largbar"] = 0.2
    parametres["legende"] = "HG"

    valeursX = list(pResults.keys())
    valeursXaxis = np.arange(len(valeursX))
    scores_acc = [round(val["accuracy"], 3) if val != {} else 0 for val in pResults.values()]
    scores_pre = [round(val["precision"], 3) if val != {} else 0 for val in pResults.values()]
    scores_rec = [round(val["recall"], 3) if val != {} else 0 for val in pResults.values()]
    scores_f1 = [round(val["f1"], 3) if val != {} else 0 for val in pResults.values()]

    grX = {'libelle': pMethod, 'valeurs': valeursXaxis, 'tics': list(pResults.keys())}
    grY = [{'libelle': "Scores", 'trace': 'l', 'axe': 'y1', 'valeurs': scores_acc, 'label': "accuracy", 'annotations': (-0.05, 0.01, 0, ["%3.3f"%x for x in scores_acc])},
           {'libelle': "Scores", 'trace': 'l', 'axe': 'y1', 'valeurs': scores_pre, 'label': "precision", 'annotations': (-0.05, 0.01, 0, ["%3.3f"%x for x in scores_pre])},
           {'libelle': "Scores", 'trace': 'l', 'axe': 'y1', 'valeurs': scores_rec, 'label': "precision", 'annotations': (-0.05, 0.01, 0, ["%3.3f"%x for x in scores_rec])},
           {'libelle': "Scores", 'trace': 'l', 'axe': 'y1', 'valeurs': scores_f1, 'label': "precision", 'annotations': (-0.05, 0.01, 0, ["%3.3f"%x for x in scores_f1])}]
    titre = pMetric
    if pNomData is not None:
        titre += " on data %s"%pNomData
    kaasrc.graphMatplot.GRAPH_matplotHisto("%s--m_%s"%(pFichProd, pMethod), titre, grX, grY, parametres)
    print("       saved in: %s.png"%pFichProd, flush=True)

# ---------------------------------------------------------------------------
## Plot metric results
# @param pDictParams : parameter dictionary
# @param pMetric : metric name
# @param pMethods4metric : explainability method list
# @param pResults : metric results
# @param pFichProd : filename to save metric results
# @param pColor : colormap to use
# @param pIndex : index of data to treat
def _metricPlot(pDictParams, pMetric, pMethods4metric, pResults, pFichProd, pColor="hot", pIndex=None):
    # shortcuts
    pDataList = pDictParams['dataList']

    if pMetric == "Sklearn-Scores":
        for method in pMethods4metric:
            if pIndex is not None:
                _metricPlotScoresSklearn(pMetric, method, pResults[method], pFichProd, pColor, pDataList[pIndex])
            else:
                _metricPlotScoresSklearn(pMetric, method, pResults[method], pFichProd, pColor)
    else:
        for method in pMethods4metric:
            _metricPlotScores(pMetric, method, pResults[method], pFichProd, pColor, pDataList[pIndex], pAnnotations=pMetric != "Monotonicity")

# ---------------------------------------------------------------------------
## Plot metric results
# @param pDictParams : parameter dictionary
# @param pIndex : index of data to treat
# @param pExtenData : suffix to add to the file name of result
def plotMetrics(pDictParams, pIndex=None, pExtenData=""):
    # shortcuts
    pDataProd = pDictParams['dataProd']
    pMetrique = pDictParams['metric']
    pSuffixMetric = pDictParams["suffixMetric"]
    pMethods4metric = pDictParams['methods4metric'].split(',')

    # get the color map
    pCmapMode = pDictParams['cmapListMode']
    if pCmapMode == 1:
        with open(pDictParams['cmapListColor'], "r", encoding="utf8") as file:
            data = json.load(file)
        key = list(data.keys())[0]
        pColor = data[key]
    else:
        pColor = pDictParams["cmap"]

    pMethodsWithSuffix4metric = []
    for methode in pMethods4metric:
        methodWithSuffix = "%s_%s"%(methode, pDictParams[methode]['suffixMethod'])
        pMethodsWithSuffix4metric.append(methodWithSuffix)

    fichierJson = os.path.join(pDataProd, "dataMetrics", pDictParams['code'], '%s_%s%s.json'%(pMetrique, pSuffixMetric, pExtenData))
    print("Metric results loaded from '%s'"%fichierJson)
    if not os.path.exists(fichierJson):
        print("ERROR: file '%s' does not exist !"%fichierJson)
        return
    with open(fichierJson, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Plot metrics
    fichProd = os.path.join(pDataProd, "dataPlotMetrics", pDictParams['code'], '%s_%s%s'%(pMetrique, pSuffixMetric, pExtenData))
    _metricPlot(pDictParams, pMetrique, pMethodsWithSuffix4metric, results, fichProd, pColor, pIndex)

# ---------------------------------------------------------------------------
## Plot metric results
# @param pDictParams : parameter dictionary
def plotMetricsAllData(pDictParams):
    # shortcuts
    pDataProd = pDictParams['dataProd']
    pMetrique = pDictParams['metric']
    pSuffixMetric = pDictParams["suffixMetric"]
    pMethods4metric = pDictParams['methods4metric'].split(',')
    pNbData = pDictParams['nbData']
    pDataList = pDictParams['dataList']

    # get the color map
    pCmapMode = pDictParams['cmapListMode']
    if pCmapMode == 1:
        with open(pDictParams['cmapListColor'], "r", encoding="utf-8") as file:
            data = json.load(file)
        key = list(data.keys())[0]
        pColor = data[key]
    else:
        pColor = pDictParams["cmap"]

    fichierJson = os.path.join(pDataProd, "dataMetrics", pDictParams['code'], '%s_%s.json'%(pMetrique, pSuffixMetric))
    print("Metric results loaded from '%s'"%fichierJson)
    if not os.path.exists(fichierJson):
        print("ERROR: file '%s' does not exist !"%fichierJson)
        return
    with open(fichierJson, "r", encoding="utf-8") as f:
        results = json.load(f)

    for methode in pMethods4metric:
        methodWithSuffix = "%s_%s"%(methode, pDictParams[methode]['suffixMethod'])
        resultsToPlot = {}
        for index in range(pNbData):
            nomData = pDataList[index].replace(' ', '')
            resultsToPlot[nomData] = results[methodWithSuffix][nomData]

        # Plot metrics
        fichProd = os.path.join(pDataProd, "dataPlotMetrics", pDictParams['code'], '%s_%s'%(pMetrique, pSuffixMetric))
        _metricPlotScores(pMetrique, methodWithSuffix, resultsToPlot, fichProd, pColor, pAnnotations=True, pModeHisto=True)

# ---------------------------------------------------------------------------
## Plot metric results
# @param pDictParams : parameter dictionary
def plotMetricsTabular(pDictParams):
    if pDictParams['methods4metric'] == "":
        print("No method selected!")
        return
    # shortcuts
    pNbData = pDictParams['nbData']
    pDataList = pDictParams['dataList']
    pMetrique = pDictParams['metric']

    for index in range(pNbData):
        extenData = "--d_%s"%pDataList[index].replace(' ', '')
        plotMetrics(pDictParams, index, extenData)
    if pMetrique != "Sklearn-Scores":
        plotMetricsAllData(pDictParams)

# ---------------------------------------------------------------------------
## Plot metric results
# @param pDictParams : parameter dictionary
def plotMetricsImage(pDictParams):
    if pDictParams['methods4metric'] == "":
        print("No method selected!")
        return
    plotMetricsAllData(pDictParams)

# ===============================================================================
# end of file
