#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================================

import os, json

from kaasrc.communs import colPrint
import kaasrc.graphMatplot

# -------------------------------------------------------------------------------
## Function that check if a class has a member function
# param pClass : the class to inspect
# param pFunction : the function to check
# def has_method(pClass, pFunction):
#    return callable(getattr(pClass, pFunction, None))

# -------------------------------------------------------------------------------
## Plot the metrics and save files
# @param pMetric : metric used for measurement
# @param pResults : metric results
# @param pFichProd : pathname prefix to save graphic
# @param pNomData : name of the data
def _metricPlotScores(pMetric, pResults, pFichProd, pNomData=None):
    if "allData" in pResults:
        pResults = pResults["allData"]
    methodsWithSuffixes4metric = pResults.keys()
    extensGraph = ""
    for methodWithSuffix in methodsWithSuffixes4metric:
        extensGraph += methodWithSuffix[0]

    # Graphique
    parametres = kaasrc.graphMatplot.GRAPH_parametresDefaut()
    parametres["rotation"] = 10

    valeursX = list(range(len(methodsWithSuffixes4metric)))
    scores = [round(pResults[methodWithSuffix], 3) for methodWithSuffix in methodsWithSuffixes4metric]
    grX = {'libelle': 'Methods', 'valeurs': valeursX, 'tics': [m.replace('__', '\n') for m in methodsWithSuffixes4metric]}
    grY = [{'couleur': '#8c6d31', 'libelle': "Scores", 'trace': 'l', 'axe': 'y1', 'valeurs': scores, 'annotations': (-0.05, 0.01, 0, ["%3.3f"%x for x in scores])}]

    titre = pMetric
    if pNomData is not None:
        titre += " on data %s"%pNomData
    fichierGraphique = "%s--g_%s"%(pFichProd, extensGraph)
    kaasrc.graphMatplot.GRAPH_matplotHisto(fichierGraphique, titre  , grX, grY, parametres)
    print("       saved in: %s.png"%fichierGraphique, flush=True)

# -------------------------------------------------------------------------------
## Plot the metrics and save files
# @param pMetric : metric name
# @param pResults : metric results
# @param pResultsMean : mean values of individual results
# @param pFichProd : pathname prefix to save graphic
def _metricPlotCausalFidelityAUC(pMetric, pResults, pResultsMean, pFichProd):
    methodsWithSuffixes4metric = pResults.keys()
    # Graphique
    parametres = kaasrc.graphMatplot.GRAPH_parametresDefaut()
    parametres["remplissage"] = 1
    parametres["rotation"] = 10

    for methodWithSuffix in methodsWithSuffixes4metric:
        scores = []
        valeursX = []
        if "allData" in pResultsMean[methodWithSuffix]:
            xscores = pResultsMean[methodWithSuffix]["allData"]["values"]
        else:
            xscores = pResultsMean[methodWithSuffix]["values"]
        for x in xscores:
            scores.append(xscores[x])
            valeursX.append(x)

        titre = "%s - %s - AUC : %5.3f"%(pMetric, methodWithSuffix, pResults[methodWithSuffix])
        grX = {'libelle': 'Steps', 'valeurs': valeursX, 'tics': valeursX}
        grY = [{'couleur': '#8c6d31', 'libelle': "Score", 'trace': 'l', 'axe': 'y1', 'valeurs': scores}]
        fichierGraphique = "%s--m_%s"%(pFichProd, methodWithSuffix)
        kaasrc.graphMatplot.GRAPH_matplot2D(fichierGraphique, titre, grX, grY, None, parametres)
        print("       saved in: %s.png"%fichierGraphique, flush=True)

# ---------------------------------------------------------------------------
## Plot metric results
# @param pMetric : metric name
# @param pResults : metric results
# @param pResultsMean : mean values of individual results
# @param pFichProd : filename to save metric results
# @param pNomData : name of the data
def _metricPlot(pMetric, pResults, pResultsMean, pFichProd, pNomData=None):

    # CausalFidelity-Deletion
    # if pMetric in ["CausalFidelity-Deletion", "CausalFidelity-Insertion"]:
    if pResultsMean is not None:
        _metricPlotCausalFidelityAUC(pMetric, pResults, pResultsMean, pFichProd)
    # else:
    #    _metricPlotIndividualScores(pMetric, pResults, pFichProd)
    _metricPlotScores(pMetric, pResults, pFichProd, pNomData)

# ---------------------------------------------------------------------------
## Plot metric results
# @param pDictParams : parameter dictionary
# @param pIndex : index of data to treat
def _plotMetrics(pDictParams, pIndex=None):
    # shortcuts
    pDataProd = pDictParams['dataProd']
    pMetrique = pDictParams['metric']
    pDataList = pDictParams['dataList']
    pSuffixMetric = pDictParams["suffixMetric"]
    if pDictParams['methods4metric'] == "":
        print("No method selected!")
        return

    # préparation des noms de fichier de score et de tracé de courbe pour causal-fidelity
    fichierJson = os.path.join(pDataProd, "dataMetrics", pDictParams['code'], '%s_%s.json'%(pMetrique, pSuffixMetric))
    fichierJsonMean = os.path.join(pDataProd, "dataMetrics", pDictParams['code'], '%s_%s--s_AUCvalues.json'%(pMetrique, pSuffixMetric))

    # Ajustement du fichier de score en cas de donnée indexée
    if pIndex is not None:
        fichierJson = fichierJson.replace(".json", "--d_%s.json"%pDataList[pIndex].replace(' ', ''))

    # Lecture du fichier resultat
    print("Metric results loaded from '%s'"%fichierJson)
    if not os.path.exists(fichierJson):
        colPrint("File '%s' does not exist !"%fichierJson, "Error")
        return
    with open(fichierJson, "r", encoding="utf-8") as f:
        results = json.load(f)

    # retrait des clefs "allData" dans le cas de calcul global
    if pIndex is None:
        for key in results:
            results[key] = results[key]["allData"]

    # valeurs mean
    if os.path.exists(fichierJsonMean):
        with open(fichierJsonMean, "r", encoding="utf-8") as f:
            meanScoresByMethod = json.load(f)
    else:
        meanScoresByMethod = None
    # Plot metrics
    fichProd = os.path.join(pDataProd, "dataPlotMetrics", pDictParams['code'], '%s_%s'%(pMetrique, pSuffixMetric))
    if pIndex is not None:
        fichProd = "%s--d_%s"%(fichProd, pDataList[pIndex].replace(' ', ''))
        nomData = pDataList[pIndex]
    else:
        nomData = None

    _metricPlot(pMetrique, results, meanScoresByMethod, fichProd, pNomData=nomData)

# ---------------------------------------------------------------------------
## Plot metric results
# @param pDictParams : parameter dictionary
def plotMetricsImage(pDictParams):
    _plotMetrics(pDictParams)

# ---------------------------------------------------------------------------
## Plot metric results
# @param pDictParams : parameter dictionary
def plotMetricsTabText(pDictParams):
    # shortcuts
    pNbData = pDictParams['nbData']

    for index in range(pNbData):
        _plotMetrics(pDictParams, index)

# ===============================================================================
# end of file
