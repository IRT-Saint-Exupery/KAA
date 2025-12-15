#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================================

import os, time, json, datetime
import numpy as np
import pickle

from kaasrc.communs import colPrint
import kaasrc.communs
import kaasrc.controles

# ---------
# To control the library version
try:
    import shap
except Exception as err:
    print("Error:", err)
    colPrint("The library 'Shap' is not or not properly installed.", "Error")
# ---------


# ---------------------------------------------------------------------------
## Create an instance of the explainability method
# @param pDictParams : parameter dictionary
# @param pIndex : in case of PartitionTabular method: index table to explain
# @return method instance
def getExplainer(pDictParams, pIndex=None):
    # shortcuts
    pMethod = pDictParams['method']
    pMethodParam = pDictParams[pMethod]
    pClasses = pDictParams['classes']

    _, aiModel = pDictParams['aiModel']
    xData, _ = pDictParams['xy']

    # Partition - for tabular
    if pMethod == "PartitionTabular":
        ms = int(pMethodParam["ms"])
        cl = pMethodParam["cl"]
        if cl == "independent":
            masker = shap.maskers._tabular.Independent(xData[pIndex], max_samples=ms)
        else:
            masker = shap.maskers._tabular.Partition(xData[pIndex], max_samples=ms, clustering=cl)
        explainer = shap.explainers.Partition(aiModel.useCase, masker, output_names=pClasses, feature_names=aiModel.useCase.features, silent=True)
    # Partition - for texts
    elif pMethod == "PartitionText":
        if not hasattr(aiModel.useCase, 'predict_from_textlist'):
            colPrint("/!\\ The model must implement a method 'predict_proba()'. No calculated metric.", "Error")
            return None
        masker = shap.maskers.Text()
        masker.mask_token = ''
        explainer = shap.explainers.Partition(aiModel.useCase.predict_from_textlist, masker, output_names=pClasses)
    # Partition - for images
    elif pMethod == "PartitionImage":
        mv = pMethodParam["mv"]
        if mv == "blur":
            kx = int(pMethodParam["kx"])
            ky = int(pMethodParam["ky"])
            mv += "(%d, %d)"%(kx, ky)
        shape = xData[0].shape
        masker = shap.maskers.Image(mv, shape)

        explainer = shap.explainers.Partition(aiModel.useCase, masker, output_names=pClasses, feature_names=aiModel.useCase.features, silent=True)
    else:
        print("The method '%s' is not defined."%pMethod)
        explainer = None
    return explainer

# ---------------------------------------------------------------------------
## Compute explanations for tabular models
# @param pDictParams : parameter dictionary
def computeExplanationsTables(pDictParams):
    # shortcuts
    pMethod = pDictParams['method']
    pMethodParam = pDictParams[pMethod]
    pNbData = pDictParams['nbData']
    pRepertProd = pDictParams['repertProd']
    pDataProd = pDictParams['dataProd']

    resultSavedOn = os.path.join(pDataProd, "dataExplanations", pRepertProd)
    _, aiModel = pDictParams['aiModel']
    xData, yPred = pDictParams['xy']

    inputXAIframework = pDictParams['inputXAIframework']
    outputXAIFramework = pDictParams['outputXAIFramework']

    # convert data and prediction to XAI framework
    xData, yPred = kaasrc.communs.convertToXAIframework(pDictParams, inputXAIframework)

    # control function for XAI_computeExplanations inputs
    pParamSpecif = {"dataSize": pDictParams['datasize']}
    kaasrc.controles.controlXAI_computeExplanationsInput(pDictParams, inputXAIframework, xData, yPred, aiModel.useCase, pNbData, __file__, pParamSpecif)

    aiModel.setInferenceCible(False)
    max_evals = int(pMethodParam["me"])

    print("==[%s] ============================================================================="%pMethod, flush=True)
    start = time.time()
    explanations = []
    for index in range(pNbData):
        print(" - Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)
        table = xData[index]
        print("   .get explainer %s"%pMethod, flush=True)
        explainer = getExplainer(pDictParams, index)
        explanationTable = explainer(table, max_evals=max_evals)
        explanations.append(explanationTable)
    end = time.time()
    print("Duration:", end - start)

    # control function for XAI_computeExplanations outputs
    pParamSpecif = {"dataSize": pDictParams['datasize'], "nbClasses": aiModel.useCase.numClasses}
    kaasrc.controles.controlXAI_computeExplanationsOutput(pDictParams, outputXAIFramework, explanations, aiModel.useCase, pNbData, __file__, pParamSpecif)

    # Sauvegarde : Boucle sur les données
    for index in range(pNbData):
        _saveExplanations(pDictParams, explanations[index], resultSavedOn, index)

# ---------------------------------------------------------------------------
## Compute explanations for text models
# @param pDictParams : parameter dictionary
def computeExplanationsTexts(pDictParams):
    # shortcuts
    pMethod = pDictParams['method']
    pMethodParam = pDictParams[pMethod]
    pNbData = pDictParams['nbData']
    pRepertProd = pDictParams['repertProd']
    pDataProd = pDictParams['dataProd']

    resultSavedOn = os.path.join(pDataProd, "dataExplanations", pRepertProd)
    _, aiModel = pDictParams['aiModel']
    xData, _ = pDictParams['xy']

    outputXAIFramework = pDictParams['outputXAIFramework']

    kaasrc.controles.NOcontrolXAI_computeExplanationsInput(pDictParams)

    aiModel.setInferenceCible(False)
    max_evals = int(pMethodParam["me"])

    print("==[%s] ============================================================================="%pMethod, flush=True)
    start = time.time()
    explanations = []
    for index in range(pNbData):
        print(" - Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)
        print("   .get explainer %s"%pMethod, flush=True)
        explainer = getExplainer(pDictParams, index)

        explanationText = explainer(xData[index], max_evals=max_evals)
        explanations.append(explanationText)
    end = time.time()
    print("Duration:", end - start)

    # control function for XAI_computeExplanations outputs
    kaasrc.controles.controlXAI_computeExplanationsOutput(pDictParams, outputXAIFramework, explanations, pDictParams['datasize'], aiModel.useCase, pNbData, __file__)

    # Sauvegarde : Boucle sur les données
    for index in range(pNbData):
        _saveExplanations(pDictParams, explanations[index], resultSavedOn, index, pMinMax=False)

    # save all explanations to use them in launchMetric() function
    _saveExplanations(pDictParams, explanations, resultSavedOn, pMinMax=False)

# ---------------------------------------------------------------------------
## Compute explanations for image models
# @param pDictParams : parameter dictionary
def computeExplanationsImages(pDictParams):
    # shortcuts
    pMethod = pDictParams['method']
    pMethodParam = pDictParams[pMethod]
    pNbData = pDictParams['nbData']
    pRepertProd = pDictParams['repertProd']
    pDataProd = pDictParams['dataProd']
    pClasses = pDictParams['classes']
    xData, yPred = pDictParams['xy']
    _, aiModel = pDictParams['aiModel']

    inputXAIframework = pDictParams['inputXAIframework']
    outputXAIFramework = pDictParams['outputXAIFramework']

    resultSavedOn = os.path.join(pDataProd, "dataExplanations", pRepertProd)

    # convert data and prediction to XAI framework
    xData, yPred = kaasrc.communs.convertToXAIframework(pDictParams, inputXAIframework)
    pDictParams['xy'] = xData, yPred

    # control function for XAI_computeExplanations inputs
    kaasrc.controles.controlXAI_computeExplanationsInput(pDictParams, inputXAIframework, xData, yPred, aiModel.useCase, pNbData, __file__)

    aiModel.setInferenceCible(False)
    max_evals = int(pMethodParam["me"])

    print("   .get explainer %s"%pMethod, flush=True)
    explainer = getExplainer(pDictParams)

    print("==[%s] ============================================================================="%pMethod, flush=True)
    print(" - Explanation all %d data (%s)"%(pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)
    start = time.time()
    explanations = explainer(xData, max_evals=max_evals, outputs=shap.Explanation.argsort.flip[:len(pClasses)])
    end = time.time()
    print("Duration:", end - start)

    # control function for XAI_computeExplanations outputs
    pParamSpecif = {"nbClasses": aiModel.useCase.numClasses}
    kaasrc.controles.controlXAI_computeExplanationsOutput(pDictParams, outputXAIFramework, explanations, aiModel.useCase, pNbData, __file__, pParamSpecif)

    # Sauvegarde : Boucle sur les données
    for index in range(pNbData):
        _saveExplanations(pDictParams, explanations[index], resultSavedOn, index)

    # save all explanations to use them in launchMetric() function
    _saveExplanations(pDictParams, explanations, resultSavedOn)

# ---------------------------------------------------------------------------
## Save explanation on file
# @param pDictParams : parameter dictionary
# @param pExplanation : explanation
# @param pResultSavedOn : path to save on
# @param pIndex : index of the treated data
# @param pExtension : extension for filename to identify explanation
# @param pMinMax : min/max vales to save (unused)
def _saveExplanations(pDictParams, pExplanation, pResultSavedOn, pIndex=None, pExtension="", pMinMax=True):
    # shortcuts
    pDataList = pDictParams['dataList']

    if pMinMax:
        expMin = np.min(pExplanation.values)
        expMax = np.max(pExplanation.values)

    if pIndex is not None:
        ficData = os.path.basename(pDataList[pIndex])
        dirName = os.path.dirname(pDataList[pIndex])
        ficExplanation = kaasrc.communs.createDirName(pResultSavedOn, os.path.join(dirName, '%s%s.json'%(ficData, pExtension)))
        if pMinMax:
            minMaxJson = {"min": str(expMin), "max": str(expMax)}
            with open(ficExplanation, "w", encoding="utf-8") as f:
                json.dump(minMaxJson, f, indent=4)
        ficExplanation = ficExplanation.replace('.json', '.pkl')
    else:
        ficExplanation = kaasrc.communs.noSpace(pResultSavedOn, 'allData%s.pkl'%pExtension)

    with open(ficExplanation, 'wb') as fid:
        pickle.dump(pExplanation, fid)
    print("    Saved in '%s'"%ficExplanation, flush=True)

# ---------------------------------------------------------------------------
## Load explanation from file
# @param pDictParams : parameter dictionary
# @param pFicExplanation : explanation filename to load
# @param pNumData : number of data in explanations to load (to control)
# @return explanations loaded
def loadExplanations(pDictParams, pFicExplanation, pNumData=None):
    # shortcuts
    _, aiModel = pDictParams['aiModel']
    outputXAIFramework = pDictParams['outputXAIFramework']

    if not os.path.exists(pFicExplanation):
        colPrint("The file containing the explanation '%s' does not exists."%pFicExplanation, "Error")
        return None
    with open(pFicExplanation, 'rb') as fid:
        explanations = pickle.load(fid)

    if pDictParams['datatype'] == "tabular":
        # Les explications sont au format (n, 1)
        pParamSpecif = {"dataSize": pDictParams['datasize'][pNumData], "nbClasses": aiModel.useCase.numClasses}
        kaasrc.controles.control_loadExplanationsOutput(pDictParams, outputXAIFramework, explanations, aiModel.useCase, __file__, pParamSpecif)
    else:
        # Les explications sont au format (H, W, B, C)
        pParamSpecif = {"explOutputDim": aiModel.useCase.numDataChannels, "nbClasses": aiModel.useCase.numClasses}
        kaasrc.controles.control_loadExplanationsOutput(pDictParams, outputXAIFramework, explanations, aiModel.useCase, __file__, pParamSpecif)

    return explanations

# ===============================================================================
# end of file
