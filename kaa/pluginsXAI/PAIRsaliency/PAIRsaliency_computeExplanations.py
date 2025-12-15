#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================================

import os, time, json, datetime
import numpy as np

from kaasrc.communs import colPrint
import kaasrc.communs
import kaasrc.controles

# ---------
try:
    import tensorflow.compat.v2 as tf
except Exception as err:
    print("Error:", err)
    colPrint("Package tensorflow is not or not properly installed.", "Error")
# ---------
try:
    from saliency.core import GuidedIG, XRAI, GradientSaliency, BlurIG, Occlusion, IntegratedGradients, GradCam
except Exception as err:
    print("Error:", err)
    colPrint("The library 'PAIRsaliency' is not or not properly installed.", "Error")


# ---------------------------------------------------------------------------
## Create an instance of the explainability method
# @param pDictParams : parameter dictionary
# @param pXinputs : model inference inputs
# @param pYpred : model inference outputs
# @return method instance
def _explain(pDictParams, pXinputs, pYpred):
    # shortcuts
    pMethod = pDictParams['method']
    pMethodParam = pDictParams[pMethod]

    if 'callModelFunctionToGradient' in pDictParams:
        pModelFunction = pDictParams['callModelFunctionToGradient']
    else:
        print("/!\\ Le dictionnaire paramètres doit contenir l'entrée 'callModelFunctionToGradient' désignant la fonction d'accès aux couches du modèle :")
        print("    dictionnaire {'CONVOLUTION_LAYER_VALUES': ... , 'CONVOLUTION_OUTPUT_GRADIENTS': ... } pour la méthode GradCam")
        print("    dictionnaire {'OUTPUT_LAYER_VALUES': ... } pour la méthode Occlusion")
        print("    dictionnaire {'INPUT_OUTPUT_GRADIENTS': ... } pour les autres méthodes")
        return

    _, aiModel = pDictParams['aiModel']

    # Guided IG
    if pMethod == 'GuidedIG':
        x_steps = int(pMethodParam['rs'])
        x_baseline = float(pMethodParam['bl'])
        max_dist = float(pMethodParam['md'])
        fraction = float(pMethodParam['ff'])

        x_baseline = np.ones_like(pXinputs) * x_baseline
        explainer = GuidedIG()
        explanations = explainer.GetMask(pXinputs, pModelFunction, aiModel.useCase,
                        x_steps=x_steps, x_baseline=x_baseline, max_dist=max_dist, fraction=fraction)
    # SmoothGrad-Guided IG
    elif pMethod == 'SmoothGrad-GuidedIG':
        x_steps = int(pMethodParam['rs'])
        x_baseline = float(pMethodParam['bl'])
        max_dist = float(pMethodParam['md'])
        fraction = float(pMethodParam['ff'])
        stdev_spread = float(pMethodParam['sd'])
        nsamples = int(pMethodParam['ns'])
        magnitude = int(pMethodParam['mg'])

        x_baseline = np.ones_like(pXinputs) * x_baseline
        explainer = GuidedIG()
        explanations = explainer.GetSmoothedMask(pXinputs, pModelFunction, aiModel.useCase,
                        x_steps=x_steps, x_baseline=x_baseline, max_dist=max_dist, fraction=fraction,
                        stdev_spread=stdev_spread, nsamples=nsamples, magnitude=magnitude)
    # XRAI
    elif pMethod == 'XRAI':
        batch_size = int(pMethodParam['bz'])
        bm = float(pMethodParam['bm'])
        bx = float(pMethodParam['bx'])
        baselines = [np.zeros_like(pXinputs) + bm, np.zeros_like(pXinputs) + bx]
        explainer = XRAI()
        explanations = explainer.GetMask(pXinputs, pModelFunction, aiModel.useCase,
                        baselines=baselines, batch_size=batch_size)
    # VanillaGradients
    elif pMethod == 'VanillaGradients':
        explainer = GradientSaliency()
        explanations = explainer.GetMask(pXinputs, pModelFunction, aiModel.useCase)
    # SmoothGrad-VanillaGradients
    elif pMethod == 'SmoothGrad-VanillaGradients':
        stdev_spread = float(pMethodParam['sd'])
        nsamples = int(pMethodParam['ns'])
        magnitude = int(pMethodParam['mg'])
        explainer = GradientSaliency()
        explanations = explainer.GetSmoothedMask(pXinputs, pModelFunction, aiModel.useCase,
                        stdev_spread=stdev_spread, nsamples=nsamples, magnitude=magnitude)
    # IntegratedGradients
    elif pMethod == 'IntegratedGradients':
        bl = float(pMethodParam['bl'])
        baselines = np.zeros_like(pXinputs) + bl
        steps = int(pMethodParam['st'])
        explainer = IntegratedGradients()
        explanations = explainer.GetMask(pXinputs, pModelFunction, aiModel.useCase, x_baseline=baselines, x_steps=steps)
        explanations = explanations.numpy()
    # GradCAM
    elif pMethod == 'GradCAM':
        explainer = GradCam()
        three_dims = int(pMethodParam['bd'])
        explanations = explainer.GetMask(pXinputs, pModelFunction, aiModel.useCase, should_resize=True, three_dims=three_dims)
    # BlurIG
    elif pMethod == 'BlurIG':
        max_sigma = int(pMethodParam['ms'])
        steps = int(pMethodParam['st'])
        grad_step = float(pMethodParam['gs'])
        sqrt = pMethodParam['sq'] != "False"
        batch_size = int(pMethodParam['bz'])
        explainer = BlurIG()
        explanations = explainer.GetMask(pXinputs, pModelFunction, aiModel.useCase,
                      max_sigma=max_sigma, steps=steps, grad_step=grad_step, sqrt=sqrt, batch_size=batch_size)
    # SmoothGrad-BlurIG
    elif pMethod == 'SmoothGrad-BlurIG':
        stdev_spread = float(pMethodParam['sd'])
        nsamples = int(pMethodParam['ns'])
        magnitude = int(pMethodParam['mg'])
        explainer = BlurIG()
        explanations = explainer.GetSmoothedMask(pXinputs, pModelFunction, aiModel.useCase,
                        stdev_spread=stdev_spread, nsamples=nsamples, magnitude=magnitude)
    # Occlusion
    elif pMethod == 'Occlusion':
        size = int(pMethodParam['sz'])
        value = float(pMethodParam['vl'])
        stride = int(pMethodParam['st'])
        explainer = Occlusion()
        call_model_args = {'aiModel': aiModel.useCase, 'targets': pYpred}
        explanations = explainer.GetMask(pXinputs, pModelFunction, call_model_args,
                      size=size, value=value, stride=stride)
    else:
        print("La méthode '%s' n'est pas définie."%pMethod)
        explanations = None
    return explanations

# ---------------------------------------------------------------------------
## Compute explanations
# @param pDictParams : parameter dictionary
# @param pClasse : class to explain : extension for filename to identify explanation
# @return parameter dictionary
def _computeExplanations(pDictParams, pClasse=None):
    # shortcuts
    pMethod = pDictParams['method']
    pNbData = pDictParams['nbData']
    pRepertProd = pDictParams['repertProd']
    pDataProd = pDictParams['dataProd']
    xData, yPred = pDictParams['xy']
    _, aiModel = pDictParams['aiModel']

    inputXAIframework = pDictParams['inputXAIframework']
    outputXAIFramework = pDictParams['outputXAIFramework']

    resultSavedOn = os.path.join(pDataProd, "dataExplanations", pRepertProd)

    # convert data and prediction to XAI framework
    xData, yPred = kaasrc.communs.convertToXAIframework(pDictParams, inputXAIframework)
    pDictParams['xy'] = xData, yPred

    # control function for XAI_computeExplanations inputs
    if pDictParams['datatype'] == "tabular":
        pParamSpecif = {"dataSize": pDictParams['datasize']}
        kaasrc.controles.controlXAI_computeExplanationsInput(pDictParams, inputXAIframework, xData, yPred, aiModel.useCase, pNbData, __file__, pParamSpecif)
    else:
        kaasrc.controles.controlXAI_computeExplanationsInput(pDictParams, inputXAIframework, xData, yPred, aiModel.useCase, pNbData, __file__)

    aiModel.setInferenceCible(False)

    print("==[%s] ============================================================================="%pMethod, flush=True)
    start = time.time()
    explanations = []
    for index in range(pNbData):
        print(" - Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)
        explanation = _explain(pDictParams, xData[index], yPred[index])
        if pDictParams['datatype'] == "image":
            explanation = np.expand_dims(explanation, axis=0)
            explanation = np.expand_dims(explanation[:, :, :, 0], axis=-1)
        explanations.append(explanation)
    if pDictParams['datatype'] != "tabular":
        explanations = np.concatenate(explanations, axis=0)
    end = time.time()
    print("Duration:", end - start)

    # control function for XAI_computeExplanations outputs
    if pDictParams['datatype'] == "tabular":
        pParamSpecif = {"dataSize": pDictParams['datasize']}
        kaasrc.controles.controlXAI_computeExplanationsOutput(pDictParams, outputXAIFramework, explanations, aiModel.useCase, pNbData, __file__, pParamSpecif)
    else:
        pParamSpecif = {"explOutputDim": 1}
        kaasrc.controles.controlXAI_computeExplanationsOutput(pDictParams, outputXAIFramework, explanations, aiModel.useCase, pNbData, __file__, pParamSpecif)

    # Sauvegarde : Boucle sur les données
    extension = ""
    if pClasse is not None:
        extension = "--c_%s"%pClasse
    for index in range(pNbData):
        _saveExplanations(pDictParams, explanations[index], resultSavedOn, index, extension)

    # save all explanations to use them in launchMetric() function
    _saveExplanations(pDictParams, explanations, resultSavedOn, pExtension=extension)
    return pDictParams

# ---------------------------------------------------------------------------
## Compute explanations
# @param pDictParams : parameter dictionary
# @param pExplainClasses : Boolean to indicate if classes have to be treated
# @return parameter dictionary
def computeExplanationsTables(pDictParams, pExplainClasses=True):
    # shortcuts
    xData, _ = pDictParams['xy']
    pClasses = pDictParams['classes']
    pNbData = pDictParams['nbData']

    pDictParams = _computeExplanations(pDictParams)
    if pExplainClasses:
        for n, _ in enumerate(pClasses):
            print(" - Explanation class %d/%d (%s)"%(n + 1, len(pClasses), str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)
            targets = []
            for index in range(pNbData):
                print("    . Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)
                nbRule = xData[index].shape[0]
                indices = [n] * nbRule
                targets.append(tf.one_hot(indices, len(pClasses)))
            pDictParams['xy'] = xData, targets
            pDictParams = _computeExplanations(pDictParams, pClasse=pClasses[n])
    return pDictParams

# ---------------------------------------------------------------------------
## Compute explanations
# @param pDictParams : parameter dictionary
# @return parameter dictionary
def computeExplanationsImages(pDictParams):
    pDictParams = _computeExplanations(pDictParams)
    return pDictParams

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

    if pIndex is not None:
        ficData = os.path.basename(pDataList[pIndex])
        dirName = os.path.dirname(pDataList[pIndex])
        ficExplanation = kaasrc.communs.createDirName(pResultSavedOn, os.path.join(dirName, '%s%s.json'%(ficData, pExtension)))
        if pMinMax:
            minMaxJson = {"min": str(np.min(pExplanation)), "max": str(np.max(pExplanation))}
            with open(ficExplanation, "w", encoding="utf-8") as f:
                json.dump(minMaxJson, f, indent=4)
        ficExplanation = ficExplanation.replace('.json', '.npy')
    else:
        ficExplanation = kaasrc.communs.noSpace(pResultSavedOn, 'allData%s.npy'%pExtension)

    np.save(ficExplanation, pExplanation)
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
    explanations = np.load(pFicExplanation)

    if pDictParams['datatype'] == "tabular":
        # Les explications sont au format (n, 1)
        pParamSpecif = {"dataSize": pDictParams['datasize'][pNumData]}
        kaasrc.controles.control_loadExplanationsOutput(pDictParams, outputXAIFramework, explanations, aiModel.useCase, __file__, pParamSpecif)
    else:
        pParamSpecif = {"explOutputDim": 1}
        kaasrc.controles.control_loadExplanationsOutput(pDictParams, outputXAIFramework, explanations, aiModel.useCase, __file__, pParamSpecif)

    return explanations

# ===============================================================================
# end of file
