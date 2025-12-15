#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================================

import os, time, datetime
import numpy as np

from kaasrc.communs import colPrint
import kaasrc.communs

# ---------
try:
    import pandas as pd
except Exception as err:
    print("Error:", err)
    colPrint("Package pandas is not or not properly installed.", "Error")
# ---------
try:
    import pickle
except Exception as err:
    print("Error:", err)
    colPrint("Package pickle is not or not properly installed.", "Error")
# ---------
try:
    from aix360.algorithms.protodash import ProtodashExplainer
    from aix360.algorithms.shap import KernelExplainer
    from aix360.algorithms.rbm import FeatureBinarizer
    from aix360.algorithms.rbm import BooleanRuleCG
    from aix360.algorithms.rbm import LinearRuleRegression, LogisticRuleRegression
except Exception as err:
    print("Error:", err)
    colPrint("The library 'AIX360' is not or not properly installed.", "Error")
# ---------
try:
    from lime.explanation import Explanation
    from lime.lime_text import LimeTextExplainer
    from lime.lime_tabular import LimeTabularExplainer
    from lime.lime_image import LimeImageExplainer, ImageExplanation
except Exception as err:
    print("Error:", err)
    colPrint("The library 'Lime' is not or not properly installed.", "Error")
# ---------
try:
    import tensorflow.python.framework.ops as eTensor
except Exception as err:
    print("Error:", err)
    colPrint("Package tensorflow is not or not properly installed.", "Error")
# ---------


# ---------------------------------------------------------------------------
# Reset the input and ouput XAI frameworks as in init function
def resetXAIFramework():
    inputXAIframework = (eTensor.EagerTensor, eTensor.EagerTensor)
    outputXAIFramework = {"LimeImage": (list, list, Explanation),
                               "LimeTabular": (list, list, Explanation),
                               "LimeText": (list, list, Explanation),
                               "Shap": (list, list, np.ndarray),
                               "Protodash": (list, list, tuple, np.ndarray),
                               "BRCGE": (list, list, dict),
                               "LinearRuleRegression": (list, list, LinearRuleRegression),
                               "LogisticRuleRegression": (list, list, LogisticRuleRegression)}
    return inputXAIframework, outputXAIFramework

# ---------------------------------------------------------------------------
## Create an instance of the explanability method
# @param pDictParams : parameter dictionary
# @param pIndex : table index to explain in case of tabular problem.
# @return method instance
def getExplainer(pDictParams, pIndex=0):
    # shortcuts
    pMethod = pDictParams['method']
    pMethodParam = pDictParams[pMethod]
    pClasses = pDictParams['classes']
    pDatatype = pDictParams['datatype']
    pModelType = pDictParams['modeltype']
    inputXAIframework = pDictParams['inputXAIframework']
    outputXAIFramework = pDictParams['outputXAIFramework']

    _, aiModel = pDictParams['aiModel']
    pFeatures = aiModel.useCase.features
    xData, _ = kaasrc.communs.convertToXAIframework(pDictParams, inputXAIframework)

    explainer = None
    # Lime
    if pMethod in ["LimeImage", "LimeTabular", "LimeText"]:
        if pDatatype == "image":
            outputXAIFramework[pMethod] = (list, list, ImageExplanation)
            kernel_width = float(pMethodParam["kw"])
            feature_selection = str(pMethodParam["fs"])
            explainer = LimeImageExplainer(kernel_width=kernel_width, feature_selection=feature_selection)
        elif pDatatype == "tabular":
            outputXAIFramework[pMethod] = (list, list, Explanation)
            xData = np.array(xData[pIndex])
            if len(pFeatures) > 1:
                explainer = LimeTabularExplainer(xData, mode="classification" , feature_names=pFeatures, class_names=pClasses)
            else:
                explainer = LimeTabularExplainer(xData, mode=pModelType , feature_names=pFeatures, class_names=pClasses)
        elif pDatatype == "text":
            kernel_width = float(pMethodParam["kw"])
            feature_selection = str(pMethodParam["fs"])
            explainer = LimeTextExplainer(kernel_width=kernel_width, class_names=pClasses, feature_selection=feature_selection)
        else:
            print("Le type de données '%s' n'est pas définie pour la méthode %s."%(pDatatype, pMethod))
    # Shap
    elif pMethod == "Shap":
        nb_samples = int(pMethodParam["ns"])
        if pDatatype == "text":
            datas = np.array([np.array([data]) for data in pDictParams['embedding']])
            explainer = KernelExplainer(aiModel.useCase.predict_list, np.array(datas), nsamples=nb_samples, silent=True)
        elif pDatatype == "image":
            # - - - - - - - - - -
            # define a function that depends on a binary mask representing if an image region is hidden
            # /!\ Attention code peut-être trop spécifique !
            def mask_image(zs, segmentation, image, background=None):
                if background is None:
                    background = image.mean((0, 1))
                out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
                for i in range(zs.shape[0]):
                    out[i, :, :, :] = image
                    for j in range(zs.shape[1]):
                        if zs[i, j] == 0:
                            out[i][segmentation == j, :] = background
                return out
            # - - - - - - - - - -
            mp = pMethodParam["mp"]
            if mp == "Watershed":
                sl = int(pMethodParam["sl"])
                pt = pMethodParam["pt"]
                ny = int(pMethodParam["ny"])
                aiModel.megaPixelsParametres.update({"mapPixels": 0, "seuil": sl, "pretraitement": pt, "noyau": ny})
            else:
                nm = int(pMethodParam["nm"])
                aiModel.megaPixelsParametres["mapPixels"] = nm
            segmentation = aiModel.megaPixels(xData[pIndex])
            img_origin = np.array(xData[pIndex])
            # - - - - - - - - - -
            def functionAsModel(z):
                return aiModel.useCase(mask_image(z, segmentation, img_origin, 255)).numpy()
            # - - - - - - - - - -
            nbReg = int(np.max(segmentation)) + 1
            explainer = KernelExplainer(functionAsModel, np.zeros((1, nbReg)), nsamples=nb_samples, silent=True)
            explainer = explainer, nbReg
        else:
            def functionAsModel(data):
                return aiModel.useCase(data).numpy()
            explainer = KernelExplainer(functionAsModel, np.array(xData[pIndex]), nsamples=nb_samples, silent=True)

    # Protodash
    elif pMethod == "Protodash":
        explainer = ProtodashExplainer()
    # BRCGE
    elif pMethod == "BRCGE":
        lambda0 = float(pMethodParam["l0"])
        lambda1 = float(pMethodParam["l1"])
        cnf = bool(pMethodParam["cf"])
        explainer = BooleanRuleCG(lambda0=lambda0, lambda1=lambda1, CNF=cnf, silent=True)
    # LinearRuleRegression
    elif pMethod == "LinearRuleRegression":
        lambda0 = float(pMethodParam["l0"])
        lambda1 = float(pMethodParam["l1"])
        useOrd = bool(pMethodParam["uo"])
        explainer = LinearRuleRegression(lambda0=lambda0, lambda1=lambda1, useOrd=useOrd)
    # LogisticRuleRegression
    elif pMethod == "LogisticRuleRegression":
        lambda0 = float(pMethodParam["l0"])
        lambda1 = float(pMethodParam["l1"])
        useOrd = bool(pMethodParam["uo"])
        explainer = LogisticRuleRegression(lambda0=lambda0, lambda1=lambda1, useOrd=useOrd)
    else:
        print("La méthode '%s' n'est pas définie."%pMethod)
    return explainer

# ---------------------------------------------------------------------------
## Compute explanations
# @param pDictParams : parameter dictionary
# @return parameter dictionary
def computeExplanationsTables(pDictParams):
    # shortcuts
    pMethod = pDictParams['method']
    pMethodParam = pDictParams[pMethod]
    pNbData = pDictParams['nbData']
    pRepertProd = pDictParams['repertProd']
    pDataProd = pDictParams['dataProd']
    pClasses = pDictParams['classes']
    pResultats = pDictParams['predictions']

    resultSavedOn = os.path.join(pDataProd, "dataExplanations", pRepertProd)
    _, aiModel = pDictParams['aiModel']

    pFeatures = aiModel.useCase.features
    xData, yPred = pDictParams['xy']

    inputXAIframework, outputXAIFramework = resetXAIFramework()
    if pMethod in ["BRCGE", "LinearRuleRegression", "LogisticRuleRegression", "LimeTabular"]:
        inputXAIframework = (pd.DataFrame, inputXAIframework[1])

    # convert data and prediction to XAI framework
    xData, yPred = kaasrc.communs.convertToXAIframework(pDictParams, inputXAIframework)

    # control function for XAI_computeExplanations inputs
    pParamSpecif = {"dataSize": pDictParams['datasize']}
    kaasrc.controles.controlXAI_computeExplanationsInput(pDictParams, inputXAIframework, xData, yPred, aiModel.useCase, pNbData, __file__, pParamSpecif)

    aiModel.setInferenceCible(False)
    # print("   .get explainer %s"%pMethod, flush=True)
    # explainer = getExplainer(pDictParams)

    print("==[%s] ============================================================================="%pMethod, flush=True)
    start = time.time()
    explanations = []
    # Lime
    if pMethod == "LimeTabular":
        # x is denormalized so you need to call a function which normalized x and then call the predict function
        npPredict = lambda x: np.array(aiModel.useCase.normPredict(x))
        for index in range(pNbData):
            print(" - Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)
            explanationClasses = []
            npXinputs = np.array(xData[index])
            explainer = getExplainer(pDictParams, index)
            for entry, _ in enumerate(npXinputs):
                if pFeatures is not None:
                    exp = explainer.explain_instance(npXinputs[entry], npPredict, num_features=len(pFeatures), labels=list(range(len(pClasses))))
                else:
                    exp = explainer.explain_instance(npXinputs[entry], npPredict, labels=list(range(len(pClasses))))
                explanationClasses.append(exp)
            explanations.append(explanationClasses)

    # Shap
    elif pMethod == "Shap":
        for index in range(pNbData):
            print(" - Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)
            explanationData = []
            explainer = getExplainer(pDictParams, index)
            explanationData = explainer.explain_instance(np.array(xData[index]), silent=True)
            explanations.append(explanationData)

    # Protodash table
    elif pMethod == "Protodash":
        if 'sameTrainingPrediction' not in pDictParams:
            colPrint("The UCXAI plugin must implement a method of searching for similar elements.", "Error")
            return
        sameTrainingPrediction = pDictParams['sameTrainingPrediction']
        explainer = getExplainer(pDictParams)
        for index in range(pNbData):
            print(" - Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)
            explanationData = []
            npXinputs = np.array(xData[index])
            for entry, _ in enumerate(npXinputs):

                dataItem, _, examplePred, _ = sameTrainingPrediction(pXData=npXinputs, pYPred=pResultats[index], pItem=entry)

                nbProtos = int(pMethodParam["m"])
                explanation = None
                while explanation is None:
                    try:
                        explanation = explainer.explain(dataItem, examplePred, m=nbProtos)
                    except Exception as err:
                        print("Error:", err)
                        explanation = None
                        newNbProto = nbProtos // 2
                        if newNbProto != 0:
                            colPrint("Reduce number of prototypes from %d to %d to have a result"%(nbProtos, newNbProto), "Normal")
                            nbProtos = newNbProto
                        else:
                            colPrint("Cannot found prototypes for entry #%d!"%(entry + 1), "Normal")
                            break
                if explanation is None:
                    explanation = (None, None, None)
                explanationData.append(explanation)
            explanations.append(explanationData)

    # BRCGE
    elif pMethod == "BRCGE":
        fb = FeatureBinarizer(negations=True, returnOrd=True)
        for index in range(pNbData):
            print(" - Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)
            explanationClasses = []
            resPredictCibles = np.array(aiModel.useCase.resPredictionsCibles[index])
            inputs = xData[index]
            dfData, dfDataStd = fb.fit_transform(inputs)
            for c, _ in enumerate(pClasses):
                yPred = np.array([1 if np.argmax(pred) == c else 0 for pred in resPredictCibles])
                if list(yPred) == list([0] * len(xData[index])):
                    explanationClasses.append({})
                else:
                    explainer = getExplainer(pDictParams)
                    explainer.fit(dfData, yPred)
                    explanation = explainer.explain()
                    explanationClasses.append(explanation)
            explanations.append(explanationClasses)

    # LinearRuleRegression
    elif pMethod == "LinearRuleRegression":
        fb = FeatureBinarizer(negations=True, returnOrd=True)
        for index in range(pNbData):
            print(" - Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)
            explanationClasses = []
            resPredictCibles = np.array(aiModel.useCase.resPredictionsCibles[index])
            inputs = xData[index]
            dfData, dfDataStd = fb.fit_transform(inputs)
            for c, _ in enumerate(pClasses):
                explainer = getExplainer(pDictParams)
                yPred = np.array([1 if np.argmax(pred) == c else 0 for pred in resPredictCibles])
                if list(yPred) == list([0] * len(xData[index])):
                    explanationClasses.append(None)
                else:
                    explainer.fit(dfData, yPred, dfDataStd)
                    explanationClasses.append(explainer)
            explanations.append(explanationClasses)

    # LogisticRuleRegression
    elif pMethod == "LogisticRuleRegression":
        fb = FeatureBinarizer(negations=True, returnOrd=True)
        for index in range(pNbData):
            print(" - Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)
            explanationClasses = []
            resPredictCibles = np.array(aiModel.useCase.resPredictionsCibles[index])
            inputs = xData[index]
            dfData, dfDataStd = fb.fit_transform(inputs)
            for c, _ in enumerate(pClasses):
                explainer = getExplainer(pDictParams)
                yPred = np.array([1 if np.argmax(pred) == c else 0 for pred in resPredictCibles])
                if list(yPred) == list([0] * len(xData[index])):
                    explanationClasses.append(None)
                else:
                    explainer.fit(dfData, yPred, dfDataStd)
                    explanationClasses.append(explainer)
            explanations.append(explanationClasses)

    end = time.time()
    print("Duration:", end - start)

    # control function for XAI_computeExplanations outputs
    if pMethod == "BRCGE":
        pParamSpecif = {"expected": (pNbData, aiModel.useCase.numClasses, None), "format": ("n", "[C]", None)}
        kaasrc.controles.controlXAI_computeExplanationsOutput(pDictParams, outputXAIFramework[pMethod], explanations, aiModel.useCase, pNbData, __file__, pParamSpecif)
    elif pMethod == "Shap":
        pParamSpecif = {"expected": (pNbData, aiModel.useCase.numClasses, pDictParams['datasize']), "levelData": (0, 2), "format": ("n", "[C]", "(l, f)")}
        kaasrc.controles.controlXAI_computeExplanationsOutput(pDictParams, outputXAIFramework[pMethod], explanations, aiModel.useCase, pNbData, __file__, pParamSpecif)
    elif pMethod == "LimeTabular":
        dataLsize = [x for x, _ in pDictParams['datasize']]
        pParamSpecif = {"expected": (pNbData, dataLsize, None), "format": ("n", "l", "?"), "levelData": (0, 1)}
        kaasrc.controles.controlXAI_computeExplanationsOutput(pDictParams, outputXAIFramework[pMethod], explanations, aiModel.useCase, pNbData, __file__, pParamSpecif)
    elif pMethod == "LinearRuleRegression":
        pParamSpecif = {"expected": (pNbData, aiModel.useCase.numClasses, None), "format": ("n", "C", "?")}
        kaasrc.controles.controlXAI_computeExplanationsOutput(pDictParams, outputXAIFramework[pMethod], explanations, aiModel.useCase, pNbData, __file__, pParamSpecif)
    elif pMethod == "LogisticRuleRegression":
        pParamSpecif = {"expected": (pNbData, aiModel.useCase.numClasses, None), "format": ("n", "C", "?")}
        kaasrc.controles.controlXAI_computeExplanationsOutput(pDictParams, outputXAIFramework[pMethod], explanations, aiModel.useCase, pNbData, __file__, pParamSpecif)
    elif pMethod == "Protodash":
        dataLsize = [x for x, _ in pDictParams['datasize']]
        pParamSpecif = {"expected": (pNbData, dataLsize, 3, (nbProtos, )), "format": ("n", "l", "w/p/v", "Proto"), "levelData": (0, 1)}
        kaasrc.controles.controlXAI_computeExplanationsOutput(pDictParams, outputXAIFramework[pMethod], explanations, aiModel.useCase, pNbData, __file__, pParamSpecif)
    else:
        kaasrc.controles.NOcontrolXAI_computeExplanationsOutput(pDictParams)

    # Sauvegarde : Boucle sur les données
    for index in range(pNbData):
        _saveExplanations(pDictParams, explanations[index], resultSavedOn, index)
        # et boucle sur les classes
        for eClasse, vClasse in enumerate(pClasses):
            _saveExplanations(pDictParams, explanations[index][eClasse], resultSavedOn, index, pExtension="--c_%s"%vClasse)
    # save all explanations to use them in launchMetric() function
    _saveExplanations(pDictParams, explanations, resultSavedOn)

    return pDictParams

# ---------------------------------------------------------------------------
## Compute explanations
# @param pDictParams : parameter dictionary
# @return parameter dictionary
def computeExplanationsImages(pDictParams):
    # shortcuts
    pMethod = pDictParams['method']
    pMethodParam = pDictParams[pMethod]
    pNbData = pDictParams['nbData']
    pRepertProd = pDictParams['repertProd']
    pDataProd = pDictParams['dataProd']
    pClasses = pDictParams['classes']
    inputXAIframework = pDictParams['inputXAIframework']
    outputXAIFramework = pDictParams['outputXAIFramework']

    resultSavedOn = os.path.join(pDataProd, "dataExplanations", pRepertProd)
    _, aiModel = pDictParams['aiModel']

    pFeatures = aiModel.useCase.features

    # convert data and prediction to XAI framework
    xData, yPred = kaasrc.communs.convertToXAIframework(pDictParams, inputXAIframework)

    # control function for XAI_computeExplanations inputs
    kaasrc.controles.controlXAI_computeExplanationsInput(pDictParams, inputXAIframework, xData, yPred, aiModel.useCase, pNbData, __file__)

    aiModel.setInferenceCible(False)
    # print("   .get explainer %s"%pMethod, flush=True)
    # explainer = getExplainer(pDictParams)

    print("==[%s] ============================================================================="%pMethod, flush=True)
    start = time.time()
    # Lime
    if pMethod == "LimeImage":
        xData = np.array(xData)
        explanations = []
        for index in range(pNbData):
            print(" - Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)
            npPredict = lambda x: np.array(aiModel.useCase(x))
            explainer = getExplainer(pDictParams, index)
            if pFeatures is not None:
                exp = explainer.explain_instance(xData[index],
                                                npPredict,
                                                num_features=len(pFeatures),
                                                labels=[i for i, _ in enumerate(pClasses)])
            else:
                exp = explainer.explain_instance(xData[index],
                                                npPredict,
                                                labels=[i for i, _ in enumerate(pClasses)])
            explanations.append(exp)
    # Shap
    elif pMethod == "Shap":
        nb_samples = int(pMethodParam["ns"])
        mp = pMethodParam["mp"]
        # hw=700
        if mp == "Watershed":
            sl = int(pMethodParam["sl"])
            pt = pMethodParam["pt"]
            ny = int(pMethodParam["ny"])
            aiModel.megaPixelsParametres.update({"mapPixels": 0, "seuil": sl, "pretraitement": pt, "noyau": ny})
        else:
            nm = int(pMethodParam["nm"])
            aiModel.megaPixelsParametres["mapPixels"] = nm
        explanations = []
        nbSegment = []
        for index in range(pNbData):
            print(" - Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)
            explainer, nbReg = getExplainer(pDictParams, index)
            explanation = explainer.explain_instance(np.ones((1, nbReg)), nsamples=nb_samples, silent=True)
            nbSegment.append(nbReg)
            explanations.append(explanation)

    end = time.time()
    print("Duration:", end - start)

    # control function for XAI_computeExplanations outputs
    if pMethod == "LimeImage":
        pParamSpecif = {"expected": (pNbData, None), "format": ("n", "?")}
        kaasrc.controles.controlXAI_computeExplanationsOutput(pDictParams, outputXAIFramework[pMethod][1:], explanations, aiModel.useCase, pNbData, __file__, pParamSpecif)
    elif pMethod == "Shap":
        aAtteindre = (pNbData, aiModel.useCase.numClasses, None)
        strFormat = ("[n]", "[C]", None)
        pParamSpecif = {"expected": aAtteindre, "format": strFormat}
        kaasrc.controles.controlXAI_computeExplanationsOutput(pDictParams, outputXAIFramework[pMethod], explanations, aiModel.useCase, pNbData, __file__, pParamSpecif)

    # Sauvegarde : Boucle sur les données
    for index in range(pNbData):
        _saveExplanations(pDictParams, explanations[index], resultSavedOn, index)

    # save all explanations to use them in launchMetric() function
    _saveExplanations(pDictParams, explanations, resultSavedOn)
    return pDictParams

# ---------------------------------------------------------------------------
## Compute explanations
# @param pDictParams : parameter dictionary
# @return parameter dictionary
def computeExplanationsTexts(pDictParams):
    # shortcuts
    pMethod = pDictParams['method']
    pMethodParam = pDictParams[pMethod]
    pNbData = pDictParams['nbData']
    pRepertProd = pDictParams['repertProd']
    pDataProd = pDictParams['dataProd']
    pClasses = pDictParams['classes']
    pUseCaseBase = pDictParams['useCaseBase']
    inputXAIframework = pDictParams['inputXAIframework']

    resultSavedOn = os.path.join(pDataProd, "dataExplanations", pRepertProd)
    _, aiModel = pDictParams['aiModel']

    pFeatures = aiModel.useCase.features
    xData, yPred = pDictParams['xy']

    # control function for XAI_computeExplanations inputs
    kaasrc.controles.controlXAI_computeExplanationsInput(pDictParams, inputXAIframework, xData, yPred, aiModel.useCase, pNbData, __file__)

    aiModel.setInferenceCible(False)
    # print("   .get explainer %s"%pMethod, flush=True)
    # explainer = getExplainer(pDictParams)

    print("==[%s] ============================================================================="%pMethod, flush=True)
    start = time.time()
    explanations = []
    # Lime
    if pMethod == "LimeText":
        npPredict = lambda x: np.array([np.array(aiModel.useCase.predict_from_text(xi)) for xi in x])
        for index in range(pNbData):
            print(" - Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)
            explanationItems = []
            npXinputs = np.array(xData[index])
            explainer = getExplainer(pDictParams, index)
            for entry, _ in enumerate(npXinputs):
                if pFeatures is not None:
                    exp = explainer.explain_instance(npXinputs[entry], npPredict, num_features=len(pFeatures), labels=[i for i, _ in enumerate(pClasses)])
                else:
                    exp = explainer.explain_instance(npXinputs[entry], npPredict, labels=[i for i, _ in enumerate(pClasses)])
                explanationItems.append(exp)
            explanations.append(explanationItems)
    # Protodash text
    elif pMethod == "Protodash":
        if 'sameTrainingPrediction' not in pDictParams:
            colPrint("The UCXAI plugin must implement a method of searching for similar elements.", "Error")
            return
        sameTrainingPrediction = pDictParams['sameTrainingPrediction']
        explainer = getExplainer(pDictParams)
        m = int(pMethodParam["m"])
        for index in range(pNbData):
            print(" - Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)
            explanationData = []
            # npXinputs = np.array(xData[index])
            for entry, _ in enumerate(xData[index]):
                example = None
                training_examples = None
                train_comment, train_label = aiModel.useCase.getData(pUseCaseBase, pDictParams['dataTrainPath'])
                example, _, training_examples, _ = sameTrainingPrediction(xData[index], train_comment, train_label, index, entry, aiModel)
                try:
                    explanationData.append(explainer.explain(example, training_examples, m=m))
                except Exception as err:
                    print("Error:", err)
                    print("Reduce parameter '(m) number of prototypes' -actually m=%d- to have a result"%m)
                    explanationData.append(None)
            explanations.append(explanationData)

    end = time.time()
    print("Duration:", end - start)

    # control function for XAI_computeExplanations outputs
    kaasrc.controles.NOcontrolXAI_computeExplanationsOutput(pDictParams)

    # Sauvegarde : Boucle sur les données
    for index in range(pNbData):
        _saveExplanations(pDictParams, explanations[index], resultSavedOn, index)

    # save all explanations to use them in launchMetric() function
    _saveExplanations(pDictParams, explanations, resultSavedOn)

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
        ficExplanation = kaasrc.communs.createDirName(pResultSavedOn, os.path.join(dirName, '%s%s.pkl'%(ficData, pExtension)))
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
# @param pAllData : are all data loaded? (True in metric computation process)
# @return explanations loaded
def loadExplanations(pDictParams, pFicExplanation, pNumData=None, pAllData=False):
    # shortcuts
    _, aiModel = pDictParams['aiModel']
    pMethod = pDictParams['method']
    pMethodParam = pDictParams[pMethod]
    pDataType = pDictParams['datatype']
    outputXAIFramework = pDictParams['outputXAIFramework']

    if not os.path.exists(pFicExplanation):
        colPrint("The file containing the explanation '%s' does not exists."%pFicExplanation, "Error")
        return None
    with open(pFicExplanation, 'rb') as fid:
        oneExplanation = pickle.load(fid)

    # Les explications sont dans formats multiples complexe sur AIX360
    # cas de lecture pour les métriques
    if pAllData:
        if pMethod in ["LimeImage", "LimeTabular", "LimeText"]:
            if pDataType == "tabular":
                outputXAIFramework[pMethod] = (list, list, Explanation)
                dataLsize = [x for x, _ in pDictParams['datasize']]
                pParamSpecif = {"expected": (pNumData, dataLsize, None), "format": ("n", "l", "?"), "levelData": (0, 1)}
                kaasrc.controles.control_loadExplanationsOutput(pDictParams, outputXAIFramework[pMethod], oneExplanation, aiModel.useCase, __file__, pParamSpecif)
            else:
                outputXAIFramework[pMethod] = (list, ImageExplanation)
                pParamSpecif = {"expected": (pNumData, None), "format": ("n", "?")}
                kaasrc.controles.control_loadExplanationsOutput(pDictParams, outputXAIFramework[pMethod], oneExplanation, aiModel.useCase, __file__, pParamSpecif)
        elif pMethod == "LinearRuleRegression":
            pParamSpecif = {"expected": (pNumData, aiModel.useCase.numClasses, None), "format": ("n", "C", "?")}
            kaasrc.controles.control_loadExplanationsOutput(pDictParams, outputXAIFramework[pMethod], oneExplanation, aiModel.useCase, __file__, pParamSpecif)
        elif pMethod == "LogisticRuleRegression":
            pParamSpecif = {"expected": (pNumData, aiModel.useCase.numClasses, None), "format": ("n", "C", "?")}
            kaasrc.controles.control_loadExplanationsOutput(pDictParams, outputXAIFramework[pMethod], oneExplanation, aiModel.useCase, __file__, pParamSpecif)
        else:
            kaasrc.controles.control_NOloadExplanationsOutput(pDictParams)
    # cas de lecture pour les explications
    else:
        if pMethod in ["LimeImage", "LimeTabular", "LimeText"]:
            if pDataType == "tabular":
                outputXAIFramework[pMethod] = (list, list, Explanation)
                pParamSpecif = {"expected": (pDictParams['datasize'][pNumData][0], None), "format": ("l", "?")}
                kaasrc.controles.control_loadExplanationsOutput(pDictParams, outputXAIFramework[pMethod][1:], oneExplanation, aiModel.useCase, __file__, pParamSpecif)
            else:
                outputXAIFramework[pMethod] = ImageExplanation
                pParamSpecif = {"expected": None, "format": "?"}
                kaasrc.controles.control_loadExplanationsOutput(pDictParams, outputXAIFramework[pMethod], oneExplanation, aiModel.useCase, __file__, pParamSpecif)
        elif pMethod in ["Shap"]:
            if pDataType == "tabular":
                pParamSpecif = {"expected": (aiModel.useCase.numClasses, pDictParams['datasize'][pNumData]), "format": ("[C]", "(l, f)")}
            else:
                pParamSpecif = {"expected": (aiModel.useCase.numClasses, None), "format": ("[C]", None)}
            kaasrc.controles.control_loadExplanationsOutput(pDictParams, outputXAIFramework[pMethod][1:], oneExplanation, aiModel.useCase, __file__, pParamSpecif)
        elif pMethod in ["BRCGE"]:
            pParamSpecif = {"expected": (aiModel.useCase.numClasses, None), "format": ("[C]", None)}
            kaasrc.controles.control_loadExplanationsOutput(pDictParams, outputXAIFramework[pMethod][1:], oneExplanation, aiModel.useCase, __file__, pParamSpecif)
        elif pMethod == "LinearRuleRegression":
            pParamSpecif = {"expected": (aiModel.useCase.numClasses, None), "format": ("C", "?")}
            kaasrc.controles.control_loadExplanationsOutput(pDictParams, outputXAIFramework[pMethod][1:], oneExplanation, aiModel.useCase, __file__, pParamSpecif)
        elif pMethod == "LogisticRuleRegression":
            pParamSpecif = {"expected": (aiModel.useCase.numClasses, None), "format": ("C", "?")}
            kaasrc.controles.control_loadExplanationsOutput(pDictParams, outputXAIFramework[pMethod][1:], oneExplanation, aiModel.useCase, __file__, pParamSpecif)
        elif pMethod == "Protodash":
            if pDataType == "tabular":
                nbProtos = int(pMethodParam["m"])
                pParamSpecif = {"expected": (pDictParams['datasize'][pNumData][0], 3, (nbProtos, )), "format": ("l", "w/p/v", "Proto")}
                kaasrc.controles.control_loadExplanationsOutput(pDictParams, outputXAIFramework[pMethod][1:], oneExplanation, aiModel.useCase, __file__, pParamSpecif)
            else:
                kaasrc.controles.NOcontrol_loadExplanationsOutput(pDictParams)
        else:
            kaasrc.controles.NOcontrol_loadExplanationsOutput(pDictParams)

    return oneExplanation

# ===============================================================================
# end of file
