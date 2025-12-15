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
    from sklearn import linear_model
except Exception as err:
    print("Error:", err)
    colPrint("Package sklearn is not or not properly installed.", "Error")
# ---------
try:
    import tensorflow.compat.v2 as tf
except Exception as err:
    print("Error:", err)
    colPrint("Package tensorflow is not or not properly installed.", "Error")
# ---------
try:
    from xplique.attributions import KernelShap, Occlusion, Rise, Lime, IntegratedGradients, SmoothGrad, VarGrad, SquareGrad, GuidedBackprop, DeconvNet, GradientInput, Saliency, SobolAttributionMethod, HsicAttributionMethod
    from xplique.commons.operators import regression_operator
    from xplique.commons.operators import object_detection_operator
    from xplique.commons.operators import semantic_segmentation_operator
    from xplique.utils_functions.segmentation import get_class_zone, list_class_connected_zones
except Exception as err:
    print("Error:", err)
    colPrint("The library 'Xplique' is not or not properly installed.", "Error")
# ---------


# -------------------------------------------------------------------------------
## Function that check if a class has a member function
# param pClass : the class to inspect
# param pFunction : the function to check
#  def has_method(pClass, pFunction):
#    return callable(getattr(pClass, pFunction, None))

# -------------------------------------------------------------------------------

# ---------------------------------------------------------------------------
## Create an instance of the explainability method
# @param pAiModel : model instance
# @param pMethod : method used for explanation
# @param pMethodParam : method params used for explanation
# @param pOperator : specific operator (cf. Xplique library)
# @return method instance
def getExplainer(pAiModel, pMethod, pMethodParam, pOperator=None):
    # KernelShap
    if pMethod == "KernelShap":
        bz = int(pMethodParam["bz"])
        ns = int(pMethodParam["ns"])
        rv = float(pMethodParam["rv"])
        if pAiModel.useCase.numDataChannels > 0:
            rv = np.full(pAiModel.useCase.numDataChannels, rv)
        mp = pMethodParam["mp"]
        if mp == "Quickshift":
            fctMegaPixels = None
        elif mp == "Watershed":
            sl = int(pMethodParam["sl"])
            pt = pMethodParam["pt"]
            ny = int(pMethodParam["ny"])
            fctMegaPixels = pAiModel.megaPixels
            pAiModel.megaPixelsParametres.update({"mapPixels": 0, "seuil": sl, "pretraitement": pt, "noyau": ny})
        else:
            nm = int(pMethodParam["nm"])
            fctMegaPixels = pAiModel.megaPixels
            pAiModel.megaPixelsParametres["mapPixels"] = nm
        print("     Fonction Mega-Pixels : %s"%mp)
        explainer = KernelShap(pAiModel.useCase, operator=pOperator, batch_size=bz, map_to_interpret_space=fctMegaPixels, ref_value=rv, nb_samples=ns)
    # Occlusion
    elif pMethod == 'Occlusion':
        bz = int(pMethodParam["bz"])
        pz = int(pMethodParam["pz"])
        ps = int(pMethodParam["ps"])
        ov = float(pMethodParam["ov"])
        explainer = Occlusion(pAiModel.useCase, operator=pOperator, batch_size=bz, patch_size=pz, patch_stride=ps, occlusion_value=ov)
    # Rise
    elif pMethod == 'Rise':
        bz = int(pMethodParam["bz"])
        ns = int(pMethodParam["ns"])
        gz = int(pMethodParam["gz"])
        pp = float(pMethodParam["pp"])
        explainer = Rise(pAiModel.useCase, operator=pOperator, batch_size=bz, nb_samples=ns, grid_size=gz, preservation_probability=pp)
    # Lime
    elif pMethod == "Lime":
        bz = int(pMethodParam["bz"])
        ns = int(pMethodParam["ns"])
        rv = float(pMethodParam["rv"])
        if pAiModel.useCase.numDataChannels > 0:
            rv = np.full(pAiModel.useCase.numDataChannels, rv)
        dm = pMethodParam["dm"]
        kw = float(pMethodParam["kw"])
        pp = float(pMethodParam["pp"])
        mp = pMethodParam["mp"]
        if mp == "Quickshift":
            fctMegaPixels = None
        elif mp == "Watershed":
            sl = int(pMethodParam["sl"])
            pt = pMethodParam["pt"]
            ny = int(pMethodParam["ny"])
            fctMegaPixels = pAiModel.megaPixels
            pAiModel.megaPixelsParametres.update({"mapPixels": 0, "seuil": sl, "pretraitement": pt, "noyau": ny})
        else:
            nm = int(pMethodParam["nm"])
            fctMegaPixels = pAiModel.megaPixels
            pAiModel.megaPixelsParametres["mapPixels"] = nm
        explainer = Lime(pAiModel.useCase, operator=pOperator, batch_size=bz, interpretable_model=linear_model.Lasso(alpha=0.1), map_to_interpret_space=fctMegaPixels, ref_value=rv, nb_samples=ns, distance_mode=dm, kernel_width=kw, prob=pp)
    # Integrated Gradients
    elif pMethod == 'IntegratedGradients':
        bz = int(pMethodParam['bz'])
        ns = int(pMethodParam['ns'])
        bl = float(pMethodParam['bl'])
        explainer = IntegratedGradients(pAiModel.useCase.model, operator=pOperator, batch_size=bz, steps=ns, baseline_value=bl)
    # SmoothGrad
    elif pMethod == 'SmoothGrad':
        bz = int(pMethodParam['bz'])
        ns = int(pMethodParam['ns'])
        nz = float(pMethodParam['nz'])
        explainer = SmoothGrad(pAiModel.useCase.model, operator=pOperator, batch_size=bz, nb_samples=ns, noise=nz)
    # VarGrad
    elif pMethod == 'VarGrad':
        bz = int(pMethodParam['bz'])
        ns = int(pMethodParam['ns'])
        nz = float(pMethodParam['nz'])
        explainer = VarGrad(pAiModel.useCase.model, operator=pOperator, batch_size=bz, nb_samples=ns, noise=nz)
    # SquareGrad
    elif pMethod == 'SquareGrad':
        bz = int(pMethodParam['bz'])
        ns = int(pMethodParam['ns'])
        nz = float(pMethodParam['nz'])
        explainer = SquareGrad(pAiModel.useCase.model, operator=pOperator, batch_size=bz, nb_samples=ns, noise=nz)
    # GuidedBackprop
    elif pMethod == 'GuidedBackprop':
        bz = int(pMethodParam['bz'])
        explainer = GuidedBackprop(pAiModel.useCase.model, operator=pOperator, batch_size=bz)
    # DeconvNet
    elif pMethod == 'DeconvNet':
        bz = int(pMethodParam['bz'])
        explainer = DeconvNet(pAiModel.useCase.model, operator=pOperator, batch_size=bz)
    # GradientInput
    elif pMethod == 'GradientInput':
        bz = int(pMethodParam['bz'])
        explainer = GradientInput(pAiModel.useCase.model, operator=pOperator, batch_size=bz)
    elif pMethod == 'Saliency':
        bz = int(pMethodParam["bz"])
        explainer = Saliency(pAiModel.useCase.model, operator=pOperator, batch_size=bz)
    elif pMethod == 'Sobol':
        bz = int(pMethodParam["bz"])
        nd = int(pMethodParam["nd"])
        gz = int(pMethodParam["gz"])
        fp = pMethodParam["fp"]
        explainer = SobolAttributionMethod(pAiModel.useCase, operator=pOperator, grid_size=gz, nb_design=nd, perturbation_function=fp, batch_size=bz)
    elif pMethod == 'Hsic':
        bz = int(pMethodParam["bz"])
        nd = int(pMethodParam["nd"])
        gz = int(pMethodParam["gz"])
        fp = pMethodParam["fp"]
        explainer = HsicAttributionMethod(pAiModel.useCase, operator=pOperator, grid_size=gz, nb_design=nd, perturbation_function=fp, batch_size=bz, estimator_batch_size=bz)
    else:
        print("La méthode '%s' n'est pas définie."%pMethod)
        explainer = None
    return explainer

# ---------------------------------------------------------------------------
## Compute explanations
# @param pDictParams : parameter dictionary
# @return parameter dictionary upgraded by object detected list
def computeExplanationsObjDetection(pDictParams):
    # shortcuts
    pMethod = pDictParams['method']
    pMethodParam = pDictParams[pMethod]
    pNbData = pDictParams['nbData']
    pDataList = pDictParams['dataList']
    pListObjets = pDictParams['objets']
    pRepertProd = pDictParams['repertProd']
    pDataProd = pDictParams['dataProd']
    _, aiModel = pDictParams['aiModel']
    xData, yPred = pDictParams['xy']
    inputXAIframework = pDictParams['inputXAIframework']
    outputXAIFramework = pDictParams['outputXAIFramework']

    resultSavedOn = os.path.join(pDataProd, "dataExplanations", pRepertProd)

    xData, yPred = kaasrc.communs.convertToXAIframework(pDictParams, inputXAIframework)

    # control function for XAI_computeExplanations inputs
    pParamSpecif = {"nbBbx": aiModel.useCase.nbObjects}
    kaasrc.controles.controlXAI_computeExplanationsInput(pDictParams, inputXAIframework, xData, yPred, aiModel.useCase, pNbData, __file__, pParamSpecif)

    aiModel.setInferenceCible(False)
    print("   .get explainer %s"%pMethod, flush=True)

    explainer = getExplainer(aiModel, pMethod, pMethodParam, object_detection_operator)

    print("==[%s] ============================================================================="%pMethod, flush=True)
    # Boucle sur les données BdBx
    explanations = []
    nbExplanations = 0
    listObjectExplained = {}
    for index in range(pNbData):
        nbObserv = len(yPred[index])
        print("Explanation Img. %d/%d Nb.Tot.BBx %d (%s)"%(index + 1, pNbData, nbObserv, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)

        # Sélection des objets à expliquer
        listObjets = pListObjets[pDataList[index]]
        if len(listObjets) == 0:
            listObjets = list(range(nbObserv))
        listObjectExplained[pDataList[index]] = listObjets
        pDictParams['objets'][pDataList[index]] = listObjets

        # Boucle sur les objets
        for idxBBx, predBBx in enumerate(listObjets):
            # Explication de l'image 'index' et la bounding box 'predBBx'
            print("Explanation Img. %d/%d BBx %d/%d n°%d"%(index + 1, pNbData, idxBBx + 1, len(listObjets), predBBx), flush=True)

            start = time.time()
            explanation = explainer.explain(xData[index], tf.expand_dims(yPred[index][predBBx], axis=0))
            end = time.time()
            print("Duration:", end - start)

            explanation = tf.squeeze(explanation, axis=0)
            explanations.append(explanation)
            nbExplanations += 1

            # Sauve l'explication
            _saveExplanations(pDictParams, explanation, resultSavedOn, index, "--i_%d"%predBBx)

    # control function for XAI_computeExplanations outputs
    kaasrc.controles.controlXAI_computeExplanationsOutput(pDictParams, outputXAIFramework, explanations, aiModel.useCase, nbExplanations, __file__)

    # save all explanations to use them in launchMetric() function
    _saveExplanations(pDictParams, explanations, resultSavedOn)

    # Sauvegarde de la liste des objets détectés
    pDictParams['objets'] = listObjectExplained
    ficObjets = os.path.join(resultSavedOn, 'listObjects.json')
    with open(ficObjets, "w", encoding="utf-8") as f:
        json.dump(pDictParams['objets'], f, indent=4)

    return pDictParams

# ---------------------------------------------------------------------------
## Compute explanations for semantic segmentation models
# @param pDictParams : parameter dictionary
# @return parameter dictionary upgraded by object detected list
def computeExplanationsObjSegmentation(pDictParams):
    # shortcuts
    pMethod = pDictParams['method']
    pMethodParam = pDictParams[pMethod]
    pNbData = pDictParams['nbData']
    pDataList = pDictParams['dataList']
    pListObjets = pDictParams['objets']
    pRepertProd = pDictParams['repertProd']
    pDataProd = pDictParams['dataProd']
    pClasses = pDictParams['classes']
    _, aiModel = pDictParams['aiModel']
    xData, yPred = pDictParams['xy']
    inputXAIframework = pDictParams['inputXAIframework']
    outputXAIFramework = pDictParams['outputXAIFramework']

    resultSavedOn = os.path.join(pDataProd, "dataExplanations", pRepertProd)
    xData, yPred = kaasrc.communs.convertToXAIframework(pDictParams, inputXAIframework)

    # control function for XAI_computeExplanations inputs
    pParamSpecif = {"nbSegm": aiModel.useCase.nbObjects}
    kaasrc.controles.controlXAI_computeExplanationsInput(pDictParams, inputXAIframework, xData, yPred, aiModel.useCase, pNbData, __file__, pParamSpecif)

    aiModel.setInferenceCible(False)
    print("   .get explainer %s"%pMethod, flush=True)

    explainer = getExplainer(aiModel, pMethod, pMethodParam, semantic_segmentation_operator)

    print("==[%s] ============================================================================="%pMethod, flush=True)
    allExplanations = []
    listObjectExplained = {}
    # Boucle sur les données Segm
    for index in range(pNbData):

        my_input = xData[index]
        predictions = yPred[index]

        predictedClassesIdx, _ = tf.unique(tf.reshape(tf.argmax(predictions, axis=2), [-1]))
        predictedClassesIdx = predictedClassesIdx.numpy().tolist()
        predictedClassesNames = [pClasses[i] for i in predictedClassesIdx]

        print("Explanation Img. %d/%d Nb.Classes %d (%s)"%(index + 1, pNbData, len(predictedClassesNames), str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)

        # Sélection des objets à expliquer
        listObjets = pListObjets[pDataList[index]]
        if len(listObjets) == 0:
            # expliquer toutes les classes prédites par le modèle
            listObjets = [[className, []] for className in predictedClassesNames]
        else:
            # laisser que les classes qui sont prédites par le modèle
            listObjets = [classListObjets for classListObjets in listObjets if classListObjets[0] in predictedClassesNames]
        listObjectExplained[pDataList[index]] = listObjets
        pDictParams['objets'][pDataList[index]] = listObjets

        # Boucle sur les classes
        explanationsClasses = []
        for classListObjets_index, classListObjets in enumerate(listObjets):
            # Explication de l'image 'index' et la classe 'classListObjets_index'
            print("Explanation Img. %d/%d Class %d/%d"%(index + 1, pNbData, classListObjets_index + 1, len(predictedClassesNames)), flush=True)
            className = classListObjets[0]
            nClasse = pClasses.index(className)
            if len(classListObjets) == 1:
                # Explication globale de la classe
                target = get_class_zone(predictions, nClasse)
                target = tf.expand_dims(target, axis=0)

                start = time.time()
                explanation = explainer.explain(my_input, target)
                end = time.time()
                print("Duration: %s"%str(end - start), flush=True)

                explanation = tf.squeeze(explanation, axis=0)
                explanationsClasses.extend([explanation])

                # Sauve l'explication
                _saveExplanations(pDictParams, explanation, resultSavedOn, index, "--c_%s"%className)

                # control function for XAI_computeExplanations outputs
                kaasrc.controles.controlXAI_computeExplanationsOutput(pDictParams, outputXAIFramework, [explanation], aiModel.useCase, 1, __file__)

            else:
                objects = classListObjets[1]
                connected_zones = list_class_connected_zones(predictions, nClasse, 5)

                if len(objects) == 0:
                    objects = list(range(0, len(connected_zones)))
                else :
                    objects = [o for o in objects if o in range(len(connected_zones))]
                pDictParams['objets'][pDataList[index]][classListObjets_index][1] = objects

                # Boucle sur les objets de chaque classe
                explanationsObjects = []
                for o_index, o in enumerate(objects):

                    # Explication de l'image 'index', classe 'c' et objet 'o'
                    print("Explanation Img. %d/%d Class %d/%d Object %d/%d"%(index + 1, pNbData, classListObjets_index + 1, len(predictedClassesNames), o_index + 1, len(objects)), flush=True)

                    # Explication de chaque objet de la classe
                    target = connected_zones[o]

                    start = time.time()
                    explanation = explainer.explain(my_input, tf.expand_dims(target, axis=0))
                    end = time.time()
                    print("Duration: %s"%str(end - start))

                    explanation = tf.squeeze(explanation, axis=0)

                    # Sauve l'explication de l'objet 'o' de la classe 'c' de la donnée 'index'
                    extension = "--i_%s--c_%s"%(str(o), className)
                    _saveExplanations(pDictParams, explanation, resultSavedOn, index, extension)
                    # Sauve le masque associé à l'explication
                    extension += "--s_mask"
                    _saveExplanations(pDictParams, target, resultSavedOn, index, extension)

                    explanationsObjects.append(explanation)

                # control function for XAI_computeExplanations outputs
                kaasrc.controles.controlXAI_computeExplanationsOutput(pDictParams, outputXAIFramework, explanationsObjects, aiModel.useCase, len(objects), __file__)

                explanationsClasses.extend(explanationsObjects)
        allExplanations.extend(explanationsClasses)

    # save all explanations to use them in launchMetric() function
    _saveExplanations(pDictParams, allExplanations, resultSavedOn)

    # Sauvegarde de la liste des objets détectés
    pDictParams['objets'] = listObjectExplained
    ficObjets = os.path.join(resultSavedOn, 'listObjects.json')
    with open(ficObjets, "w", encoding="utf-8") as f:
        json.dump(pDictParams['objets'], f, indent=4)

    return pDictParams

# ---------------------------------------------------------------------------
## Compute explanations
# @param pDictParams : parameter dictionary
# @param pClasse : class to explain : extension for filename to identify explanation
# @return parameter dictionary
def computeExplanations(pDictParams, pClasse=None):
    # shortcuts
    pMethod = pDictParams['method']
    pNbData = pDictParams['nbData']
    pRepertProd = pDictParams['repertProd']
    pDataProd = pDictParams['dataProd']
    pModelType = pDictParams['modeltype']
    pMethodParam = pDictParams[pMethod]
    _, aiModel = pDictParams['aiModel']

    inputXAIframework = pDictParams['inputXAIframework']
    outputXAIFramework = pDictParams['outputXAIFramework']

    resultSavedOn = os.path.join(pDataProd, "dataExplanations", pRepertProd)

    # convert data and prediction to XAI framework
    xData, yPred = kaasrc.communs.convertToXAIframework(pDictParams, inputXAIframework)

    # control function for XAI_computeExplanations inputs
    pParamSpecif = {"dataSize": pDictParams['datasize']}
    kaasrc.controles.controlXAI_computeExplanationsInput(pDictParams, inputXAIframework, xData, yPred, aiModel.useCase, pNbData, __file__, pParamSpecif)

    aiModel.setInferenceCible(False)
    print("   .get explainer %s"%pMethod, flush=True)

    if pModelType == "regression":
        operator = regression_operator
    else:
        operator = None
    explainer = getExplainer(aiModel, pMethod, pMethodParam, operator)

    print("==[%s] ============================================================================="%pMethod, flush=True)
    start = time.time()
    explanations = []
    for index in range(pNbData):
        print(" - Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)
        explanation = explainer.explain(xData[index], yPred[index])
        explanations.append(explanation)
    end = time.time()
    print("Duration:", end - start)

    # control function for XAI_computeExplanations outputs
    pParamSpecif = {"dataSize": pDictParams['datasize']}
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
# @param pClasse : class to explain : extension for filename to identify explanation
# @return parameter dictionary
def computeExplanationsImage(pDictParams, pClasse=""):
    # shortcuts
    pMethod = pDictParams['method']
    pNbData = pDictParams['nbData']
    pRepertProd = pDictParams['repertProd']
    pDataProd = pDictParams['dataProd']
    pModelType = pDictParams['modeltype']
    pMethodParam = pDictParams[pMethod]
    _, aiModel = pDictParams['aiModel']

    inputXAIframework = pDictParams['inputXAIframework']
    outputXAIFramework = pDictParams['outputXAIFramework']

    resultSavedOn = os.path.join(pDataProd, "dataExplanations", pRepertProd)

    # convert data and prediction to XAI framework
    xData, yPred = kaasrc.communs.convertToXAIframework(pDictParams, inputXAIframework)

    # control function for XAI_computeExplanations inputs
    kaasrc.controles.controlXAI_computeExplanationsInput(pDictParams, inputXAIframework, xData, yPred, aiModel.useCase, pNbData, __file__)

    aiModel.setInferenceCible(False)
    print("   .get explainer %s"%pMethod, flush=True)

    if pModelType == "regression":
        operator = regression_operator
    else:
        operator = None
    explainer = getExplainer(aiModel, pMethod, pMethodParam, operator)

    print("==[%s] ============================================================================="%pMethod, flush=True)
    start = time.time()
    print(" - Explanation all %d data (%s)"%(pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)
    explanations = explainer.explain(xData, yPred)
    end = time.time()
    print("Duration:", end - start)

    # control function for XAI_computeExplanations outputs
    pParamSpecif = {"explOutputDim": 1}
    kaasrc.controles.controlXAI_computeExplanationsOutput(pDictParams, outputXAIFramework, explanations, aiModel.useCase, pNbData, __file__, pParamSpecif)

    # Sauvegarde : Boucle sur les données
    for index in range(pNbData):
        _saveExplanations(pDictParams, explanations[index], resultSavedOn, index, pClasse)

    # save all explanations to use them in launchMetric() function
    _saveExplanations(pDictParams, explanations, resultSavedOn, pExtension=pClasse)
    return pDictParams

# ---------------------------------------------------------------------------
## Compute explanations
# @param pDictParams : parameter dictionary
# @param pExplainClasses : are classes to be explained (boolean)
# @return parameter dictionary
def computeExplanationsTabText(pDictParams, pExplainClasses):
    # shortcuts
    xData, yPred = pDictParams['xy']
    pClasses = pDictParams['classes']
    pNbData = pDictParams['nbData']

    pDictParams = computeExplanations(pDictParams)
    if pExplainClasses:
        for n, _ in enumerate(pClasses):
            print("Explanations for class %d/%d"%(n + 1, len(pClasses)))
            targets = []
            for index in range(pNbData):
                nbRule = xData[index].shape[0]
                indices = [n] * nbRule
                targets.append(tf.one_hot(indices, len(pClasses)))
            pDictParams['xy'] = xData, targets
            computeExplanations(pDictParams, pClasse=pClasses[n])
        pDictParams['xy'] = xData, yPred
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
# @param pFullData : are all data loaded? (True in metric computation process)
# @param pNumClasses : number of class of the use case
# @return explanations loaded
def loadExplanations(pDictParams, pFicExplanation, pNumData=None, pFullData=None, pNumClasses=None):
    # shortcuts
    _, aiModel = pDictParams['aiModel']
    outputXAIFramework = pDictParams['outputXAIFramework']

    if not os.path.exists(pFicExplanation):
        colPrint("The file containing the explanation '%s' does not exists."%pFicExplanation, "Info")
        return None
    explanations = np.load(pFicExplanation, allow_pickle=True)

    if pDictParams['datatype'] == "tabular":
        # Les explications sont au format (n, 1)
        pParamSpecif = {"dataSize": pDictParams['datasize'][pNumData]}
        kaasrc.controles.control_loadExplanationsOutput(pDictParams, outputXAIFramework, explanations, aiModel.useCase, __file__, pParamSpecif)
    elif pDictParams['modeltype'] == "segmentation":
        # Les explications sont au format (H, W, C)
        pParamSpecif = {"explOutputDim": pNumData, "fullData": pFullData, "nbClasses": pNumClasses}
        kaasrc.controles.control_loadExplanationsOutput(pDictParams, outputXAIFramework, explanations, aiModel.useCase, __file__, pParamSpecif)
    else:
        # Les explications sont au format (H, W, 1) ou (n, H, W, 1)
        pParamSpecif = {"explOutputDim": 1, "fullData": pFullData}
        kaasrc.controles.control_loadExplanationsOutput(pDictParams, outputXAIFramework, explanations, aiModel.useCase, __file__, pParamSpecif)

    return explanations

# ===============================================================================
# end of file
