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
    import torch
except Exception as err:
    print("Error:", err)
    colPrint("Package torch is not or not properly installed.", "Error")
# ---------
try:
    import pickle
except Exception as err:
    print("Error:", err)
    colPrint("Package pickle is not or not properly installed.", "Error")
# ---------
try:
    from sentence_transformers import SentenceTransformer, util
except Exception as err:
    print("Error:", err)
    colPrint("Package transformers is not or not properly installed.", "Error")
# ---------
try:
    from captum.attr import ShapleyValueSampling, Occlusion, LimeBase, KernelShap, FeatureAblation, FeaturePermutation, Lime
    from captum._utils.models.linear_model import SkLearnLinearRegression
except Exception as err:
    print("Error:", err)
    colPrint("The library 'Captum' is not or not properly installed.", "Error")
# ---------

# ---------------------------------------------------------------------------
## Create an instance of the explanability method
# @param pDictParams : parameter dictionary
# @return method instance
def _getExplainer(pDictParams):
    # shortcuts
    pMethod = pDictParams['method']
    pMethodParam = pDictParams[pMethod]
    _, pAiModel = pDictParams['aiModel']
    pDatatype = pDictParams['datatype']

    explainer = None

    # KernelSHAP
    if pMethod == "KernelSHAP":
        if pDatatype == "text":
            explainer = KernelShap(pAiModel.useCase.predict_torchtensor)
        else :
            explainer = KernelShap(pAiModel.useCase)

    # Occlusion
    elif pMethod == "Occlusion":
        if pDatatype == "text":
            explainer = Occlusion(pAiModel.useCase.predict_torchtensor)
        else :
            explainer = Occlusion(pAiModel.useCase)

    # Shapley Value Sampling
    elif pMethod == "ShapleyValueSampling":
        if pDatatype == "text":
            explainer = ShapleyValueSampling(pAiModel.useCase.predict_torchtensor)
        else :
            explainer = ShapleyValueSampling(pAiModel.useCase)

    # FeatureAblation
    elif pMethod == "FeatureAblation":
        if pDatatype == "text":
            explainer = FeatureAblation(pAiModel.useCase.predict_torchtensor)
        else :
            explainer = FeatureAblation(pAiModel.useCase)

    # FeaturePermutation
    elif pMethod == "FeaturePermutation":
        if pDatatype == "text":
            explainer = FeaturePermutation(pAiModel.useCase.predict_torchtensor)
        else :
            explainer = FeaturePermutation(pAiModel.useCase)

    # Lime
    elif pMethod == "Lime":
        explainer = Lime(pAiModel.useCase)

    # LimeBase
    elif pMethod == "LimeBase":
        # nb_pixel = int(pMethodParam["nm"])
        sf = pMethodParam["sf"]
        similarity_function = None
        if sf == "exp_embedding_cosine_distance":
            model_emb = SentenceTransformer("dangvantuan/sentence-camembert-large")
            # - - - - - - - - -
            def exp_embedding_cosine_distance(original_inp, perturbed_inp, _, **kwargs):
                decoded_original = pAiModel.useCase.tokenizer.decode(original_inp[0])
                decoded_perturbed = pAiModel.useCase.tokenizer.decode(perturbed_inp[0])
                original_emb = model_emb.encode(decoded_original, convert_to_tensor=True)
                perturbed_emb = model_emb.encode(decoded_perturbed, convert_to_tensor=True)
                cosine_score = util.cos_sim(original_emb, perturbed_emb)
                distance = 1 - cosine_score
                return torch.exp(-1 * (distance ** 2) / 2)
            # - - - - - - - - -
            similarity_function = exp_embedding_cosine_distance
        else:
            print('This similarity function is not implemented')

        perturbation_function = None
        pf = pMethodParam["pf"]
        if pf == "bernoulli_perturb":
            def bernoulli_perturb(text, **kwargs):
                probs = torch.ones_like(text[:1]) * 0.5
                return torch.bernoulli(probs).long()
            perturbation_function = bernoulli_perturb
        else:
            print("This perturbation function is not implemented")

        ps = pMethodParam["ps"]
        it = str(pMethodParam["it"])
        if it == "interp_to_input":
            def interp_to_input(interp_sample, original_input, **kwargs):
                result_interp = original_input[0][interp_sample[0].bool()]
                mask_list = torch.ones(result_interp.size(), dtype=int)
                result_interp = torch.stack([result_interp, mask_list], 0)
                return result_interp
            from_interp_function = interp_to_input
        else:
            from_interp_function = None
            print("This interp function is not implemented")
        to_interp_function = None
        rp = str(pMethodParam["rp"])
        if rp == "None":
            print("This interp function is not implemented")

        if pDatatype == "text":
            explainer = LimeBase(pAiModel.useCase.predict_torchtensor, interpretable_model=SkLearnLinearRegression(),
                                similarity_func=similarity_function,
                                perturb_func=perturbation_function,
                                perturb_interpretable_space=ps,
                                from_interp_rep_transform=from_interp_function,
                                to_interp_rep_transform=to_interp_function)

    else:
        print("La méthode '%s' n'est pas définie."%pMethod)

    return explainer

# ---------------------------------------------------------------------------
## Compute explanations
# @param pDictParams : parameter dictionary
def computeExplanationsTexts(pDictParams):
    # shortcuts
    pMethod = pDictParams['method']
    pMethodParam = pDictParams[pMethod]
    pNbData = pDictParams['nbData']
    pRepertProd = pDictParams['repertProd']
    pDataProd = pDictParams['dataProd']
    inputXAIframework = pDictParams['inputXAIframework']
    outputXAIFramework = pDictParams['outputXAIFramework']

    resultSavedOn = os.path.join(pDataProd, "dataExplanations", pRepertProd)
    _, aiModel = pDictParams['aiModel']
    xData, yPred = pDictParams['xy']

    kaasrc.controles.controlXAI_computeExplanationsInput(pDictParams, inputXAIframework, xData, yPred, aiModel.useCase, pNbData, __file__)

    aiModel.setInferenceCible(False)
    print("   .get explainer %s"%pMethod, flush=True)
    explainer = _getExplainer(pDictParams)

    print("==[%s] ============================================================================="%pMethod, flush=True)
    start = time.time()

    if pMethod == "KernelSHAP":
        ns = int(pMethodParam["ns"])
        bs = pMethodParam["bs"]
        if isinstance(bs, str):
            bs = None

        nppe = int(pMethodParam["np"])
        ri = bool(pMethodParam["ri"])

        explanations = []
        for index in range(pNbData):
            print(" - Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)

            tg = yPred[index]
            if not isinstance(yPred[index], torch.Tensor):
                tg = torch.Tensor(yPred[index])

            tData = [aiModel.useCase.tokenize_text(data) for data in xData[index]]
            tData = [torch.stack([data['input_ids'], data['attention_mask']]) for data in tData]

            explanation = []
            nbItems = len(tData)
            step = nbItems // 10 if nbItems > 9 else 1
            for i in range(nbItems):
                if i%step == 0:
                    print("       . item %d/%d"%(i + 1, nbItems))
                exp = explainer.attribute(tData[i],
                                            baselines=bs,
                                            target=int(tg[i]),
                                            additional_forward_args=None,
                                            feature_mask=None,
                                            n_samples=ns,
                                            perturbations_per_eval=nppe,
                                            return_input_shape=ri,
                                            show_progress=False)

                explanation.append(exp)
            explanations.append(explanation)

    elif pMethod == "Occlusion":
        wx = int(pMethodParam["wx"])
        # wy = int(pMethodParam["wy"])
        sz = int(pMethodParam["sz"])
        sw = (wx, )
        bs = pMethodParam["bs"]
        if isinstance(bs, str):
            bs = None
        nppe = int(pMethodParam["np"])

        explanations = []
        for index in range(pNbData):
            print(" - Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)

            tg = yPred[index]
            if not isinstance(yPred[index], torch.Tensor):
                tg = torch.Tensor(yPred[index])

            tData = [aiModel.useCase.tokenize_text(data) for data in xData[index]]
            tData = [torch.stack([data['input_ids'], data['attention_mask']]) for data in tData]

            explanation = []
            nbItems = len(tData)
            step = nbItems // 10 if nbItems > 9 else 1
            for i in range(nbItems):
                if i%step == 0:
                    print("       . item %d/%d"%(i + 1, nbItems))
                exp = explainer.attribute(tData[i],
                                            sliding_window_shapes=sw,
                                            strides=sz,
                                            baselines=bs,
                                            target=int(tg[i]),
                                            additional_forward_args=None,
                                            perturbations_per_eval=nppe,
                                            show_progress=False)

                explanation.append(exp)
            explanations.append(explanation)

    elif pMethod == "ShapleyValueSampling":
        ns = int(pMethodParam["ns"])
        bs = pMethodParam["bs"]
        if isinstance(bs, str):
            bs = None

        nppe = int(pMethodParam["np"])

        explanations = []
        for index in range(pNbData):
            print(" - Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)

            tg = yPred[index]
            if not isinstance(yPred[index], torch.Tensor):
                tg = torch.Tensor(yPred[index])

            tData = [aiModel.useCase.tokenize_text(data) for data in xData[index]]
            tData = [torch.stack([data['input_ids'], data['attention_mask']]) for data in tData]

            explanation = []
            nbItems = len(tData)
            step = nbItems // 10 if nbItems > 9 else 1
            for i in range(nbItems):
                if i%step == 0:
                    print("       . item %d/%d"%(i + 1, nbItems))
                exp = explainer.attribute(tData[i],
                                            baselines=bs,
                                            target=int(tg[i]),
                                            additional_forward_args=None,
                                            feature_mask=None,
                                            n_samples=ns,
                                            perturbations_per_eval=nppe,
                                            show_progress=False)

                explanation.append(exp)
            explanations.append(explanation)

    elif pMethod == "LimeBase":

        ns = int(pMethodParam["ns"])
        nppe = int(pMethodParam["np"])
        explanations = []
        for index in range(pNbData):
            print(" - Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)

            tg = yPred[index]
            if not isinstance(yPred[index], torch.Tensor):
                tg = torch.Tensor(yPred[index])

            tData = [aiModel.useCase.tokenize_text(data) for data in xData[index]]
            tData = [torch.stack([data['input_ids'], data['attention_mask']]) for data in tData]

            explanation = []
            nbItems = len(tData)
            step = nbItems // 10 if nbItems > 9 else 1
            for i in range(nbItems):
                if i%step == 0:
                    print("       . item %d/%d"%(i + 1, nbItems))
                exp = explainer.attribute(tData[i],
                                            target=int(tg[i]),
                                            additional_forward_args=None,
                                            n_samples=ns,
                                            perturbations_per_eval=nppe,
                                            show_progress=False)

                explanation.append(exp)
            explanations.append(explanation)

    elif pMethod == "FeatureAblation":
        nppe = int(pMethodParam["np"])
        bs = pMethodParam["bs"]
        if isinstance(bs, str):
            bs = None

        explanations = []
        for index in range(pNbData):
            print(" - Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)

            tg = yPred[index]
            if not isinstance(yPred[index], torch.Tensor):
                tg = torch.Tensor(yPred[index])

            tData = [aiModel.useCase.tokenize_text(data) for data in xData[index]]
            tData = [torch.stack([data['input_ids'], data['attention_mask']]) for data in tData]

            explanation = []
            nbItems = len(tData)
            step = nbItems // 10 if nbItems > 9 else 1
            for i in range(nbItems):
                if i%step == 0:
                    print("       . item %d/%d"%(i + 1, nbItems))
                exp = explainer.attribute(tData[i],
                                            baselines=bs,
                                            target=int(tg[i]),
                                            feature_mask=None,
                                            perturbations_per_eval=nppe,
                                            show_progress=False)

                explanation.append(exp)
            explanations.append(explanation)

    elif pMethod == "FeaturePermutation":
        nppe = int(pMethodParam["np"])

        explanations = []
        for index in range(pNbData):
            print(" - Explanation data %d/%d (%s)"%(index + 1, pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)

            tg = yPred[index]
            if not isinstance(yPred[index], torch.Tensor):
                tg = torch.Tensor(yPred[index])

            tData = [aiModel.useCase.tokenize_text(data) for data in xData[index]]
            tData = [torch.stack([data['input_ids'], data['attention_mask']]) for data in tData]

            explanation = []
            nbItems = len(tData)
            step = nbItems // 10 if nbItems > 9 else 1
            for i in range(nbItems):
                if i%step == 0:
                    print("       . item %d/%d"%(i + 1, nbItems))
                exp = explainer.attribute(tData[i],
                                            target=int(tg[i]),
                                            feature_mask=None,
                                            perturbations_per_eval=nppe,
                                            show_progress=False)

                explanation.append(exp)
            explanations.append(explanation)

    end = time.time()
    print("Duration:", end - start)

    # control function for XAI_computeExplanations outputs
    kaasrc.controles.controlXAI_computeExplanationsOutput(pDictParams, outputXAIFramework, explanations, aiModel.useCase, pNbData, __file__)

    # Sauvegarde : Boucle sur les données
    for index in range(pNbData):
        _saveExplanations(pDictParams, explanations[index], resultSavedOn, index, pMinMax=False)

    # save all explanations to use them in launchMetric() function
    _saveExplanations(pDictParams, explanations, resultSavedOn)

    return pDictParams

# ---------------------------------------------------------------------------
## Compute explanations
# @param pDictParams : parameter dictionary
def computeExplanationsImages(pDictParams):
    # shortcuts
    pMethod = pDictParams['method']
    pNbData = pDictParams['nbData']
    pRepertProd = pDictParams['repertProd']
    pDataProd = pDictParams['dataProd']
    pMethodParam = pDictParams[pMethod]
    xData, yPred = pDictParams['xy']
    _, aiModel = pDictParams['aiModel']
    inputXAIframework = pDictParams['inputXAIframework']
    outputXAIFramework = pDictParams['outputXAIFramework']

    resultSavedOn = os.path.join(pDataProd, "dataExplanations", pRepertProd)

    xData, yPred = kaasrc.communs.convertToXAIframework(pDictParams, inputXAIframework)

    pDictParams['xy'] = xData, yPred
    xDataDim = xData[0].shape[0]

    kaasrc.controles.controlXAI_computeExplanationsInput(pDictParams, inputXAIframework, xData, yPred, aiModel.useCase, pNbData, __file__)

    aiModel.setInferenceCible(False)
    print("   .get explainer %s"%pMethod, flush=True)
    explainer = _getExplainer(pDictParams)
    explanations = None

    print("==[%s] ============================================================================="%pMethod, flush=True)
    print(" - Explanation all %d data (%s)"%(pNbData, str(datetime.datetime.now().strftime("%H:%M:%S"))), flush=True)
    start = time.time()

    if pMethod == "KernelSHAP":
        ns = int(pMethodParam["ns"])
        bs = pMethodParam["bs"]
        if isinstance(bs, str):
            bs = None

        mp = pMethodParam["mp"]
        if mp == "Watershed":
            sl = int(pMethodParam["sl"])
            pt = pMethodParam["pt"]
            ny = int(pMethodParam["ny"])
            aiModel.megaPixelsParametres.update({"mapPixels": 0, "seuil": sl, "pretraitement": pt, "noyau": ny})
        else:
            nm = int(pMethodParam["nm"])
            aiModel.megaPixelsParametres["mapPixels"] = nm
        segmentations = np.array([aiModel.megaPixels(vXData) for _, vXData in enumerate(xData)]).astype(int)
        feature_mask = torch.tensor(np.repeat(np.expand_dims(segmentations, axis=1), xDataDim, axis=1))

        nppe = int(pMethodParam["np"])
        ri = bool(pMethodParam["ri"])

        tg = torch.argmax(torch.Tensor(yPred), dim=1)
        explanations = explainer.attribute(xData,
                                            baselines=bs,
                                            target=tg,
                                            additional_forward_args=None,
                                            feature_mask=feature_mask,
                                            n_samples=ns,
                                            perturbations_per_eval=nppe,
                                            return_input_shape=ri,
                                            show_progress=False)

    elif pMethod == "Occlusion":
        wx = int(pMethodParam["wx"])
        wy = int(pMethodParam["wy"])
        sw = (xDataDim, wx, wy)
        sz = int(pMethodParam["sz"])
        bs = pMethodParam["bs"]
        if isinstance(bs, str):
            bs = None
        nppe = int(pMethodParam["np"])
        tg = torch.argmax(torch.Tensor(yPred), dim=1)
        explanations = explainer.attribute(xData,
                                           sliding_window_shapes=sw,
                                           strides=sz,
                                           baselines=bs,
                                           target=tg,
                                           additional_forward_args=None,
                                           perturbations_per_eval=nppe,
                                           show_progress=False)

    elif pMethod == "ShapleyValueSampling":
        ns = int(pMethodParam["ns"])
        bs = pMethodParam["bs"]
        if isinstance(bs, str):
            bs = None
        nppe = int(pMethodParam["np"])
        mp = pMethodParam["mp"]
        if mp == "Watershed":
            sl = int(pMethodParam["sl"])
            pt = pMethodParam["pt"]
            ny = int(pMethodParam["ny"])
            aiModel.megaPixelsParametres.update({"mapPixels": 0, "seuil": sl, "pretraitement": pt, "noyau": ny})
        else:
            nm = int(pMethodParam["nm"])
            aiModel.megaPixelsParametres["mapPixels"] = nm
        segmentations = np.array([aiModel.megaPixels(vXData) for _, vXData in enumerate(xData)]).astype(int)
        feature_mask = torch.tensor(np.repeat(np.expand_dims(segmentations, axis=1), xDataDim, axis=1))

        tg = torch.argmax(torch.Tensor(yPred), dim=1)
        explanations = explainer.attribute(xData,
                                           baselines=bs,
                                           target=tg,
                                           additional_forward_args=None,
                                           feature_mask=feature_mask,
                                           n_samples=ns,
                                           perturbations_per_eval=nppe,
                                           show_progress=False)

    elif pMethod == "Lime":
        ns = int(pMethodParam["ns"])
        nppe = int(pMethodParam["np"])
        mp = pMethodParam["mp"]
        if mp == "Watershed":
            sl = int(pMethodParam["sl"])
            pt = pMethodParam["pt"]
            ny = int(pMethodParam["ny"])
            aiModel.megaPixelsParametres.update({"mapPixels": 0, "seuil": sl, "pretraitement": pt, "noyau": ny})
        else:
            nm = int(pMethodParam["nm"])
            aiModel.megaPixelsParametres["mapPixels"] = nm
        segmentations = np.array([aiModel.megaPixels(vXData) for _, vXData in enumerate(xData)]).astype(int)
        feature_mask = torch.tensor(np.repeat(np.expand_dims(segmentations, axis=1), xDataDim, axis=1))

        tg = torch.argmax(torch.Tensor(yPred), dim=1)
        explanations = explainer.attribute(xData,
                                           target=tg,
                                           feature_mask=feature_mask,
                                           additional_forward_args=None,
                                           n_samples=ns,
                                           perturbations_per_eval=nppe,
                                           show_progress=False)

    elif pMethod == "FeatureAblation":
        nppe = int(pMethodParam["np"])
        bs = pMethodParam["bs"]
        if isinstance(bs, str):
            bs = None

        mp = pMethodParam["mp"]
        if mp == "Watershed":
            sl = int(pMethodParam["sl"])
            pt = pMethodParam["pt"]
            ny = int(pMethodParam["ny"])
            aiModel.megaPixelsParametres.update({"mapPixels": 0, "seuil": sl, "pretraitement": pt, "noyau": ny})
        else:
            nm = int(pMethodParam["nm"])
            aiModel.megaPixelsParametres["mapPixels"] = nm
        segmentations = np.array([aiModel.megaPixels(vXData) for _, vXData in enumerate(xData)]).astype(int)
        feature_mask = torch.tensor(np.repeat(np.expand_dims(segmentations, axis=1), xDataDim, axis=1))

        tg = torch.argmax(torch.Tensor(yPred), dim=1)
        explanations = explainer.attribute(xData,
                                          baselines=bs,
                                          target=tg,
                                          additional_forward_args=None,
                                          feature_mask=feature_mask,
                                          perturbations_per_eval=nppe,
                                          show_progress=False)

    elif pMethod == "FeaturePermutation":
        nppe = int(pMethodParam["np"])

        nm = int(pMethodParam["nm"])
        aiModel.megaPixelsParametres["mapPixels"] = nm
        segmentations = np.array([aiModel.megaPixels(vXData) for _, vXData in enumerate(xData)]).astype(int)
        feature_mask = torch.tensor([np.repeat(np.expand_dims(segmentations, axis=1), xDataDim, axis=1)[0]])

        tg = torch.argmax(torch.Tensor(yPred), dim=1)
        explanations = explainer.attribute(xData,
                                         target=tg,
                                         feature_mask=feature_mask,
                                         perturbations_per_eval=nppe,
                                         show_progress=False)

    end = time.time()
    print("Duration:", end - start)

    # control function for XAI_computeExplanations outputs
    kaasrc.controles.controlXAI_computeExplanationsOutput(pDictParams, outputXAIFramework, explanations, aiModel.useCase, pNbData, __file__)

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
        ficExplanation = kaasrc.communs.createDirName(pResultSavedOn, os.path.join(dirName, '%s%s.json'%(ficData, pExtension)))

        if pMinMax:
            minMaxJson = {"min": str(torch.min(pExplanation).item()), "max": str(torch.max(pExplanation).item())}
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
# @return explanations loaded
def loadExplanations(pDictParams, pFicExplanation):
    # shortcuts
    _, aiModel = pDictParams['aiModel']
    outputXAIFramework = pDictParams['outputXAIFramework']

    if not os.path.exists(pFicExplanation):
        colPrint("The file containing the explanation '%s' does not exists."%pFicExplanation, "Error")
        return None
    with open(pFicExplanation, 'rb') as fid:
        explanations = pickle.load(fid)

    kaasrc.controles.control_loadExplanationsOutput(pDictParams, outputXAIFramework, explanations, aiModel.useCase, __file__)

    return explanations

# ===============================================================================
# end of file
