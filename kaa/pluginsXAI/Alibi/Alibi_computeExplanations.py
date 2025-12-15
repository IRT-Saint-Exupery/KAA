#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================================

import os, json, time
import numpy as np

from kaasrc.communs import colPrint
import kaasrc.communs
import kaasrc.controles

# ---------
try:
    import spacy
    from alibi.explainers import AnchorText
    from alibi.utils import spacy_model
    from alibi.utils import LanguageModel
except Exception as err:
    print("Error:", err)
    colPrint("The 'Alibi' library is not or not properly installed.", "Error")


# ---------------------------------------------------------------------------
## Create an instance of the explainability method
# @param pAiModel : model instance
# @param pDictParams : parameter dictionary
# @return method instance
def _getExplainer(pAiModel, pDictParams):
    # shortcuts
    pMethod = pDictParams['method']
    pMethodParam = pDictParams[pMethod]
    pDatatype = pDictParams['datatype']
    _, aiModel = pDictParams['aiModel']

    if pDatatype == "text" and pMethod == "Anchors" and not hasattr(pAiModel, 'predict_from_text'):
        colPrint("/!\\ The model must implement a method 'predict_proba()'. No calculated metric.", "Error")
        return

    explainer = None

    # Anchors
    if pMethod == "Anchors":
        predict_fn = lambda x: np.array([np.argmax(pAiModel.predict_from_text(xi)) for xi in x])

        sampling_strategy = str(pMethodParam["ss"])
        if sampling_strategy == 'unknown':
            sample_proba = float(pMethodParam["sp"])

            model = str(pMethodParam["sm"])
            spacy_model(model=model)
            nlp = spacy.load(model)

            explainer = AnchorText(
                    predictor=predict_fn,
                    sampling_strategy=sampling_strategy,
                    nlp=nlp,
                    sample_proba=sample_proba)
        elif sampling_strategy == 'similarity':
            sample_proba = float(pMethodParam["sp"])
            top_n = int(pMethodParam["tn"])
            temperature = float(pMethodParam["te"])
            use_proba = bool(pMethodParam["up"])
            model = str(pMethodParam["sm"])
            spacy_model(model=model)
            nlp = spacy.load(model)

            explainer = AnchorText(
                    predictor=predict_fn,
                    sampling_strategy=sampling_strategy,
                    nlp=nlp,
                    sample_proba=sample_proba,
                    top_n=top_n,
                    temperature=temperature,
                    use_proba=use_proba)

        elif sampling_strategy == 'language_model':

            class CamembertBase(LanguageModel):
                SUBWORD_PREFIX = 'G'

                def __init__(self, preloading: bool = True):
                    """
                    Initialize `CamembertBase`.
                    Parameters
                    ----------
                    preloading
                        See :py:meth:`alibi.utils.lang_model.LanguageModel.__init__`.
                    """
                    super().__init__("camembert-base", preloading)

                @property
                def mask(self) -> str:
                    return CamembertBase.SUBWORD_PREFIX + aiModel.useCase.tokenizer.mask_token

                def is_subword_prefix(self, token: str) -> bool:
                    return token.startswith(CamembertBase.SUBWORD_PREFIX)

            sample_proba = float(pMethodParam["sp"])
            top_n = int(pMethodParam["tn"])
            temperature = float(pMethodParam["te"])
            use_proba = bool(pMethodParam["up"])
            filling = str(pMethodParam["f"])
            frac_mask_template = float(pMethodParam["fm"])
            batch_size_lm = int(pMethodParam["bs"])
            punctuation = pMethodParam["p"]
            stopwords = list(pMethodParam["s"])
            sample_punctuation = bool(pMethodParam["spc"])

            explainer = AnchorText(
                    predictor=predict_fn,
                    sampling_strategy=sampling_strategy,
                    language_model=CamembertBase(),
                    sample_proba=sample_proba,
                    top_n=top_n,
                    temperature=temperature,
                    use_proba=use_proba,
                    filling=filling,
                    frac_mask_template=frac_mask_template,
                    batch_size_lm=batch_size_lm,
                    punctuation=punctuation,
                    stopwords=stopwords,
                    sample_punctuation=sample_punctuation)

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
    pClasses = pDictParams['classes']

    resultSavedOn = os.path.join(pDataProd, "dataExplanations", pRepertProd)
    _, aiModel = pDictParams['aiModel']
    xData, yPred = pDictParams['xy']

    # control function for XAI_computeExplanations inputs
    kaasrc.controles.NOcontrolXAI_computeExplanationsInput(pDictParams)

    aiModel.setInferenceCible(False)
    print("   .get explainer %s"%pMethod, flush=True)
    explainer = _getExplainer(aiModel.useCase, pDictParams)

    print("==[%s] ============================================================================="%pMethod, flush=True)
    start = time.time()
    explanations = []
    alternatives = []
    predictions = []
    threshold = float(pMethodParam["th"])
    if pMethod == 'Anchors':
        for index in range(pNbData):
            explanationItems = []
            alternativeItems = []
            predictionItems = []
            npxData = xData[index]
            npyPred = yPred[index]
            for entry, text in enumerate(npxData):
                explanation = explainer.explain(text, threshold)
                explanationItems.append(explanation)
                index_pred = npyPred[entry]
                prediction = pClasses[index_pred]
                predictionItems.append(prediction)
                alternative = pClasses[:index_pred] + pClasses[index_pred + 1:]
                alternativeItems.append(alternative)
            explanations.append(explanationItems)
            alternatives.append(alternativeItems)
            predictions.append(predictionItems)

            kaasrc.controles.NOcontrolXAI_computeExplanationsOutput(pDictParams)

            # Sauvegarde
            _saveExplanation(pDictParams, explanationItems, resultSavedOn, index, predictionItems, alternativeItems)

    end = time.time()
    print("Duration:", end - start)

    return pDictParams

# ---------------------------------------------------------------------------
## Plot explanation on file
# @param pDictParams : parameter dictionary
# @param pExplanations : explanation
# @param pResultSavedOn : path to save on
# @param pIndex : index of the data to treat
# @param pPredictions : predicted classes
# @param pAlternatives : other possible classes
# @param pExtension : extension to identify the explanation file
def _saveExplanation(pDictParams, pExplanations, pResultSavedOn, pIndex, pPredictions, pAlternatives, pExtension=""):
    # shortcuts
    pDataList = pDictParams['dataList']

    ficData = os.path.basename(pDataList[pIndex])
    dirName = os.path.dirname(pDataList[pIndex])
    ficExplanation = kaasrc.communs.createDirName(pResultSavedOn, os.path.join(dirName, '%s%s.json'%(ficData, pExtension)))

    saveExplainJson = {}
    for i, explanation in enumerate(pExplanations):
        dico = None
        if len(explanation.anchor) > 0:
            dico = {"Anchor": explanation.anchor, "Precision": explanation.precision, "Coverage": explanation.coverage,
                  "Predictions": pPredictions[i], "covered_true": [], "uncovered_true": [],
                  "Alternatives": pAlternatives[i], "covered_false": [], "uncovered_false": [],
                  "Success": explanation.raw["success"]}
            if len(explanation.raw['examples']) > 0:
                dico["covered_true"] = explanation.raw['examples'][-1]['covered_true'].tolist()
                dico["covered_false"] = explanation.raw['examples'][-1]['covered_false'].tolist()
                dico["uncovered_true"] = explanation.raw['examples'][-1]['uncovered_true'].tolist()
                dico["uncovered_false"] = explanation.raw['examples'][-1]['uncovered_false'].tolist()

        saveExplainJson[i] = dico

    with open(ficExplanation, "w", encoding="utf-8") as f:
        json.dump(saveExplainJson, f, ensure_ascii=False, indent=4)

    print("    Saved in '%s'"%ficExplanation)

# ---------------------------------------------------------------------------
## Load explanation from file
# @param pFicExplanation : explanation filename to load
# @param pNbData : number of data in explanations to load (to control)
# @return explanations loaded
def loadExplanations(pDictParams, pFicExplanation, pNbData=None):
    if not os.path.exists(pFicExplanation):
        print("     -Le fichier d'explication '%s' n'existe pas."%pFicExplanation, flush=True)
        return None
    with open(pFicExplanation, "r", encoding="utf-8") as f:
        explanations = json.load(f)

    if pNbData is not None and len(explanations) < pNbData:
        print("     -Le fichier d'explication '%s' ne contient pas assez de données: %d vs %d."%(pFicExplanation, len(explanations), pNbData), flush=True)
        return None

    kaasrc.controles.NOcontrol_loadExplanationsOutput(pDictParams)

    return explanations

# ===============================================================================
# end of file
