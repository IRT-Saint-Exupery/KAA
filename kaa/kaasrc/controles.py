#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================================
# import numpy as np
from kaasrc.communs import colPrint

import kaasrc.controlesImage, kaasrc.controlesObject, kaasrc.controlesSegment, kaasrc.controlesTabular, kaasrc.controlesText

# -------------------------------------------------------------------------------
## Function to control fctPrepareData output
# @param pDictParams : parameter dictionary
# @param pData : data to control
# @param pUseCase : class of the usecase plugin
# @param pFile : file in which the control is done
# @param pParamSpecif : specific params in dictionnary
def controlFctPrepareDataOutput(pDictParams, pData, pUseCase, pFile, pParamSpecif=None):
    if pParamSpecif is None:
        pParamSpecif = {}
    if not pDictParams['controls']:
        return
    # shortcuts
    pDataType = pDictParams['datatype']
    pLibrary = pDictParams['library']
    pMethod = pDictParams['method']

    colPrint(" ============================================ ", "Normal")
    colPrint("Control (1) fctPrepareData outputs  --  %s: %s"%(pLibrary, pMethod), "Normal")

    if pDataType == "image":
        kaasrc.controlesImage.controlFctPrepareDataOutputImage(pData, pUseCase, pFile)

    elif pDataType == "tabular":
        kaasrc.controlesTabular.controlFctPrepareDataOutputTabular(pData, pUseCase, pFile, pParamSpecif)

    elif pDataType == "text":
        # kaasrc.controlesText.controlFctPrepareDataOutputText(pData, pUseCase, pFile, pParamSpecif)
        pass

    else:
        colPrint("  KO  --Cannot control the data-- ", "Select")

    colPrint(" ============================================ ", "Normal")

# -------------------------------------------------------------------------------
## Function to control Inference output
# @param pDictParams : parameter dictionary
# @param pData : data to control
# @param pUseCase : class of the usecase plugin
# @param pFile : file in which the control is done
# @param pParamSpecif : specific params in dictionnary
def controlInferenceInput(pDictParams, pData, pUseCase, pFile, pParamSpecif=None):
    if pParamSpecif is None:
        pParamSpecif = {}
    if not pDictParams['controls']:
        return
    # shortcuts
    pDataType = pDictParams['datatype']
    pLibrary = pDictParams['library']
    pMethod = pDictParams['method']

    colPrint(" ============================================ ", "Normal")
    colPrint("Control (2) Inference input  --  %s: %s"%(pLibrary, pMethod), "Normal")

    if pDataType == "image":
        kaasrc.controlesImage.controlInferenceInputImage(pData, pUseCase, pFile)

    elif pDataType == "tabular":
        kaasrc.controlesTabular.controlInferenceInputTabular(pData, pUseCase, pFile, pParamSpecif)

    elif pDataType == "text":
        # kaasrc.controlesText.controlInferenceInputText(pData, pUseCase, pFile, pParamSpecif)
        pass

    else:
        colPrint("  KO  --Cannot control the data-- ", "Select")

    colPrint(" ============================================ ", "Normal")

# -------------------------------------------------------------------------------
## Function to control Inference output
# @param pDictParams : parameter dictionary
# @param pData : data to control
# @param pUseCase : class of the usecase plugin
# @param pFile : file in which the control is done
# @param pParamSpecif : specific params in dictionnary
def controlInferenceOutput(pDictParams, pData, pUseCase, pFile, pParamSpecif=None):
    if pParamSpecif is None:
        pParamSpecif = {}
    if not pDictParams['controls']:
        return
    # shortcuts
    pDataType = pDictParams['datatype']
    pLibrary = pDictParams['library']
    pMethod = pDictParams['method']

    colPrint(" ============================================ ", "Normal")
    colPrint("Control (3) Inference outputs  --  %s: %s"%(pLibrary, pMethod), "Normal")

    if pDataType == "image":
        kaasrc.controlesImage.controlInferenceOutputImage(pData, pUseCase, pFile, pParamSpecif)

    elif pDataType == "tabular":
        kaasrc.controlesTabular.controlInferenceOutputTabular(pData, pUseCase, pFile, pParamSpecif)

    elif pDataType == "text":
        pass

    else:
        colPrint("  KO  --Cannot control the data-- ", "Select")

    colPrint(" ============================================ ", "Normal")

# -------------------------------------------------------------------------------
## Function to control UC_computeInference output
# @param pDictParams : parameter dictionary
# @param pXData : input data to control
# @param pYPred : prediction to control
# @param pUseCase : class of the usecase plugin
# @param pNbData : number of data to treat
# @param pFile : file in which the control is done
# @param pParamSpecif : specific params in dictionnary
def controlUC_computeInferenceOutput(pDictParams, pXData, pYPred, pUseCase, pNbData, pFile, pParamSpecif=None):
    if pParamSpecif is None:
        pParamSpecif = {}
    if not pDictParams['controls']:
        return
    # shortcuts
    pDataType = pDictParams['datatype']
    pModelType = pDictParams['modeltype']
    pLibrary = pDictParams['library']
    pMethod = pDictParams['method']

    colPrint(" ============================================ ", "Normal")
    colPrint("Control (4) UC_computeInference outputs  --  %s: %s"%(pLibrary, pMethod), "Normal")

    if pDataType == "image":
        if pModelType == "detection":
            kaasrc.controlesObject.controlUC_computeInferenceOutputObject(pXData, pYPred, pUseCase, pNbData, pFile, pParamSpecif)
        elif pModelType == "segmentation":
            kaasrc.controlesSegment.controlUC_computeInferenceOutputSegment(pXData, pYPred, pUseCase, pNbData, pFile)
        else:
            kaasrc.controlesImage.controlUC_computeInferenceOutputImage(pXData, pYPred, pUseCase, pNbData, pFile)

    elif pDataType == "tabular":
        kaasrc.controlesTabular.controlUC_computeInferenceOutputTabular(pXData, pYPred, pUseCase, pNbData, pFile, pParamSpecif)

    elif pDataType == "text":
        # kaasrc.controlesText.controlUC_computeInferenceOutputText(pXData, pYPred, pUseCase, pNbData, pFile)
        pass
    else:
        colPrint("  KO  --Cannot control the data-- ", "Select")

    colPrint(" ============================================ ", "Normal")

# -------------------------------------------------------------------------------
## Function to NO control XAI_computeExplanations input
# @param pDictParams : parameter dictionary
def NOcontrolXAI_computeExplanationsInput(pDictParams):
    if not pDictParams['controls']:
        return
    # shortcuts
    pLibrary = pDictParams['library']
    pMethod = pDictParams['method']

    colPrint(" ============================================ ", "Normal")
    colPrint("Control (5) XAI_computeExplanations inputs  --  %s: %s"%(pLibrary, pMethod), "Normal")
    colPrint("    No control available for %s %s method."%(pLibrary,pMethod), "Config")

# -------------------------------------------------------------------------------
## Function to control XAI_computeExplanations input
# @param pDictParams : parameter dictionary
# @param pInputXAIframework : framework of the method input
# @param pXData : input data to control
# @param pYPred : prediction to control
# @param pUseCase : class of the usecase plugin
# @param pNbData : number of data to treat
# @param pFile : file in which the control is done
# @param pParamSpecif : specific params in dictionnary
def controlXAI_computeExplanationsInput(pDictParams, pInputXAIframework, pXData, pYPred, pUseCase, pNbData, pFile, pParamSpecif=None):
    if pParamSpecif is None:
        pParamSpecif = {}
    if not pDictParams['controls']:
        return
    # shortcuts
    pDataType = pDictParams['datatype']
    pModelType = pDictParams['modeltype']
    pLibrary = pDictParams['library']
    pMethod = pDictParams['method']

    colPrint(" ============================================ ", "Normal")
    colPrint("Control (5) XAI_computeExplanations inputs  --  %s: %s"%(pLibrary, pMethod), "Normal")

    pParamSpecif["initFormat1"] = True
    if pDataType == "image":
        if pModelType == "detection":
            kaasrc.controlesObject.controlXAI_computeExplanationsInputObject(pInputXAIframework, pXData, pYPred, pUseCase, pNbData, pFile, pParamSpecif)
        elif pModelType == "segmentation":
            kaasrc.controlesSegment.controlXAI_computeExplanationsInputSegment(pInputXAIframework, pXData, pYPred, pUseCase, pNbData, pFile, pParamSpecif)
        else:
            kaasrc.controlesImage.controlXAI_computeExplanationsInputImage(pInputXAIframework, pXData, pYPred, pUseCase, pNbData, pFile)

    elif pDataType == "tabular":
        kaasrc.controlesTabular.controlXAI_computeExplanationsInputTabular(pInputXAIframework, pXData, pYPred, pUseCase, pNbData, pFile, pParamSpecif)

    elif pDataType == "text":
        pass

    else:
        colPrint("  KO  --Cannot control the data-- ", "Select")

    colPrint(" ============================================ ", "Normal")

# -------------------------------------------------------------------------------
## Function to control XAI_computeMetrics input
# @param pDictParams : parameter dictionary
# @param pInputXAIframework : framework of the method input
# @param pXData : input data to control
# @param pYPred : prediction to control
# @param pUseCase : class of the usecase plugin
# @param pNbData : number of data to treat
# @param pFile : file in which the control is done
# @param pParamSpecif : specific params in dictionnary
def controlXAI_computeMetricsInput(pDictParams, pInputXAIframework, pXData, pYPred, pUseCase, pNbData, pFile, pParamSpecif=None):
    if pParamSpecif is None:
        pParamSpecif = {}
    if not pDictParams['controls']:
        return
    # shortcuts
    pDataType = pDictParams['datatype']
    pModelType = pDictParams['modeltype']
    pLibrary = pDictParams['library']

    colPrint(" ============================================ ", "Normal")
    colPrint("Control (5) XAI_computeMetrics inputs  --  %s"%pLibrary, "Normal")

    pParamSpecif["initFormat1"] = True
    if pDataType == "image":
        if pModelType == "detection":
            kaasrc.controlesObject.controlXAI_computeExplanationsInputObject(pInputXAIframework, pXData, pYPred, pUseCase, pNbData, pFile, pParamSpecif)
        elif pModelType == "segmentation":
            pParamSpecif["initFormat1"] = False
            kaasrc.controlesSegment.controlXAI_computeExplanationsInputSegment(pInputXAIframework, pXData, pYPred, pUseCase, pNbData, pFile, pParamSpecif)
        else:
            kaasrc.controlesImage.controlXAI_computeExplanationsInputImage(pInputXAIframework, pXData, pYPred, pUseCase, pNbData, pFile)

    elif pDataType == "tabular":
        kaasrc.controlesTabular.controlXAI_computeExplanationsInputTabular(pInputXAIframework, pXData, pYPred, pUseCase, pNbData, pFile, pParamSpecif)

    elif pDataType == "text":
        pass

    else:
        colPrint("  KO  --Cannot control the data-- ", "Select")

    colPrint(" ============================================ ", "Normal")

# -------------------------------------------------------------------------------
## Function to NO control XAI_computeExplanations output
# @param pDictParams : parameter dictionary
def NOcontrolXAI_computeExplanationsOutput(pDictParams):
    if not pDictParams['controls']:
        return
    # shortcuts
    pLibrary = pDictParams['library']
    pMethod = pDictParams['method']

    colPrint(" ============================================ ", "Normal")
    colPrint("Control (6) XAI_computeExplanations outputs  --  %s: %s"%(pLibrary, pMethod), "Normal")
    colPrint("    No control available for %s %s method."%(pLibrary,pMethod), "Config")

# -------------------------------------------------------------------------------
## Function to control XAI_computeExplanations output
# @param pDictParams : parameter dictionary
# @param pOutputXAIFramework : framework of the method output
# @param pExplanation : explanation to control
# @param pUseCase : class of the usecase plugin
# @param pNbData : number of data to treat
# @param pFile : file in which the control is done
# @param pParamSpecif : specific params in dictionnary
def controlXAI_computeExplanationsOutput(pDictParams, pOutputXAIFramework, pExplanation, pUseCase, pNbData, pFile, pParamSpecif=None):
    if pParamSpecif is None:
        pParamSpecif = {}
    if not pDictParams['controls']:
        return
    # shortcuts
    pDataType = pDictParams['datatype']
    pModelType = pDictParams['modeltype']
    pLibrary = pDictParams['library']
    pMethod = pDictParams['method']

    colPrint(" ============================================ ", "Normal")
    colPrint("Control (6) XAI_computeExplanations outputs  --  %s: %s"%(pLibrary, pMethod), "Normal")

    if pDataType == "image":
        if pModelType == "detection":
            kaasrc.controlesObject.controlXAI_computeExplanationsOutputObject(pOutputXAIFramework, pExplanation, pUseCase, pNbData, pFile)
        elif pModelType == "segmentation":
            kaasrc.controlesSegment.controlXAI_computeExplanationsOutputSegment(pOutputXAIFramework, pExplanation, pUseCase, pNbData, pFile, pParamSpecif)
        else:
            kaasrc.controlesImage.controlXAI_computeExplanationsOutputImage(pOutputXAIFramework, pExplanation, pUseCase, pNbData, pFile, pParamSpecif)

    elif pDataType == "tabular":
        kaasrc.controlesTabular.controlXAI_computeExplanationsOutputTabular(pOutputXAIFramework, pExplanation, pUseCase, pNbData, pFile, pParamSpecif)

    elif pDataType == "text":
        pass

    else:
        colPrint("  KO  --Cannot control the data-- ", "Select")

    colPrint(" ============================================ ", "Normal")

# -------------------------------------------------------------------------------
## Function to NO control _loadExplanations output
# @param pDictParams : parameter dictionary
def NOcontrol_loadExplanationsOutput(pDictParams):
    if not pDictParams['controls']:
        return
    # shortcuts
    pLibrary = pDictParams['library']
    pMethod = pDictParams['method']

    colPrint(" ============================================ ", "Normal")
    colPrint("Control (7) _loadExplanations outputs  --  %s: %s"%(pLibrary, pMethod), "Normal")
    colPrint("    No control available for %s %s method."%(pLibrary,pMethod), "Config")

# -------------------------------------------------------------------------------
## Function to control _loadExplanations output
# @param pDictParams : parameter dictionary
# @param pOutputXAIFramework : framework of the method output
# @param pExplanation : explanation to control
# @param pUseCase : class of the usecase plugin
# @param pFile : file in which the control is done
# @param pParamSpecif : specific params in dictionnary
def control_loadExplanationsOutput(pDictParams, pOutputXAIFramework, pExplanation, pUseCase, pFile, pParamSpecif=None):
    if pParamSpecif is None:
        pParamSpecif = {}
    if not pDictParams['controls']:
        return
    # shortcuts
    pDataType = pDictParams['datatype']
    # pModelType = pDictParams['modeltype']
    pLibrary = pDictParams['library']
    pMethod = pDictParams['method']

    colPrint(" ============================================ ", "Normal")
    colPrint("Control (7) _loadExplanations outputs  --  %s: %s"%(pLibrary, pMethod), "Normal")

    if pDataType == "image":
        kaasrc.controlesImage.control_loadExplanationsOutputImage(pOutputXAIFramework, pExplanation, pUseCase, pFile, pParamSpecif)

    elif pDataType == "tabular":
        kaasrc.controlesTabular.control_loadExplanationsOutputTabular(pOutputXAIFramework, pExplanation, pUseCase, pFile, pParamSpecif)

    elif pDataType == "text":
        pass

    else:
        colPrint("  KO  --Cannot control the data-- ", "Select")

    colPrint(" ============================================ ", "Normal")

# -------------------------------------------------------------------------------
## Function to NO control _plotExplanationsImages intput
# @param pDictParams : parameter dictionary
def NOcontrol_plotExplanationInput(pDictParams):
    if not pDictParams['controls']:
        return
    # shortcuts
    pLibrary = pDictParams['library']
    pMethod = pDictParams['method']

    colPrint(" ============================================ ", "Normal")
    colPrint("Control (8) _plotExplanations inputs  --  %s: %s"%(pLibrary, pMethod), "Normal")
    colPrint("    No control available for %s %s method."%(pLibrary,pMethod), "Config")

# -------------------------------------------------------------------------------
## Function to control _plotExplanationsImages intput
# @param pDictParams : parameter dictionary
# @param pOutputXAIFramework : framework of the method output
# @param pPlotDataSize : data size when plotting
# @param pExplanation : explanation to control
# @param pData : data to control
# @param pUseCase : class of the usecase plugin
# @param pFile : file in which the control is done
# @param pParamSpecif : specific params in dictionnary
def control_plotExplanationInput(pDictParams, pOutputXAIFramework, pPlotDataSize, pExplanation, pData, pUseCase, pFile, pParamSpecif=None):
    if pParamSpecif is None:
        pParamSpecif = {}
    if not pDictParams['controls']:
        return
    # shortcuts
    pDataType = pDictParams['datatype']
    pLibrary = pDictParams['library']
    pMethod = pDictParams['method']

    colPrint(" ============================================ ", "Normal")
    colPrint("Control (8) _plotExplanations inputs  --  %s: %s"%(pLibrary, pMethod), "Normal")

    if pDataType == "image":
        kaasrc.controlesImage.control_plotExplanationInputImage(pOutputXAIFramework, pPlotDataSize, pExplanation, pData, pUseCase, pFile, pParamSpecif)

    elif pDataType == "tabular":
        kaasrc.controlesTabular.control_plotExplanationOneTable(pOutputXAIFramework, pExplanation, pUseCase, pFile, pParamSpecif)

    elif pDataType == "text":
        pass

    else:
        colPrint("  KO  --Cannot control the data-- ", "Select")

    colPrint(" ============================================ ", "Normal")

# ===============================================================================
# end of file
