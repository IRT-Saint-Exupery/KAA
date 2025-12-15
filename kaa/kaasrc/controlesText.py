#!/usr/bin/env python3.,
# -*- coding: utf-8 -*-
# ===============================================================================
# import numpy as np
from kaasrc.communs import colPrint

# -------------------------------------------------------------------------------
## Basis control function
# @param pType : type of control (type vs size of data)
# @param pExpected : expected value
# @param pObtained : obtained value to control
# @param pForm : data schema
# @param pFunction : function in which the control is done
# @param pFile : file in which the control is done
def controle(pType, pExpected, pObtained, pForm, pFunction, pFile):
    if pType:
        strTypeShape = "  TYPE:  "
    else:
        strTypeShape = "  SHAPE: "
    if pObtained == pExpected:
        colPrint("%sOK - %s - %s"%(strTypeShape, str(pExpected), pForm), "Action")
    else:
        colPrint("%sKO"%strTypeShape, "Config")
        colPrint("    expected %s %s"%(str(pExpected), pForm), "Config")
        colPrint("    obtained %s"%str(pObtained), "Config")
        colPrint("  Checked by '%s()' in file: "%pFunction, "Debug")
        colPrint("    %s"%pFile, "Debug")

# -------------------------------------------------------------------------------
## recursive control function
# @param pType : type of control (type vs size of data)
# @param pExpected : expected value
# @param pObtained : obtained value to control
# @param pForm : data schema
# @param pFunction : function in which the control is done
# @param pFile : file in which the control is done
# @param pIter : iteration number
# @param pLevelData : level of data if its list or tuple
# @param pCurLevelData : current level
def controleRecursif(pType, pExpected, pObtained, pForm, pFunction, pFile, pIter=0, pLevelData=None, pCurLevelData=None):
    # sauvegarede du message
    forme = pForm

    # Cas du test TYPE
    if pType:
        if pIter > 0:
            forme = forme + " - level %d"%pIter
        controle(pType, pExpected[0], type(pObtained), forme, pFunction, pFile)
        rForme = pForm
    else:
        # Cas du test SHAPE
        if isinstance(pObtained, dict) or pExpected[0] is None:
            colPrint("   --Cannot control the element size-- ", "Info")
        elif type(pObtained) in [list, tuple]:
            controle(pType, pExpected[0], len(pObtained), pForm[0], pFunction, pFile)
        else:
            controle(pType, pExpected[0], pObtained.shape, pForm[0], pFunction, pFile)
        rForme = pForm[1:]

    if type(pObtained) in [list, tuple]:
        if pLevelData is not None:
            for item, _ in enumerate(pObtained):
                newExpected = pExpected[1:]
                if newExpected[0] is None:
                    colPrint("   --Cannot control the element size-- ", "Info")
                    break
                if pIter == pLevelData[0]:
                    pCurLevelData = item
                if pIter == pLevelData[1] - 1:
                    curData = pExpected[1:][0][pCurLevelData]
                    newExpected = [curData]
                    if len(pExpected) > 2:
                        newExpected.extend(pExpected[2:])
                controleRecursif(pType, newExpected, pObtained[item], rForme, pFunction, pFile, pIter + 1, pLevelData, pCurLevelData)
        else:
            for item in pObtained:
                controleRecursif(pType, pExpected[1:], item, rForme, pFunction, pFile, pIter + 1)

# -------------------------------------------------------------------------------
## Function to control (1)
# @param pData : data to control
# @param pUseCase : class of the usecase plugin
# @param pFile : file in which the control is done
# @param pParamSpecif : specific params in dictionnary
def controlFctPrepareDataOutputText(pData, pUseCase, pFile, pParamSpecif):

    expectedShape = pParamSpecif["expected"]
    levelData = pParamSpecif["levelData"] if "levelData" in pParamSpecif else None
    levelData = None
    formeShape = pParamSpecif["format"]

    if isinstance(pUseCase.inputModelFramework, tuple):
        expected = pUseCase.inputModelFramework
        obtained = pData
        controleRecursif(True, expected, obtained, "(inputModelFramework)", "controlFctPrepareDataOutput", pFile)

        expected = expectedShape
        obtained = pData
        controleRecursif(False, expected, obtained, formeShape, "controlFctPrepareDataOutput", pFile, pLevelData=levelData)
    else:
        expected = pUseCase.inputModelFramework
        obtained = type(pData)
        controle(True, expected, obtained, "(inputModelFramework)", "controlFctPrepareDataOutput", pFile)

        expected = expectedShape
        obtained = pData.shape
        controle(False, expected, obtained, formeShape, "controlFctPrepareDataOutput", pFile)

# -------------------------------------------------------------------------------
## Function to control (2)
# @param pData : data to control
# @param pUseCase : class of the usecase plugin
# @param pFile : file in which the control is done
# @param pParamSpecif : specific params in dictionnary
def controlInferenceInputText(pData, pUseCase, pFile, pParamSpecif):

    expectedShape = pParamSpecif["expected"]
    levelData = pParamSpecif["levelData"] if "levelData" in pParamSpecif else None
    levelData = None
    formeShape = pParamSpecif["format"]

    if isinstance(pUseCase.inputModelFramework, tuple):
        expected = pUseCase.inputModelFramework
        obtained = pData
        controleRecursif(True, expected, obtained, "(inputModelFramework)", "controlInferenceInput", pFile)

        expected = expectedShape
        obtained = pData
        controleRecursif(False, expected, obtained, formeShape, "controlInferenceInput", pFile, pLevelData=levelData)
    else:
        expected = pUseCase.inputModelFramework
        obtained = type(pData)
        controle(True, expected, obtained, "(inputModelFramework)", "controlInferenceInput", pFile)

        expected = expectedShape
        obtained = pData.shape
        controle(False, expected, obtained, formeShape, "controlInferenceInput", pFile)

# ===============================================================================
# end of file
