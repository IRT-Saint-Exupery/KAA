#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================================

import os, sys, json
from kaasrc.communs import colPrint
import kaasrc.reports

KAAVERSION = 'v25.12'

# -------------------------------------------------------------------------------
## Function that check if a class has a member function
# @param pClass : the class to inspect
# @param pFunction : the function to check
def hasMethod(pClass, pFunction):
    return callable(getattr(pClass, pFunction, None))

# -------------------------------------------------------------------------------
## Recursive function that collect file in directory
# @param pRacine : main directory
# @param pRepert : sub - directory
# @param pDataExtension : file extension
# @param pFiltre : filename filter
# @param pSousfic : file collection
# @param pNivmax : sublevel to stop
# @param pNivcour : current level
# @return list of filenames
def contenuRep(pRacine, pRepert, pDataExtension, pFiltre, pSousfic=None, pNivmax=0, pNivcour=1):
    if pSousfic is None:
        pSousfic = []
    # trouver le contenu du répertoire rep
    entrees = os.listdir(os.path.join(pRacine, pRepert))
    sousrep = []  # recevra les noms des sous - répertoires
    for entree in entrees:
        if os.path.isdir(os.path.join(pRacine, pRepert, entree)):
            sousrep.append(entree)
        else:
            fic, ext = os.path.splitext(entree)
            # print(pRepert, fic, ext, pDataExtension)
            if ext == pDataExtension:
                if pFiltre is None:
                    pSousfic.append(os.path.join(pRepert, fic))
                elif fic.find(pFiltre) != -1:
                    pSousfic.append(os.path.join(pRepert, fic))

    if pNivmax == 0 or pNivcour < pNivmax:
        for srep in sousrep:
            pSousfic = contenuRep(pRacine, os.path.join(pRepert, srep), pDataExtension, pFiltre, pSousfic, pNivmax, pNivcour + 1)
    return pSousfic

# -------------------------------------------------------------------------------
## function that collect file in directory and sub - directory
# @param pUseCaseBase : value of repository of the use cases
# @param pDataset : name of the dataset
# @param pDataExtension : file extension
# @param pFiltre : filename filter
# @return list of filenames
def recupDataset(pUseCaseBase, pDataset, pDataExtension, pFiltre=None):
    lData = []
    for e in pDataset:
        lData.extend(contenuRep(os.path.join(pUseCaseBase, e), "", ".%s"%pDataExtension, pFiltre))
    return lData

# -------------------------------------------------------------------------------
## Kaa initialisation
# @param pDictParams : parameter dictionary
# @param pScripDir : launch directory (pwd)
# @return parameter dictionary updated
def initialisation(pDictParams, pScripDir):
    # shortcuts
    pMethodToUse = pDictParams['method']
    pDataListMode = pDictParams['dataListMode']
    pDataListFichier = pDictParams['dataListFichier']
    pNbData = pDictParams['nbData']
    pOffset = pDictParams['offset']
    pStepData = pDictParams['stepData']
    pDictParams['scriptDir'] = pScripDir

    # get configuration elements from <useCase > .json
    ficJson = os.path.join(pDictParams['scriptDir'], pDictParams['useCase'], "%s.json"%pDictParams['useCase'])
    colPrint("Use case configuration load from '%s'"%ficJson)
    if os.path.exists(ficJson):
        with open(ficJson, 'r', encoding="utf-8") as fJson:
            dictData = json.load(fJson)
    else:
        colPrint("%s not found."%ficJson, "Error")
        sys.exit()

    # extend pDictParams with dictData[pDictParams['model']]
    if pDictParams['code'] not in dictData:
        colPrint("Unkwoned key '%s' in: "%pDictParams['code'], "Error")
        colPrint(str(dictData), "Debug")
    pDictParams.update(dictData[pDictParams['code']])
    pDictParams['dataExtension'] = dictData['dataExtension']

    # get pNbData data from the list in dataset and associated model dictionnary
    pDataset = [pDictParams['dataset']] if isinstance(pDictParams['dataset'], str) else pDictParams['dataset']
    pData = pDictParams['data']
    pDataExtension = pDictParams["dataExtension"]
    pDictParams['objets'] = {}

    dataPathList = []
    if pDataListMode == 0:
        colPrint("Get data from the list in file '%s'."%pDataListFichier)
        if not os.path.exists(pDataListFichier):
            colPrint("List of data file '%s' not found."%pDataListFichier, "Error")
            sys.exit()
        with open(pDataListFichier, 'r', encoding="utf-8") as f:
            contenu = f.read().split('\n')
        dataList = []
        for dataset in pDataset:
            for ligne, _ in enumerate(contenu):
                if contenu[ligne] != '':
                    data = contenu[ligne].split(',')
                    # marqueur @ pour un chemin d'accès relatif au chemin de la liste de données
                    if data[0][0] == '@':
                        data[0] = data[0][1:]
                        fichier = os.path.join(os.path.dirname(pDataListFichier), "%s.%s"%(data[0], pDataExtension))
                    else:
                        fichier = os.path.join(pDictParams['useCaseBase'], dataset, "%s.%s"%(data[0], pDataExtension))
                    if os.path.exists(fichier):
                        dataList.append(data[0])
                        dataPathList.append(fichier)
                        if pDictParams['modeltype'] == "detection":
                            if len(data[1:]) > 0:
                                pDictParams['objets'][data[0]] = [int(x) for x in data[1:]]
                            else:
                                pDictParams['objets'][data[0]] = []
                        elif pDictParams['modeltype'] == "segmentation":
                            if len(data[1:]) > 0:
                                pDictParams['objets'][data[0]] = [[x.split(': ')[0], [int(o) for o in x.split(': ')[1].split(';') if o.isdigit()]]
                                                                                if len(x.split(': ')) > 1 else [x.split(': ')[0]] for x in data[1:]]
                            else:
                                pDictParams['objets'][data[0]] = []
                    else:
                        colPrint("Fichier de donnée %s inexistant."%fichier, "Error")
        if pNbData == 0 or pNbData > len(dataList):
            pDictParams["nbData"] = len(dataList)  # Adjust pNbData
        else:
            dataList = dataList[: pNbData]

    elif pDataListMode == 1:
        colPrint("Get pNbData data from the list in dataset and associated model dictionnary (pNbData = %d)"%pNbData)
        dataList = []
        for dataset in pDataset:
            for data in pData:
                if pDictParams['modeltype'] == "detection" or pDictParams['modeltype'] == "segmentation":
                    fichier = os.path.join(pDictParams['useCaseBase'], dataset, "%s.%s"%(data[0], pDataExtension))
                    if os.path.exists(fichier):
                        dataList.append(data[0])
                        pDictParams['objets'][data[0]] = data[1]
                    else:
                        colPrint("Fichier de donnée %s inexistant."%fichier, "Error")
                else:
                    fichier = os.path.join(pDictParams['useCaseBase'], dataset, "%s.%s"%(data, pDataExtension))
                    if os.path.exists(fichier):
                        dataList.append(data)
                    else:
                        colPrint("Fichier de donnée %s inexistant."%fichier, "Error")
        if pNbData == 0 or pNbData > len(dataList):
            pDictParams["nbData"] = len(dataList)  # Adjust pNbData
        else:
            dataList = dataList[: pNbData]

    # get all data from the dataset directory
    elif pDataListMode == 2:
        colPrint("Get all data from the dataset directory")
        dataList = recupDataset(pDictParams['useCaseBase'], pDataset, pDataExtension)
        if pNbData != 0 and len(dataList) > pNbData:
            dataList = dataList[: pNbData]
        if pDictParams['modeltype'] == "detection" or pDictParams['modeltype'] == "segmentation":
            for data in dataList:
                pDictParams['objets'][data] = []
        pDictParams['nbData'] = len(dataList)

    # get pNbData data from dataset directory ; one each pStepData
    elif pDataListMode == 3:
        colPrint("Get pNbData data from dataset directory ; one each pStepData from pOffset ((pNbData, pStepData, pOffset) = (%d, %d, %d))"%(pNbData, pStepData, pOffset))
        dataList = []
        allDataList = recupDataset(pDictParams['useCaseBase'], pDataset, pDataExtension)
        increment = 0
        counter = 0
        for d in allDataList:
            if increment >= pOffset:
                if increment%pStepData == 0:
                    dataList.append(d)
                    pDictParams['objets'][d] = []
                    counter += 1
            increment += 1
            if counter == pNbData:
                break
        pDictParams['nbData'] = len(dataList)

    else:
        dataList = []
        colPrint("Unknown mode %s."%pDataListMode, "Warning")

    colPrint("Nb treated data: %s"%pDictParams['nbData'])

    # Set the list of full path of data to treat in case of mode 1 to 3
    if pDataListMode != 0:
        for dataset in pDataset:
            for data in dataList:
                fichier = os.path.join(pDictParams['useCaseBase'], dataset, "%s.%s"%(data, pDataExtension))
                # print(" > ", fichier, os.path.exists(fichier))
                if os.path.exists(fichier):
                    dataPathList.append(fichier)

    # construct subpath to store results
    if pMethodToUse not in pDictParams:
        colPrint("Element clef %s inconnu dans"%pMethodToUse, "Error")
        for e in pDictParams.keys():
            print("DEBUG> ", e, " :", pDictParams[e])

    repertProd = os.path.join(pDictParams['code'], "%s_%s"%(pMethodToUse, pDictParams[pMethodToUse]['suffixMethod']))

    # dictionnary upgrade
    pDictParams['dataList'] = dataList
    pDictParams['dataPathList'] = dataPathList
    pDictParams['dataBandPathList'] = None
    pDictParams['repertProd'] = repertProd
    pDictParams['aiModel'] = None
    pDictParams['xy'] = None

    # create output paths
    for repert in ["dataInference", "dataExplanations", "dataPlotExplanations"]:
        os.makedirs(os.path.join(pDictParams['dataProd'], repert, repertProd), exist_ok=True)
        os.chmod(os.path.join(pDictParams['dataProd'], repert, repertProd), 0o0777)

    for repert in ["dataMetrics", "dataPlotMetrics"]:
        os.makedirs(os.path.join(pDictParams['dataProd'], repert, pDictParams['code']), exist_ok=True)
        os.chmod(os.path.join(pDictParams['dataProd'], repert, pDictParams['code']), 0o0777)

    return pDictParams

# -------------------------------------------------------------------------------
## Launch the inference computation
# @param pDictParams : parameter dictionary
# @return parameter dictionary updated
def launchInference(pDictParams):
    colPrint("Action: Inference", "Action")

    # create the model instance
    aiModel = None
    if hasMethod(pDictParams['pluginUCXAI'], 'UCXAI_createModel'):
        aiModel = pDictParams['pluginUCXAI'].UCXAI_createModel(pDictParams)
        if aiModel is None:
            colPrint("Le modèle n'a pas pu être créé.", "Error")
            pDictParams['aiModel'] = pDictParams['useCase'], None
            return pDictParams
    else:
        colPrint("La méthode 'UCXAI_createModel' n'est pas implémentée pour le plugin '%s_%s'"%(pDictParams['pluginUCXAI'].nom, pDictParams['pluginUCXAI'].bibliotheque), "Warning")
    # inference
    xData, xDataSize, yPred = None, None, None
    if hasMethod(pDictParams['pluginUCXAI'], 'UCXAI_computeInference') and aiModel is not None:
        xData, xDataSize, yPred, _ = pDictParams['pluginUCXAI'].UCXAI_computeInference(aiModel, pDictParams)
    else:
        colPrint("La méthode 'UCXAI_computeInference' n'est pas implémentée pour le plugin '%s_%s'"%(pDictParams['pluginUCXAI'].nom, pDictParams['pluginUCXAI'].bibliotheque), "Warning")

    pDictParams['aiModel'] = pDictParams['useCase'], aiModel
    pDictParams['xy'] = xData, yPred
    pDictParams['datasize'] = xDataSize
    return pDictParams

# -------------------------------------------------------------------------------
## Launch the explanations computation
# @param pDictParams : parameter dictionary
# @return parameter dictionary updated
def launchExplanations(pDictParams):
    # create the prediction and the model instance if not exists
    if pDictParams['xy'] is None:
        pDictParams = launchInference(pDictParams)

    _, aiModel = pDictParams['aiModel']
    if aiModel is None:
        colPrint("Le modèle n'est pas créé.", "Error")
        return pDictParams

    colPrint("Action: Explain", "Action")

    # explanation
    if hasMethod(pDictParams['pluginUCXAI'], 'UCXAI_computeExplanations'):
        pDictParams['pluginUCXAI'].UCXAI_computeExplanations(pDictParams)
    else:
        colPrint("La méthode 'UCXAI_computeExplanations' n'est pas implémentée pour le plugin '%s_%s'"%(pDictParams['pluginUCXAI'].nom, pDictParams['pluginUCXAI'].bibliotheque), "Warning")

    return pDictParams

# -------------------------------------------------------------------------------
## Launch the explanations plotting
# @param pDictParams : parameter dictionary
def launchPlotExplanations(pDictParams):
    # create the prediction and the model instance if not exists
    if pDictParams['xy'] is None:
        pDictParams = launchInference(pDictParams)
    _, aiModel = pDictParams['aiModel']
    if aiModel is None:
        colPrint("Le modèle n'est pas créé.", "Error")
        return

    colPrint("Action: Plot explanations", "Action")

    # plot
    if hasMethod(pDictParams['pluginUCXAI'], 'UCXAI_plotExplanations'):
        pDictParams['pluginUCXAI'].UCXAI_plotExplanations(pDictParams)
    else:
        colPrint("La méthode 'UCXAI_plotExplanations' n'est pas implémentée pour le plugin '%s_%s'"%(pDictParams['pluginUCXAI'].nom, pDictParams['pluginUCXAI'].bibliotheque), "Warning")

# -------------------------------------------------------------------------------
## Launch the metrics computation
# @param pDictParams : parameter dictionary
# @return parameter dictionary updated
def launchMetrics(pDictParams):
    # create the prediction and the model instance if not exists
    if pDictParams['xy'] is None:
        pDictParams = launchInference(pDictParams)
    _, aiModel = pDictParams['aiModel']
    if aiModel is None:
        colPrint("Le modèle n'est pas créé.", "Error")
        return pDictParams

    colPrint("Action: Metric", "Action")

    # metric
    if hasMethod(pDictParams['pluginUCXAI'], 'UCXAI_computeMetrics'):
        pDictParams['pluginUCXAI'].UCXAI_computeMetrics(pDictParams)
    else:
        colPrint("La méthode 'UCXAI_computeMetrics' n'est pas implémentée pour le plugin '%s_%s'"%(pDictParams['pluginUCXAI'].nom, pDictParams['pluginUCXAI'].bibliotheque), "Warning")

    return pDictParams

# -------------------------------------------------------------------------------
## Lance l'action de calcul des métriques
# @param pDictParams : parameter dictionary
def launchPlotMetrics(pDictParams):
    # create the prediction and the model instance if not exists
    if pDictParams['xy'] is None:
        pDictParams = launchInference(pDictParams)
    _, aiModel = pDictParams['aiModel']
    if aiModel is None:
        colPrint("Le modèle n'est pas créé.", "Error")
        return

    colPrint("Action: Plot metric", "Action")

    # metric
    if hasMethod(pDictParams['pluginUCXAI'], 'UCXAI_plotMetrics'):
        pDictParams['pluginUCXAI'].UCXAI_plotMetrics(pDictParams)
    else:
        colPrint("La méthode 'UCXAI_plotMetrics' n'est pas implémentée pour le plugin '%s_%s'"%(pDictParams['pluginUCXAI'].nom, pDictParams['pluginUCXAI'].bibliotheque), "Warning")

# -------------------------------------------------------------------------------
## Launch the report writing as contact sheet
# @param pDictParams : parameter dictionary
def launchReport(pDictParams):
    colPrint("Action: Report", "Action")

    # report
    pDictParams['pluginUCXAI'].UCXAI_writeReport(pDictParams)

# -------------------------------------------------------------------------------
## Launch the full report
# @param pDictParams : parameter dictionary
def launchFullReport(pDictParams):
    colPrint("Action: Full Report", "Action")

    # report
    kaasrc.reports.writeFullReport(pDictParams)

# -------------------------------------------------------------------------------
## Main function to launch actions
# @param pAction : action in {Inference|Explain|Metric|Report}
# @param pDictParams : parameter dictionary
# @return parameter dictionary updated
def launchAction(pAction, pDictParams):
    if pAction == 'Inference':
        pDictParams = launchInference(pDictParams)
    elif pAction == 'Explain':
        pDictParams = launchExplanations(pDictParams)
    elif pAction == 'PlotE':
        launchPlotExplanations(pDictParams)
    elif pAction == 'Metric':
        pDictParams = launchMetrics(pDictParams)
    elif pAction == 'PlotM':
        launchPlotMetrics(pDictParams)
    elif pAction == 'Report':
        launchReport(pDictParams)
    elif pAction == 'FullReport':
        launchFullReport(pDictParams)
    else:
        colPrint("Unknown action %s: 'Inference', 'Explain', 'PlotE', 'Metric', 'PlotM' or 'Report'"%pAction, "Warning")
    return pDictParams

# ===============================================================================
# end of file
