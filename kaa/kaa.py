#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================================

import os, random, sys
from kaasrc.communs import colPrint
import kaasrc.plugin_collection
import kaasrc.kaaTUIapplication
import argparse
os.environ["TQDM_DISABLE"] = "1"

# -------------------------------------------------------------------------------
## Kaa's main function
def mainKaa(pDebug=False, pRepKaaPluginsXAI=None, pRepPluginsXAI=None, pRepPluginsUCXAI=None, pActivePluginsUCXAI="All", pDeactivePluginsUCXAI="", pFichierCommandes=None, pRepResultProd=None, pRepUseCaseBase=None):
    random.seed(0)

    # Récolte des plugins pluginsXAI du répertoire 'pluginsXAI'
    pluginsXAI = kaasrc.plugin_collection.PluginCollection()

    colPrint("Collection of pluginsXAI from the directory '%s'"%os.path.join(os.getcwd(), 'pluginsXAI'), "Config")
    pluginsXAI.loadPluginsXAI(pRepKaaPluginsXAI)

    if pRepPluginsXAI is not None:
        colPrint("Collection of pluginsXAI from the directory '%s'"%pRepPluginsXAI, "Config")
        pluginsXAI.loadPluginsXAI(pRepPluginsXAI)

    if len(pluginsXAI.plugins) == 0:
        colPrint("    No plugin for explainability library...", "Error")
        sys.exit()

    # Récolte des plugins pluginsUCXAI du répertoire pRepPluginsUCXAI
    pluginsUCXAI = kaasrc.plugin_collection.PluginCollection(pluginsLibrary=pluginsXAI.plugins, pluginsActive=pActivePluginsUCXAI, pluginsDeactive=pDeactivePluginsUCXAI)

    if pRepPluginsUCXAI is None:
        pRepPluginsUCXAI = os.path.join(os.path.dirname(sys.argv[0]), 'pluginsUCXAI')

    colPrint("Collection of pluginsUCXAI from the directory '%s'"%pRepPluginsUCXAI, "Config")
    pluginsUCXAI.loadPluginsUCXAI(pRepPluginsUCXAI)

    if len(pluginsUCXAI.plugins) == 0:
        colPrint("    No plugin of Use case vs Explainability ...", "Error")
        sys.exit()

    # Lancement du Text User Interface
    kaasrc.kaaTUIapplication.TUI(pluginsXAI, pluginsUCXAI, pRepResultProd, pRepUseCaseBase, pFichierCommandes, pDebug)

# -------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaa : platform for explanability works.")

    parser.add_argument('-D', '--Debug', dest='Debug', action='store_true', default=False, help="Debug mode")
    parser.add_argument('-c', '--command', dest='Command', type=str, default=None, help="Command file")
    parser.add_argument('-a', '--activate', dest='ActivatePlugin', type=str, default="All", help="Comma-separated list of use case plugins to activate (default 'All')")
    parser.add_argument('-d', '--deactivate', dest='DeactivatePlugin', type=str, default="", help="Comma-separated list of use case plugins to deactivate (default '')")
    parser.add_argument('-x', '--pluginXAI', dest='RepPluginsXAI', type=str, default=None, help="Path to add your own pluginsXAI (optional)")
    parser.add_argument('-u', '--pluginUCXAI', dest='RepPluginsUCXAI', type=str, default=None, help="Path to the pluginsUCXAI (default <Kaa path>/pluginsUCXAI)")
    parser.add_argument('-b', '--useCaseBase', dest='UseCaseBase', type=str, default=None, help="Path to the directory containing the use cases (default USECASE_BASE environment variable)")
    parser.add_argument('-r', '--resultProd', dest='ResultProd', type=str, default=None, help="Path to the directory containing the produced results (default RESULT_PROD environment variable)")
    args = parser.parse_args()

    # Traitement du mode debug
    debug = args.Debug

    # Traitement du fichier de commandes
    fichierCommandes = args.Command

    # Traitement de pluginsUCXAI à activer
    activePluginsUCXAI = args.ActivatePlugin
    if activePluginsUCXAI != "All":
        activePluginsUCXAI = activePluginsUCXAI.split(', ')

    # Traitement de pluginsUCXAI à desactiver
    deactivePluginsUCXAI = args.DeactivatePlugin
    deactivePluginsUCXAI = deactivePluginsUCXAI.split(', ')

    # Traitement du répertoire des pluginsXAI de Kaa
    repKaaPluginsXAI = os.path.join(os.path.dirname(sys.argv[0]), 'pluginsXAI')

    # Traitement du répertoire des pluginsXAI à ajouter
    repPluginsXAI = args.RepPluginsXAI

    # Traitement du répertoire des pluginsUCXAI à utiliser
    repPluginsUCXAI = args.RepPluginsUCXAI

    # Traitement du répertoire des cas d'usage
    repUseCaseBase = args.UseCaseBase
    if repUseCaseBase is None:
        repUseCaseBase = os.getenv("USECASE_BASE", './')

    # Traitement du répertoire des données produites
    repResultProd = args.ResultProd
    if repResultProd is None:
        repResultProd = os.getenv("RESULT_PROD", './')

    mainKaa(pDebug=debug,
            pRepKaaPluginsXAI=repKaaPluginsXAI,
            pRepPluginsXAI=repPluginsXAI,
            pRepPluginsUCXAI=repPluginsUCXAI,
            pActivePluginsUCXAI=activePluginsUCXAI,
            pDeactivePluginsUCXAI=deactivePluginsUCXAI,
            pFichierCommandes=fichierCommandes,
            pRepResultProd=repResultProd,
            pRepUseCaseBase=repUseCaseBase
            )

# ===============================================================================
# end of file
