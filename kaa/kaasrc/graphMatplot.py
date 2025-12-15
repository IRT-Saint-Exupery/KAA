#!/usr/bin/python
# -*- coding: utf-8 -*-
# ===============================================================================
#  Chargement de l'environnement
# ===============================================================================

try:
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import kaasrc.communs
    import numpy as np
except Exception as err:
    print("Error: ", err)

# -----------------------------------------------------------------------------------
def matplotCreationFigure(pTitre, pParametres, ticsX=None, libelleX=None, ticsY=None, libelleY=None, ticsZ=None, libelleZ=None, polar=False):
    grille = pParametres['grille']
    largeur = pParametres['largeur']
    hauteur = pParametres['hauteur']
    margin = pParametres['margin']

    figure = plt.figure(figsize=(largeur / 100., hauteur / 100.))

    if ticsZ is not None:
        axe1 = figure.add_subplot(111, projection='3d')
    else:
        axe1 = figure.add_subplot(111, polar=polar)
    axe1.margins(margin)
    plt.title(pTitre, fontsize=10)

    # Gestion des tics des coordonnees
    if ticsX is not None:
        if ticsX != "off":
            plt.tick_params(axis='x', labelsize=8)
        else:
            plt.tick_params(axis='x', labelbottom='off', labeltop='off', labelsize=8)

    if ticsY is not None:
        if ticsY != "off":
            plt.tick_params(axis='y', labelsize=8)
        else:
            plt.tick_params(axis='y', labelleft='off', labelright='off', labelsize=8)

    if ticsZ is not None:
        if ticsZ != "off":
            plt.tick_params(axis='z', labelsize=8)
        else:
            plt.tick_params(axis='z', labelleft='off', labelright='off', labelsize=8)

    # Gestion des labels des coordonnees
    if libelleX is not None:
        plt.xlabel(libelleX, fontsize=8)
    if libelleY is not None:
        plt.ylabel(libelleY, fontsize=8)
    if libelleZ is not None:
        axe1.set_zlabel(libelleZ, fontsize=8)

    # Affichage de la grille
    if grille:
        plt.grid(which='both', axis=grille)

    return figure, axe1

# -----------------------------------------------------------------------------------
def localisation(loc):
    if loc == " * ":
        return 'best', None
    out = False
    if loc[0] == "O":
        out = True
        loc = loc[1:]
    if loc == "HD":
        return 'upper right', out
    if loc == "HG":
        return 'upper left', out
    if loc == "BG":
        return 'lower left', out
    if loc == "BD":
        return 'lower right', out
    if loc == "CG":
        return 'center left', out
    if loc == "CD":
        return 'center right', out
    if loc == "BC":
        return 'lower center', out
    if loc == "HC":
        return 'upper center', out
    return 'best', None

# -----------------------------------------------------------------------------------
def valeursTrace(valeurs, lim, flagAnnotations):
    #  Gestion des limites
    limmin = None
    limmax = None
    valideYlim = False
    if lim:
        (limmin, limmax) = lim
        for v in valeurs:
            if v > limmax:
                valideYlim = True
    if not valideYlim:
        lim = None
    if lim and flagAnnotations:
        (limmin, limmax) = lim
        listValeursTrace = []
        for v in valeurs:
            if v > limmax:
                listValeursTrace.append(limmax)
            else:
                listValeursTrace.append(v)
    else:
        listValeursTrace = valeurs

    return None, None, listValeursTrace, limmin, limmax, lim

# -----------------------------------------------------------------------------------
def GRAPH_parametresDefaut(pPalette="jet", nb_class=4):
    parametres = {
        'grille': None,
        'decalage': 0,
        'largbar': 0.8,
        'alpha': 0.5,
        'largeur': 800,
        'hauteur': 600,
        'legende': ' * ',
        'rotation': 45,
        'amplitude': True,
        'ylim': None,
        'lignes': '-',
        'points': '.',
        'couleurs': kaasrc.communs.getHexColors(pPalette, nb_class + 1),
        'epaisseur': 1.0,
        'lut': 16,
        'format': '%1.1f%%',
        'shadow': True,
        'explode': None,
        'ticsX': None,
        'offset': '0, 1',
        'doubleAxe': False,
        'dpi': 100,
        'margin': 0.05,
        'export': 'png',
        '8b': False,
        'logx': False,
        'logy': False,
        'remplissage': None}
    return parametres

# -----------------------------------------------------------------------------------
def GRAPH_matplotHisto(pDonneeGraphique, pTitre, pX, pY, pParametres):
    grille = pParametres['grille']
    legende = pParametres['legende']
    rotation = pParametres['rotation']
    amplitude = pParametres['amplitude']
    ylim = pParametres['ylim']
    couleurs = pParametres['couleurs']
    doubleAxe = pParametres['doubleAxe']
    largbar = pParametres['largbar']
    alpha = pParametres['alpha']
    dpi = pParametres['dpi']
    export = pParametres['export']
    margin = pParametres['margin']
    decalage = pParametres['decalage']

    # Récupération des abscisses des points à tracer
    libelleX = pX['libelle']
    ticsX = pX['tics']
    valeursX = pX['valeurs']

    # Création de la figure
    (figure, axe1) = matplotCreationFigure(pTitre, pParametres, 'on', libelleX)
  #  axe1 = figure.add_subplot(111)
    axe1.margins(margin)
    if doubleAxe:
        axe2 = axe1.twinx()
        axe2.margins(margin)

    # Donnees à tracer
    numCouleur = 0
    ylabels = {'y1': None, 'y2': None}
    for nb_courbe, courbe in enumerate(pY):
        labelY = ''
        if 'label' in courbe:
            labelY = courbe['label']
        valeursY = courbe['valeurs']
        flagAnnotations = 'annotations' in courbe
        if flagAnnotations:
            dAnnotationX, dAnnotationY, annRotation, annotations = courbe['annotations']
            if annotations is None:
                annotations = valeursY

        # Valeurs et limites
        ymin = None
        ymax = None
        (ymin, ymax, valeursY, ylimmin, ylimmax, ylim) = valeursTrace(valeursY, ylim, flagAnnotations)

        # Trace
        axe = 'y1'
        if 'axe' in courbe:
            axe = courbe['axe']
        if axe == 'y1':
            axe1.bar(np.array(valeursX) + (decalage * (nb_courbe - (len(pY) / 2) + 0.5)), valeursY, width=largbar, alpha=alpha, color=couleurs[numCouleur], label=labelY)
        if doubleAxe:
            if axe == 'y2':
                axe2.bar(np.array(valeursX) + (decalage * (nb_courbe - (len(pY) / 2) + 0.5)), valeursY, width=largbar, alpha=alpha, color=couleurs[numCouleur], label=labelY)
        if 'libelle' in courbe:
            ylabels[axe] = courbe['libelle']
        numCouleur += 1
        if numCouleur == len(couleurs):
            numCouleur = 0

        # Annotations des valeurs
        if flagAnnotations:
            positionsX = range(len(annotations))
            if axe == 'y1':
                for x, a, y in zip(positionsX, annotations, valeursY):
                    axe1.annotate(a, (x + (decalage * (nb_courbe - (len(pY) / 2) + 0.5)) + dAnnotationX, y + dAnnotationY), fontsize=6, rotation=annRotation, ha='left', va='bottom')
            if axe == 'y2':
                for x, a, y in zip(positionsX, annotations, valeursY):
                    axe2.annotate(a, (x + (decalage * (nb_courbe - (len(pY) / 2) + 0.5)) + dAnnotationX, y + dAnnotationY), fontsize=6, rotation=annRotation, ha='left', va='bottom')

    # Affichage des labels et ticks
    h1, l1 = axe1.get_legend_handles_labels()
    axe1.set_xticks(valeursX)
    axe1.set_xticklabels(ticsX)
    axe1.set_ylabel(ylabels['y1'], fontsize=8)
    axe1.tick_params(axis="both", labelsize=8)
    for tick in axe1.get_xticklabels():
        tick.set_rotation(rotation)
    h2 = []
    l2 = []
    if doubleAxe:
        h2, l2 = axe2.get_legend_handles_labels()
        axe2.set_ylabel(ylabels['y2'], fontsize=8)
        axe2.tick_params(axis="both", labelsize=8)
    if doubleAxe:
        for tick in axe2.get_xticklabels():
            tick.set_rotation(rotation)

    # Legende
    if legende is not None:
        loc, out = localisation(legende)
        if out is None:
            axe1.legend(h1 + h2, l1 + l2, loc=loc, fontsize=6, facecolor="white", framealpha=1.)
        else:
            axe1.legend(h1 + h2, l1 + l2, loc=loc, bbox_to_anchor=(1, 1), fontsize=6, facecolor="white", framealpha=1.)

    # Grille
    if grille:
        axe1.grid(axis=grille, linewidth=0.5)
        if doubleAxe:
            axe2.grid(axis=grille, linewidth=0.5)
    else:
        axe1.grid(linewidth=0)
        if doubleAxe:
            axe2.grid(linewidth=0)

    # Amplitude et limite du graphique
    if not amplitude:
        ymin = min(ymin)
        ymax = max(ymax)
        plt.ylim([(ymin - 0.5 * (ymax - ymin)), (ymax + 0.5 * (ymax - ymin))])
    elif ylim:
        (ylimmin, ylimmax) = ylim
        plt.ylim([ylimmin, ylimmax])

    # Sauvegarde
    fichierIMG = "%s.%s"%(pDonneeGraphique, export)
    try:
        plt.tight_layout()
        plt.savefig(fichierIMG, dpi=dpi, bbox_inches="tight")
    except Exception as err:
        print("Error: ", err)
        print("Erreur a la creation du fichier %s"%export)
    plt.cla()
    plt.close(figure)

# -----------------------------------------------------------------------------------
def GRAPH_matplot2D(pDonneeGraphique, pTitre, pX, pY, pXY, pParametres):
    grille = pParametres['grille']
    legende = pParametres['legende']
    rotation = pParametres['rotation']
    amplitude = pParametres['amplitude']
    ylim = pParametres['ylim']
    couleurs = pParametres['couleurs']
    doubleAxe = pParametres['doubleAxe']
    lignes = pParametres['lignes']
    points = pParametres['points']
    dpi = pParametres['dpi']
    export = pParametres['export']
    margin = pParametres['margin']
    margin = pParametres['margin']
    epaisseur = pParametres['epaisseur']
    remplissage = pParametres['remplissage']

    # Récupération des abscisses des points à tracer
    libelleX = pX['libelle']
    ticsX = None
    if 'tics' in pX:
        ticsX = pX['tics']
    valeursX = pX['valeurs']

    # Création de la figure
    (figure, axe1) = matplotCreationFigure(pTitre, pParametres, 'on', libelleX)
    # axe1 = figure.add_subplot(111)
    axe1.margins(margin)
    if doubleAxe:
        axe2 = axe1.twinx()
        axe2.margins(margin)

    # Lecture du fichier des donnees à tracer
    numCouleur = 0
    ylabels = {'y1': None, 'y2': None}
    for courbe in pY:
        labelY = ''
        if 'label' in courbe:
            labelY = courbe['label']
        valeursY = courbe['valeurs']
        flagAnnotations = 'annotations' in courbe
        if flagAnnotations:
            dAnnotationX, dAnnotationY, annRotation, annotations = courbe['annotations']
            if annotations is None:
                annotations = valeursY

        # Valeurs et limites
        ymin = None
        ymax = None
        (ymin, ymax, valeursY, ylimmin, ylimmax, ylim) = valeursTrace(valeursY, ylim, flagAnnotations)

        # Lignes et Points
        linewidth = 0
        if 'l' in courbe['trace']:
            linewidth = epaisseur
        marker = None
        if 'p' in courbe['trace']:
            marker = points

        # Couleur
        couleur = couleurs[numCouleur]
        if 'couleur' in courbe:
            couleur = courbe['couleur']

        # Trace
        axe = 'y1'
        if 'axe' in courbe:
            axe = courbe['axe']
        if axe == 'y1':
            axe1.plot(valeursX, valeursY, linewidth=linewidth, linestyle=lignes, marker=marker, color=couleur, label=labelY)
        # Fill under the curve
        if remplissage is not None:
            plt.fill_between(valeursX, valeursY, color=couleurs[remplissage], alpha=0.2)

        if axe == 'y2':
            axe2.plot(valeursX, valeursY, linewidth=linewidth, linestyle=lignes, marker=marker, color=couleur, label=labelY)
        if 'libelle' in courbe:
            ylabels[axe] = courbe['libelle']
        numCouleur += 1
        if numCouleur == len(couleurs):
            numCouleur = 0

        # Annotations des valeurs
        if flagAnnotations:
            positionsX = range(len(annotations))
            if axe == 'y1':
                for x, a, y in zip(positionsX, annotations, valeursY):
                    axe1.annotate(a, (x + dAnnotationX, y + dAnnotationY), fontsize=6, rotation=annRotation, ha='left', va='bottom')
            if axe == 'y2':
                for x, a, y in zip(positionsX, annotations, valeursY):
                    axe2.annotate(a, (x + dAnnotationX, y + dAnnotationY), fontsize=6, rotation=annRotation, ha='left', va='bottom')

    # XY
    if pXY is not None:
        couleur = couleurs[numCouleur]
        if 'couleur' in pXY:
            couleur = pXY['couleur']
        axe1.scatter(pXY['valeursX'], pXY['valeursY'], color=couleur, label=pXY['label'])

    # Affichage des labels et ticks
    if ticsX is not None:
        axe1.set_xticks(pX['valeurs'])
        axe1.set_xticklabels(ticsX)

        every_nth = len(axe1.xaxis.get_ticklabels()) // 10
        for n, label in enumerate(axe1.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)

    # Legende
    h1, l1 = axe1.get_legend_handles_labels()
    axe1.set_ylabel(ylabels['y1'], fontsize=8)
    axe1.tick_params(axis="both", labelsize=8)
    for tick in axe1.get_xticklabels():
        tick.set_rotation(rotation)
    h2 = []
    l2 = []
    if doubleAxe:
        h2, l2 = axe2.get_legend_handles_labels()
        axe2.set_ylabel(ylabels['y2'], fontsize=8)
        axe2.tick_params(axis="both", labelsize=8)
    if doubleAxe:
        for tick in axe2.get_xticklabels():
            tick.set_rotation(rotation)

    if legende is not None:
        loc, out = localisation(legende)
        if out is None:
            axe1.legend(h1 + h2, l1 + l2, loc=loc, fontsize=6, facecolor="white", framealpha=1.)
        else:
            axe1.legend(h1 + h2, l1 + l2, loc=loc, bbox_to_anchor=(1, 1), fontsize=6, facecolor="white", framealpha=1.)

    # Grille
    if grille:
        axe1.grid(axis=grille, linewidth=0.5)
        if doubleAxe:
            axe2.grid(axis=grille, linewidth=0.5)
    else:
        axe1.grid(linewidth=0)
        if doubleAxe:
            axe2.grid(linewidth=0)

    # Amplitude et limite du graphique
    if not amplitude:
        ymin = min(ymin)
        ymax = max(ymax)
        plt.ylim([(ymin - 0.5 * (ymax - ymin)), (ymax + 0.5 * (ymax - ymin))])
    elif ylim:
        (ylimmin, ylimmax) = ylim
        plt.ylim([ylimmin, ylimmax])

    # Sauvegarde
    fichierIMG = "%s.%s"%(pDonneeGraphique, export)
    plt.tight_layout()
    plt.savefig(fichierIMG, dpi=dpi, bbox_inches="tight")
    plt.cla()
    plt.close(figure)

# ===============================================================================
# end of file
