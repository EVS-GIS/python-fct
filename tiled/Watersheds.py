# coding: utf-8

from collections import defaultdict, Counter
import click
import rasterio as rio
import speedup
from config import tileindex, filename

# Input:
# - RHT hiérarchisé (Hack)
# - Plan de drainage

# 1. Graph pixel A -> pixel B, rang, longueur
# 2. Éliminer les noeuds
#    qui sont à une distance < d0 du noeud aval de _même rang_
# 3. Numéroter les noeuds -> Numéro de BVU
#    Identifier la tuile d'appartenance
# 4. Watershed Analysis par tuile
#    Identifier BVU des inlets -> outlets
# 5. Refaire Watershed analysis par tuile
#    en ajoutant les outlets
# 6. Vectoriser chaque tuile
# 7. Aggréger les polygones jointifs qui appartiennent au même BVU
