# Référentiel pour la France métropolitaine

Mise en application sur la BD TOPO 2021

## création d'un référentiel des exutoires
Le référentiel des exutoires vise à créer une ligne de référence permettant d'identifier et de sélectionner l'ensemble des exutoires des fleuves français. Ce référentiel doit être en mesure de pouvoir sélectionnner l'ensemble d'un réseau hydrographique bien orienté et connecté en remontant vers l'amont. Le référentiel prend en compte : 
- La couche limite_terre_mer de la BD TOPO pour les exutoires marins.
- Deux plans d'eau partuculiers, le lac du Bourget et le lac d'Annecy
- Des Lagunes méditerranéennes comme celle de Thau car le réseau hydrographique de la couche cours d'eau de l'IGN ne va pas toujours au dela de la lagune et la couche limite_terre_mer ne rentre pas tooujours dans les lagunes. Il y a donc une déconnexion entre les exutoires et la mer.


# Création du référentiel hydrographique de la France métropolitaine



Le référentiel hydrographique vise à être réseau des cours d'eau français, coulant et topologiquement juste. On doit pouvoir retrouver l'ensemble des affluents d'un fleuve en remontant le sens de l'écoulement vers l'amont. 

Couche troncons_hydrographiques:
- cpx_toponyme_de_cours_d_eau NOT NULL
- IdentifyNetworkNodes, sélection de l'exutoire puis de tous les troncons amonts
- MeasureNetworkFrom Outlet + Hack Order + Stralher
- LENGTH > 5000 OR STRALHER > 1