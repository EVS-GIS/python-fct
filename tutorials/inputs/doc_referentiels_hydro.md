# Référentiel pour la France métropolitaine

Mise en application sur la BD TOPO 2021

## création d'un référentiel des exutoires
Le référentiel des exutoires vise à créer une ligne de référence permettant d'identifier et de sélectionner l'ensemble des exutoires des fleuves français. Ce référentiel doit être en mesure de pouvoir sélectionnner l'ensemble d'un réseau hydrographique bien orienté et connecté en remontant vers l'amont. Le référentiel prend en compte : 
- La couche limite_terre_mer de la BD TOPO pour les exutoires marins.
- Deux plans d'eau particuliers, le lac du Bourget, le lac d'Annecy
- Des Lagunes méditerranéennes comme celle de Thau car le réseau hydrographique de la couche cours d'eau de l'IGN ne va pas toujours au dela de la lagune et la couche limite_terre_mer ne rentre pas tooujours dans les lagunes. Il y a donc une déconnexion entre les exutoires et la mer.
- Les frontières pour les réseaux des bassins qui s'écoulent en dehors de la France

Traitements de création du référentiel des exutoires QGIS : 
- Couche limite_terre_mer
  - Extraction de la couche
- Couche plan_d_eau
  - Sélection manuelle et extraction de la couche : le lac du Bourget, le lac d'Annecy et les lagunes méditerranéenne
  - Polygones vers polylignes
- Couche bassin_versant_topographique
  - Fusion des polygones de la couche
  - réparer les géométries
  - Polygones vers polylignes
- Les trois couches vecteur
  - Fusionner des couches vecteur
  - Edition, calculatrice de champ, mise à jour du champ "fid", @row_number
  - buffer 50m



- Sélection manuelle et extraction des bassins versants dont l'exutoire est la frontière de la couche bassin_versant_topographique avec l'aide de la couche troncon_hydrographique

- Edition, séparation des entités pour avoir les frontières extérieures, suppression des limites maritimes.
- Modifier les noeuds des limites de frontières avec la limite_terre_mer pour connecter les deux limites
- Sélection des 


# Création du référentiel hydrographique de la France métropolitaine



Le référentiel hydrographique vise à être réseau des cours d'eau français, coulant et topologiquement juste. On doit pouvoir retrouver l'ensemble des affluents d'un fleuve en remontant le sens de l'écoulement vers l'amont. 

Depuis la couche troncon_hydrographique de la BD TOPO:
- prendre les tronçons avec liens_vers_cours_d_eau IS NOT NULL or EMPTY
  - SELECT * FROM troncon_hydrographique WHERE liens_vers_cours_d_eau IS NOT NULL AND liens_vers_cours_d_eau != ''; Le nom de la couche est troncon_hydrographique_cours_d_eau.
- Trois types d'erreur à corriger mais les corrections ne se font pas directement dans la couche troncon_hydrographique_cours_d_eau pour mieux assurer la tracabilité et la reproductibilité à une autre version de la BD TOPO : 
  - Des confluences non connectées ou des cours d'eau trop loin des exutoire. Il s'agit alors de regarder sur la couche troncon_hydrographique complète les tronçons, sélectionner les tronçons manquant, les enregistrer dans une nouvelle couche geopackage et leur mettre l'identifiant de cours d'eau auquels ils sont rattachés dans le champ liens_vers_cours_d_eau. Le nom de cette couche est troncon_hydrographique_cours_d_eau_conn.
  - Des confluences non connectées ou des cours d'eau trop loin des exutoire mais sans troncons existant pour compléter les connexions. Il faut alors modifier la géométrie d'un tronçon pour permettre la liaison ou l'extension du cours d'eau. Il s'agit alors de sélectionner le tronçon à modifier, l'enregister dans une couche de modification puis modifier cette nouvelle entité dans cette même couche. La couche de modification est troncon_hydrographique_cours_d_eau_modif_geom
  - Des sens d'écoulement sont erronnés et doivent être inversé. Les tronçons concernés sont sélectionnés puis enregistrés dans une nouvelle couche. Le nom de cette couche est troncon_hydrographique_cours_d_eau_corr_dir_ecoulement.
- IdentifyNetworkNodes
- sélection depuis le buffer de la couche exutoire
- SelectConnectedReaches avec option upstream pour sélectionner tout les tronçons depuis l'aval

- ajustement du réseau hydrographique : troncon_hydrographique_connexion
  - suppression des doublons
  - ajout au troncon_hydrographique sans NULL et Vide

Couche troncons_hydrographiques:
- cpx_toponyme_de_cours_d_eau NOT NULL
- IdentifyNetworkNodes, sélection de l'exutoire puis de tous les troncons amonts
- MeasureNetworkFrom Outlet + Hack Order + Stralher
- LENGTH > 5000 OR STRALHER > 1