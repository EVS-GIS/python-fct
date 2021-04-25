# Fluvial Corridor Toolbox

## Overview

The scope of the Fluvial Corridor Toolbox (FCT) is fluvial geomorphology at the network scale, sometimes called upscaled hydromorphology.

The toolbox enables mapping fluvial corridors and measuring river features from very large datasets.
We first developed the toolbox for the case of the French Rhone basin, which is 90 000 km^2 wide.

![Corridor Width Profile Example (River le Var, France)](docs/img/corridor_maps.png)

The overall goal of the toolbox is to propose a quantitative application of the river style framework to support evidence based river management and river condition monitoring in the context of integrated river management.

This new version of the Fluvial Corridor Toolbox (FCT) started as an effort to implement port the ArcGIS code (Roux et al., 2015) to the QGis platform for promoting open science and sharing our tools with river practitioners.

The new version has been completely rewritten and incorporates ideas from Nardi et al. (2018) and Clubb et al. (2017) for improving the calculation of riverscape feature heights above the water level and delineate floodplain through the river network.
We also borrowed the concept of swath profiles from Hergarten et al. (2014) as the basis of a new raster-based approach to characterize floodplain features on cross-sections.

These new functionalities are based on high resolution DEM and landcover data to produce nested floodplain envelops. Finally, we implemented tiled processing of very large raster datasets after Barnes (2016, 2017).
This new version of the FCT also provides a lightweight framework for developing new processing toolchains/workflows.

## Basin-wide cartography

The cartography of the [French Rhone Basin](https://ebf.mapkiwiz.fr/qwc)
is visible online as an example of the output of the FCT.

## Principles

- The FCT provides a framework to characterize river corridors at the network scale
- Enables the processing of high resolution datasets and/or wide watersheds
- Robust, reproducible and automated raster-based metrology
- FAIR and open source platform

## Documentation

Documentation is available [here](https://tramebleue.github.io/fct-cli/)

## River Styles

Based on the FCT extracted metrics,
the river network can be segmented into functional units and classified into river styles.

A river style represents a type of river behavior.
River styles support better comparison between river reaches and are a useful tool to assess river condition.

![River Styles](docs/img/river_style.png)

## Workflows

The FCT implements a number of complex workflows :

- **Drainage network** derived from DEM's topography
- **Height maps** relative to the river or drainage network
- **Valley bottom delineation** from height and slope
- **Longitudinal reference system** for river characterization and monitoring
- **Lateral continuity**
- **Planform metrics**
- **Corridor metrics**

## Heights

![Heights](docs/img/height.png)

- A. flow height (Nardi et al., 2018)
- B. shortest height
- C. height above nearest drainage
- D. height above floodplain

## Example Metrics

In order to measure and calculate metrics, the valley bottom is divided into longitudinal units of constant length. A number of metrics can be calculated by the toolbox.

See the full [list of metrics](https://tramebleue.github.io/fct-cli/docs/metrics).

![Amplitude](docs/img/amplitude_drome.png)

![Sinuosity](docs/img/sinuosity_drome.png)

## Acknowledgements

The toolbox is developed within the programme
[Trame bleue, espaces et continuités](https://www.tramebleue.fr/)
in collaboration with :

* [GeoPeka](http://www.geopeka.com)
* [UMR 5600 Environnement, ville et société](http://umr5600.cnrs.fr/fr/accueil/).

Funding and support have been kindly provided by :

* [European Union](http://www.europe-en-france.gouv.fr/Centre-de-ressources/Actualites/Le-FEDER-qu-est-ce-que-c-est)
  and [Région Auvergne-Rhône-Alpes](https://www.auvergnerhonealpes.fr/)
* [Association nationale pour la recherche technologique](http://www.anrt.asso.fr/fr)
* [Agence de l'eau Rhône-Méditerranée-Corse](https://www.eaurmc.fr/)

![Supporting Partners](docs/img/partners.png)

## References

1. Alber, A. and Piégay, H. (2011). *Spatial disaggregation and aggregation procedures for characterizing fluvial features at the network-scale : Application to the Rhône basin (France)*. Geomorphology, 125(3), 343 360. [doi:10.1016/j.geomorph.2010.09.009](https://doi.org/10.1016/j.geomorph.2010.09.009)

2. Barnes, R. (2017). *Parallel non-divergent flow accumulation for trillion cell digital elevation models on desktops or clusters*. Environmental Modelling & Software, 92, 202 212. [doi:10.1016/j.envsoft.2017.02.022](https://doi.org/10.1016/j.envsoft.2017.02.022)

3. Clubb, F. J., Mudd, S. M. & al. (2017). *Geomorphometric delineation of floodplains and terraces from objectively defined topographic thresholds*. Earth Surface Dynamics, 5(3), 369 385. [doi:10.5194/esurf-5-369-2017](https://doi.org/10.5194/esurf-5-369-2017)

4. Nardi, F., Morrison, R. R. & al. (2018). *Hydrologic scaling for hydrogeomorphic floodplain mapping : Insights into human‐induced floodplain disconnectivity*. River Research and Applications, 34(7), 675 685. [doi:10.1002/rra.3296](https://doi.org/10.1002/rra.3296)

5. Roux, C., Alber, A. & al. (2015). *“FluvialCorridor” : A new ArcGIS toolbox package for multiscale riverscape exploration*. Geomorphology, 242, 29 37. [doi:10.1016/j.geomorph.2014.04.018](https://doi.org/10.1016/j.geomorph.2014.04.018)

## License

The Fluvial Corridor Toolbox is released under the [GNU Public License v3](./GPLv3.txt).