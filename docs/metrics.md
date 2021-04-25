Example river profile summary, as printed by `xarray`, showing the list of synthetic metrics calculated by the FCT :

```
<xarray.Dataset>
Dimensions:                   (side: 2, swath: 8070)
Coordinates:
  * side                      (side) object 'left' 'right'
  * swath                     (swath) MultiIndex
  - axis                      (swath) int64 15 15 15 15 15 ... 977 977 977 977
  - measure                   (swath) float64 100.0 300.0 ... 1.7e+03 1.9e+03
Data variables:
    drainage_area             (swath) float32 6.663e+04 6.574e+04 ... 5.682
    length_axis               (swath) float32 200.0 200.0 200.0 ... 200.0 200.0
    length_talweg             (swath) float32 201.7 222.5 181.7 ... 202.4 131.3
    distance_source_talweg    (swath) float32 1.124e+05 1.122e+05 ... 131.3
    distance_source_refaxis   (swath) float32 1.028e+05 1.026e+05 ... 200.0 0.0
    elevation_talweg          (swath) float32 90.69 90.7 90.77 ... 180.5 182.9
    elevation_talweg_med      (swath) float32 90.69 90.69 90.77 ... 180.5 182.6
    elevation_valley_bottom   (swath) float32 89.84 89.91 89.36 ... 180.1 183.3
    slope_talweg              (swath) float32 5.557e-05 0.0002063 ... 0.009102
    slope_valley_bottom       (swath) float32 5.509e-05 0.000214 ... 0.009033
    height_talweg             (swath) float32 0.9209 0.7964 ... 0.09631 11.23
    height_valley_bottom      (swath) float32 -0.1442 -1.106 ... -0.3784 -0.059
    gradient_height           (swath) float32 0.003197 0.0744 ... 2.074 0.0
    gradient_index            (swath) float32 1.643 38.17 35.02 ... 2.074 0.0
    area_valley_bottom        (swath) float32 2.043e+06 7.07e+05 ... 4.255e+04
    width_valley_bottom       (swath, side) float32 8.593e+03 ... 125.7
    width_valley_bottom_ma    (swath) float32 2.78e+03 2.78e+03 ... 191.3 154.6
    width_continuity          (swath) float32 1.022e+04 3.535e+03 ... 212.8
    width_water_channel       (swath) float32 5.028e+03 1.037e+03 ... 0.0 6.855
    width_active_channel      (swath) float32 5.268e+03 1.051e+03 ... 0.0 6.855
    width_natural_corridor    (swath, side) float32 1.14e+03 79.6 ... 6.491
    width_connected_corridor  (swath, side) float32 1.505e+03 79.6 ... 117.7
    amplitude                 (swath) float32 30.69 35.05 39.78 ... 13.89 11.68
    omega                     (swath) float32 0.1366 0.1715 ... 0.1125 0.0844
    wavelength                (swath) float32 883.1 ... 1.257e+03 1.257e+03
    sinuosity_length_ratio    (swath) float32 1.01 1.002 ... 0.9153 0.8896
    sinuosity_omega           (swath) float32 1.017 1.027 1.042 ... 1.012 1.007
    sinuosity_slope_ratio     (swath) float32 1.0 1.038 1.024 ... 1.034 1.0
    functional_unit           (swath) uint32 1 1 1 1 1 1 ... 563 563 563 563 563
    river_style               (swath) object 'S4' 'S4' 'S4' ... 'S4' 'S4' 'S4'
```