# Profiles and metrics

[ ] fct-network export-axis -j 6
[x] fct-swath profile elevation -j 6
[x] fct-swath export elevation
[>] fct-metrics talweg
[ ] fct-corridor refine-valley-mask -j 6 # delineate
[ ] fct-swath axes -j 6
[ ] fct-corridor valley-profile
[ ] fct-corridor height-above-valley-floor -j 6
[ ] fct-corridor talweg-profile
[x] fct-corridor landcover -j 6
[x] fct-corridor continuity -j 6
[x] fct-corridor continuity-weighted -j 6
[x] fct-corridor continuity-remap -j 6
[x] fct-swath profile valleybottom -j 6
[x] fct-swath export valleybottom
[x] fct-metrics valleybottom-width
[x] fct-swath profile landcover -j 6
[x] fct-swath export landcover
[x] fct-swath profile continuity
[>] fct-metrics landcover-width
[>] fct-metrics continuity-width
[x] fct-metrics planform
[ ] swath-intercepted talweg length
[ ] planform amplitude, wavelength, etc.
[ ] network/axis metrics workflow
