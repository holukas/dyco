"""
_VENDOR: SELF-CONTAINED COPIES OF FORMERLY EXTERNAL HELPERS
===========================================================

Small leaf utilities that dyco used to import from the `diive` library. They are
copied here so dyco has no runtime dependency on diive.

Why they were copied rather than depended on: diive's internal restructuring
broke dyco four separate ways (two import paths moved, and the Python and pandas
floors diverged past the point where the two could be installed together). These
helpers are generic enough that any package would otherwise write them itself.

Provenance is recorded in each module's docstring. If a bug is found here, check
whether diive's copy has the same one.

Part of the dyco package: https://github.com/holukas/dyco
"""
