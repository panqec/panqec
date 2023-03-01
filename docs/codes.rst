Codes
======

Quantum error correcting codes are represented as classes.
As stabilizer codes defined on lattices have much in common,
the specific classes for each code are inherited from the abstract class
:class:`panqec.codes.base.StabilizerCode`.
See
`Adding a new code to PanQEC <tutorials/Adding%20new%20code.html>`_
for example usage.
Note that most of the codes listed here can be visualized in the GUI,
for which you can see a demo at
`gui.quantumcodes.io <https://gui.quantumcodes.io/>`_.
Alternatively, you can also run it locally using `panqec start-gui` and opening
`localhost:5000` in your browser.

Abstract Classes
-----------------
.. automodule:: panqec.codes.base
    :members:
    :special-members: __init__

2D Surface Codes
-----------------
.. automodule:: panqec.codes.surface_2d
    :members:
    :special-members: __init__

3D Surface Codes
-----------------
.. automodule:: panqec.codes.surface_3d
    :members:
    :special-members: __init__

2D Color Codes
---------------
.. automodule:: panqec.codes.color_2d
    :members:
    :special-members: __init__

3D Color Codes
---------------
.. automodule:: panqec.codes.color_3d
    :members:
    :special-members: __init__

Fracton Codes
--------------
.. automodule:: panqec.codes.fractons
    :members:
    :special-members: __init__
