# (WIP) PNMPM

## Basic
This project contains code for Projected Newton Material Point Method.

## Examples
Run python3 examples/test2d.py. Create new test cases in the py file for new simulations.

## Parameters
The following parameters/methods can be set in examples:

- symplectic: True for explicit time integration, False for newton method.
- clf: CFL number.
- setDXandDT: set grid dx and stepsize dt.
- setGravity: set gravity.
- setLameParameter: set default lambda and mu.
- add_cube: sample MPM particle in a box.
- add_analytic_box: Box Boundary.
- add_surface_collider: Halfplane Boundary.

## Current Issues
The following are known issues at present:

- Float precision may cause non-convergency in Newton iterations. 
- Cuda + double precision is currently not supported for 3D.