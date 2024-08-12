# Supplementary Code
Repository with the Thermo-hygro-chemical (THC) model of concrete at high temperatures based in FEniCS described in "Thermo-hygro-chemical Model of Concrete - From Curing to High Temperature Behavior" by G.Sciumé, M.Moreira and S.Dal Pont accepted at the Materials and Structures (2024).
Please, in case of use, cite the corresponding paper.

## Repository Structure
```
├── thermohygrochemo
│   ├── core
│   │   ├── CK_blocks.py: File with the coefficients for the mathematical model.
│   │   ├── constant_constitutive_laws.py: Constant constitutive laws used in the model.
│   │   ├── global_constants.py: Global constants used in multiple files.
│   │   └── thc_model.py: Core of the simulation.
│   │
│   ├── materials
│   │   └── material_young_age_kanema.py: Materials considered for the Kanema et al. benchmark.
│   │
│   ├── meshes
│   │   ├── 2D_Cylinder_MeshFunction.h5: Boundary markers for the 2D cylinder mesh.
│   │   ├── 2D_Cylinder_MeshFunction.xdmf: Boundary markers for the 2D cylinder mesh.
│   │   ├── 2D_Cylinder.h5: 2D cylinder mesh.
│   │   └── 2D_Cylinder.xdmf: 2D cylinder mesh.
│   │
│   └── simulations
│       ├── case_input_ya_kanema_2D_axi_manual_7_days.py: File with the case conditions.
│       └── run_ya_kanema_2D_axi_manual_7_days.py: File to run the simulation.
│
├── README.md (this file)
├── LICENSE.md
└── .gitignore
```

## How to run the simulation:
1. On the `./thermohygrochemo` folder of this repository run `python3 ./simulations/run_ya_kanema_2D_axi_manual_7_days.py`
2. A `./temp.py` will be created in the root folder of the repository, this is only a temporary copy of the case input and will be re-writen everytime a given `run_*.py` case is run
2. A `./results` directory will be created and the results will be saved there.
3. Once it finished the xdmf files can be loaded using Paraview, or Python scripts.
4. If you want to create a custom simulation, make a copy of the `run_ya_kanema_2D_axi_manual_7_days.py` and `case_input_ya_kanema_2D_axi_manual_7_days.py` in the `./simulations` directory. Update the boundary conditions and the name for the simulation. Don't forget to change the filename for the correct material and case_input in your new `run_ya_*.py` file. If you have any doubts, feel free to open an issue.

## Dependencies
- [Python 3.7+](https://www.python.org/)
- [NumPy](https://www.numpy.org)
- [FEniCS 2019.1.0](https://fenicsproject.org/download/archive/)


## References
The model is based on previous works, such as:
- [Meftah 2010] F. Meftah, F., and S. Dal Pont "Staggered finite volume modeling of transport phenomena in porous materials with convective boundary conditions", Transport in porous media 82:275-298, 2010.
- [Dal Pont 2011] S. Dal Pont, F. Meftah, B.A. Schrefler, "Modeling concrete under severe conditions as a multiphase material", Nuclear Engineering and Design 241(3):562-572, 2011
- [Sciumè 2013] G. Sciumè "Thermo-hygro-chemo-mechanical model of concrete at early ages and its extension to tumor growth numerical analysis", École normale supérieure de Cachan-ENS Cachan; Università degli studi di Padova, PhD Thesis, 2013.
- [Moreira 2021] M. H. Moreira, S. Dal Pont, R. F. Ausas, T. M. Cunha, A. P. Luz, V. C. Pandolfelli, "Direct comparison of multi and single-phase models depicting the drying process of refractory castables", Open Ceramics, 6:100111, 2021.

## License
[GPL](link_to_repo_license.MD)


Supplementary code to "Thermo-hygro-chemical Model of Concrete - From Curing to High Temperature Behavior". Copyright (C) 2024 Giuseppe Sciumé, Murilo Henrique Moreira and Stefano Dal Pont.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
