import os
import shutil
import pathlib
util_dir_ = pathlib.Path(__file__).parent.absolute()
util_dir = str(util_dir_)
thermo_dir_ = pathlib.Path(util_dir).parent.absolute()
thermo_dir = str(thermo_dir_)
TH_Model_dir = str(pathlib.Path(thermo_dir_).parent.absolute())

shutil.copy(thermo_dir + '/materials/material_young_age_kanema.py', thermo_dir + '/materials/materials_constitutive_laws.py')
shutil.copy(util_dir + '/case_input_ya_kanema_2D_axi_manual_7_days.py', TH_Model_dir + '/temp.py')
os.system('python3 ' + TH_Model_dir + '/temp.py ')
