# 2D_EM_EC_curvilinear_implicit_PiC
Test code for verifiying the fully electromagnetic energy conserving curvilinear implicit PiC method in 2D, as presented in Croonen et al. 2024. 


Manual:
1) Chose the physics case you want to run (two stream instability, weibel instability, thermal plasma, TE mode, or GEM) and set the corresponding flag to True and all others to False
2) Chose the geometry case you want to run and set the corresponding flag to True and all others to False and set the appropriate geometry parameters
   1) If 1D: Chose between Sinusoidal or Hyperbolic Tangent
   2) If 2D: Chose between CnC (based on Chacon and Chen 2016), skewed, squared, double hypertan, or double smooth heaviside
   3) If specifically doing the GEM case: Only double smooth heaviside is compatible
4) Chose the simulation paramters you want to use (size, number of cells, number of particles, etc.)
5) Run "Grid Builder.py" using the exact same settings as previously decided
   1) If you run the GEM case: Also run "GEM_setup.py" using the same settings
6) Set the frequency to: save field data, generate restart backups and plot runtime images to the desired values
7) Optional: If starting from a restart backup point, specify the restart file location, restart time and set the restart flag to True
8) Run "2D_EM_EC_curvilinear_implicit_PiC_test_implemntation.py"
9) Figures will be saved in the fig folder, and field data, logs, restart backups, and diagnostic data will be saved in their respective folders within save_data
