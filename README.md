# Variable Importance Analysis for Building Heat Loss Modelling

The presented repository contains Python scripts and data for the Bachelor's thesis of a Civil Engineering student of the University of Bath, Studnikova Vasilisa. The title of the research is "Variable Importance Analysis for Building Heat Loss Modelling". 

The first step of the analysis was to generate the datasets, which was done using the independent_sample_generation.py and realistic_sample_generation.py scripts. The resultant realistic and independent datasets are also included in the folder as  realistic_data1-3.csv and independent_data1-3.csv. 

These samples were analysed using cross_validation.py and Analysis_and_visualisation.py. As the variables were filtered to exclude the most significant features, Analysis_and_visualisation_Filtered.py script was used to rerun the analysis. The results are summarised in all_methods_score.csv and visualised using the results_summary.py script.
