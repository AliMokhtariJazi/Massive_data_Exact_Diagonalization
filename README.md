# Massive_data_Exact_Diagonalization
This repository allows you to use the exact diagonalization method, i.e., Quspin library, and create Massive data, in this particular example, single-particle correlation in the Bose-Hubbard Model Hamiltonian, for specifically deep learning purposes. This can be easily extended to different Hamiltonians to study quantum many-body problems. It is designed so that you can generate Massive data for training the deep neural networks and extrapolate the many-body behaviors of the system in the part of the parameter space that is not accessible to other computational methods.
By choosing the proper Hamiltonian, you need to tailor the "one_dim_exact_diagonalization_class.py" for your specific case. 
In the "run.py," you can initiate the parameters of the Hamiltonian and the parameters to determine the number of times you want to run the code and, finally, the number of CPUs.
You can run this code on the cluster computers. The "once_dim_exact_diagonalization_class.py" is written in parallel; to run it on cluster computers, you can use the script file "python_run.sh". Remember the number of CPUs needs to be matched with the number of CPUs you assigned in the "run.py" file
