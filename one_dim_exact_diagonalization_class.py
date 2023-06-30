 #!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from quspin.operators import hamiltonian,quantum_operator
from quspin.basis import boson_basis_1d
from joblib import Parallel, delayed


class DensityCalculator:

    def __init__(self, U, mu, 
                    num_cores,
                    dt, num_steps, tc, tau_Q,
                    Ji, jf_initial, jf_final, jf_steps,
                    min_num_sites, max_num_sites):
        self.U = U
        self.mu = mu
        self.dt = dt
        self.num_steps = num_steps
        self.Ji = Ji
        self.tc = tc
        self.tau_Q = tau_Q
        self.num_cores = num_cores
        self.jf_initial = jf_initial
        self.jf_final = jf_final
        self.jf_steps = jf_steps
        self.min_num_sites = min_num_sites
        self.max_num_sites = max_num_sites

    def _ramp(self, t, Ji, Jf, tc, tau_Q):
        return -0.5 * (Ji-Jf) * np.tanh((t-tc)/tau_Q) + 0.5 * (Ji+Jf)



    def Hamiltonian(self, num_sites, Jf):
        n_list = [[-0.5*self.U-self.mu, i] for i in range(num_sites)]
        nn_list = [[0.5*self.U, i, i] for i in range(num_sites)]
        ramp_args = [self.Ji, Jf, self.tc, self.tau_Q]
        pm_list = [[-1., i, (i+1)%num_sites] for i in range(num_sites)]
        mp_list = pm_list

        static_ops = [["n", n_list], ["nn", nn_list]]

        dynamic_ops = [["+-", pm_list, self._ramp, ramp_args],
                    ["-+", mp_list, self._ramp, ramp_args]]

        num_particles = num_sites
        basis = boson_basis_1d(L=num_sites,
                                Nb=num_particles,
                                kblock=0,
                                pblock=1)

        no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)

        H = hamiltonian(static_ops,
                        dynamic_ops,
                        static_fmt="csr",
                        dynamic_fmt="csr",
                        dtype=np.float64,
                        basis=basis,
                        **no_checks)
        return H


    def create_rho1_operator(self,num_sites, r):
        if (r==0):
            b=1 
            c=0
        else:
            b=0
            c=0.25

        n_list = [[b/num_sites, i] for i in range(num_sites)]
        static_ops_list_0 = [["n", n_list]]
        pm_list = ([[c/num_sites, i, (i+r)%num_sites] for i in range(num_sites)]
                +[[c/num_sites, i, (i-r)%num_sites] for i in range(num_sites)])
        mp_list = pm_list

        static_ops_list_1 = [["+-", pm_list], ["-+", mp_list]]

        num_particles = num_sites
        basis = boson_basis_1d(L=num_sites,
                                Nb=num_particles,
                                kblock=0,
                                pblock=1)
        no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
        operator_dict=dict(rho_diogonal=static_ops_list_0,rho_non_diagonal=static_ops_list_1)
        rho_1_op = quantum_operator(operator_dict,
                            dtype=np.float64,
                            basis=basis,
                            **no_checks)
        return rho_1_op


    def create_single_particle_density_clean_case(self, num_sites, r, Jf):
        H = self.Hamiltonian(num_sites, Jf)
        initial_H = H.aslinearoperator(time=0)
        eigenvalues, eigenvectors = eigsh(initial_H, k=1, which="SA")
        ground_state = eigenvectors[:, 0]

        ti = 0.
        tf = ti + self.num_steps * self.dt
        times = np.linspace(ti, tf, num=self.num_steps+1)
        states = H.evolve(ground_state, ti, times, iterate=True)

        rho_1_op = self.create_rho1_operator(num_sites, r)
        rho_1 = np.empty([self.num_steps+1])
        rho_1_array = []
        time_array = []
        t_idx = 0
        for psi_t in states:
            rho_1[t_idx] = rho_1_op.expt_value(psi_t).real
            rho_1_array.append(rho_1[t_idx])
            time_array.append(t_idx) 
            t_idx += 1
        rho_1_array = np.array(rho_1_array)
        df = pd.DataFrame(rho_1_array)

        cwd = os.getcwd()
        data_path = os.path.join(cwd, 'exact_diagonalization_data')
        os.makedirs(data_path, exist_ok=True)
        
        print("L is: ", num_sites, "            r is: ", r, "               jf:  ", Jf)
        df.to_csv(os.path.join(data_path, f'num_steps_{self.num_steps} Jf_{Jf:.6e} num_site_{num_sites} r_{r}.csv'), index=False)
        return 0  


    def create_results(self):
        for L in range(self.min_num_sites, self.max_num_sites+1):
            for r in range(1, np.floor(L/2).astype(int)+1):
                jf_values = np.arange(self.jf_initial, self.jf_final + self.jf_steps, self.jf_steps)
                Parallel(n_jobs=self.num_cores)(delayed(self.create_single_particle_density_clean_case)(L, r, jf) for jf in jf_values)


