from one_dim_exact_diagonalization_class import DensityCalculator

if __name__ == '__main__':
    max_num_sites = 6 #maximum lattice size
    min_num_sites = 2 #minimum lattice size

    mu = 0.42 #chemical potential
    U = 1.0    # on-site interacteion strength

    Ji = 0.0 #initial hopping before quenching
    tc = 5.0 #initial delay time
    dt = 0.05 #time step increment
    tau_Q = 0.1
    num_steps = 800 #totol number of time steps

    num_cores = 8 # number of cores

    jf_initial = 0.001 #final hopping term after the quench - minimum value
    jf_final = 0.200 #final hopping term after the quench - maximum  value
    jf_steps = 0.001 #final hopping term after the quench - increment value

    # Create an instance of DensityCalculator
    density_calculator = DensityCalculator(U, mu, 
                                            num_cores,
                                            dt, num_steps, tc, tau_Q,
                                            Ji, jf_initial, jf_final, jf_steps,
                                            min_num_sites, max_num_sites)
    # Call the create_results method on the instance
    density_calculator.create_results()
