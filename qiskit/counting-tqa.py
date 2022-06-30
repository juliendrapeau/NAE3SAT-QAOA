#################################################################
#                                                               #
#   COUNTING NAE3SAT FROM SAMPLING USING QAOA                   #
#   ==========================================                  #
#   ( counting-from-sampling.py )                               #
#   First instance: 20220517                                    #
#   Written by Julien Drapeau                                   #
#                                                               #
#   This script samples solutions of the NAE3SAT problem        #
#   using QAOA. It computes many properties of the probability  #
#   distribution obtained. Then, an approximative count of the  #
#   number of solutions to the problem is estimated from the    #
#   distribution.                                               #
#                                                               #
#   DEPENDENCIES: nae3sat_qaoa.py                               #
#                                                               #
#################################################################


from nae3sat_qaoa import *


#parameters of the random bicubic graphs
numqubit = range(3,7)
vardeg = 3  
caudeg = 3    
seed = None

#parameters of the qaoa
p = range(1,5)
A = 1
numtrial = 10
numgraph = 5
numshot = 512

#instantiating the properties of probability distribution
prob_sol = np.zeros((len(numqubit), numgraph, len(p), numtrial))
tvd = np.zeros((len(numqubit), numgraph, len(p), numtrial))
energy = np.zeros((len(numqubit), numgraph, len(p), numtrial))
approx_ratio = np.zeros((len(numqubit), numgraph, len(p), numtrial))
err_num_sol = np.zeros((len(numqubit), numgraph, len(p), numtrial))

#instantiating the optimal initial parameters
theta_opt = np.zeros((len(numqubit), numgraph, len(p), numtrial, 2*np.max(p)))

#instantiating the number of solutions
exact_num_sol = np.zeros((len(numqubit), numgraph))
approx_num_sol = np.zeros((len(numqubit), numgraph, len(p), numtrial))

for iterqubit, qubit in enumerate(numqubit):

    for graph in range(numgraph):

        #generating a random bicubic graph and finding the exact solutions
        G = bicubic_graph(qubit, qubit, vardeg, caudeg, seed)
        exact_sol_pdf, sol_int = solve_sharp_3sat(G.cf_nae, qubit)
        exact_num_sol[iterqubit, graph] = len(sol_int)

        if len(sol_int) == 0:
            print("There is no solutions to this NAE3SAT problem.")
            continue

        for iterdepth, depth in enumerate(p):

            for trial in range(numtrial):

                #initializing the parameters with TQA
                theta_ini = TQA_ini(depth, G, A)

                #running qaoa
                approx_prob_pdf, counts, theta_opt[iterqubit, graph, iterdepth, trial, :2*depth] = regular_qaoa(G, theta_ini, A).run_qaoa_circ()
                theta_opt[iterqubit, graph, iterdepth, trial, iterdepth:] = None

                #calculating the properties
                prob_sol[iterqubit, graph, iterdepth, trial] = find_prob_sol(G, counts)
                tvd[iterqubit, graph, iterdepth, trial] = find_tvd(approx_prob_pdf, exact_sol_pdf)
                energy[iterqubit, graph, iterdepth, trial] = regular_qaoa(G, theta_ini, A).compute_expectation(counts)
                approx_ratio[iterqubit, graph, iterdepth, trial] = -energy[iterqubit, graph, iterdepth, trial]/qubit

                #calculating the number of solutions
                approx_num_sol[iterqubit, graph, iterdepth, trial] = counting(G, counts)
                err_num_sol[iterqubit, graph, iterdepth, trial] = abs(approx_num_sol[iterqubit, graph, iterdepth, trial] - exact_num_sol[iterqubit, graph])

#saving the arrays to npz
if len(numqubit) == 1:
    np.savez('results/qubit-'+str(numqubit[0]), 
             prob_sol=prob_sol,
             tvd=tvd,
             energy=energy,
             approx_ratio=approx_ratio,
             err_num_sol=err_num_sol,
             theta_opt=theta_opt,
             exact_num_sol=exact_num_sol,
             approx_num_sol=approx_num_sol)
                    
else:
    np.savez('results/qubit-'+str(numqubit[0])+'-to-'+str(numqubit[-1]),
             prob_sol=prob_sol,
             tvd=tvd,
             energy=energy,
             approx_ratio=approx_ratio,
             err_num_sol=err_num_sol,
             theta_opt=theta_opt,
             exact_num_sol=exact_num_sol,
             approx_num_sol=approx_num_sol)
