#################################################################
#                                                               #
#   COUNTING NAE3SAT USING QAOA                                 #
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


numqubit = range(3,11)
vardeg = 3  
caudeg = 3    
seed = None

p = range(1,11)
A = 1
numtrials = 1
numgraph = 1

prob_sol = np.zeros((len(numqubit),len(p)))
err_num_sol = np.zeros((len(numqubit),len(p)))
tvi = np.zeros((len(numqubit),len(p)))
energy = np.zeros((len(numqubit),len(p)))
approx_ratio = np.zeros((len(numqubit),len(p)))

ex_num_sol = np.zeros((len(numqubit),len(p)))
ex_true_num_sol = np.zeros((len(numqubit),len(p)))
ex_prob_sol = np.zeros((len(numqubit),len(p)))

for i in numqubit:

    for j in p:

        err_sol_temp = np.zeros(numgraph)
        tvi_temp = np.zeros(numgraph)
        prob_sol_temp = np.zeros(numgraph)
        energy_temp = np.zeros(numgraph)
        true_num_sol = np.zeros(numgraph)
        num_sol = np.zeros(numgraph)
        approx_ratio_temp = np.zeros(numgraph)

        for n in range(numgraph):

            G = bicubic_graph(i, i, vardeg, caudeg, seed)
            sol, sol_int = solve_sharp_3sat(G.cf_nae, i)
            true_num_sol[n] = len(sol_int)

            if len(sol_int) == 0:
                print("There is no solutions to this NAE3SAT problem.")
                continue

            theta = np.zeros((numtrials,2*j))

            prob_sol_temp_temp = np.zeros(numtrials)
            tvi_temp_temp = np.zeros(numtrials)
            energy_temp_temp = np.zeros(numtrials)

            for k in range(numtrials):

                theta_ini = TQA_ini(j, G, A)
                prob, counts, theta[k,:] = regular_qaoa(G, theta_ini, A).run_qaoa_circ()

                energy_temp_temp[k] = regular_qaoa(G, theta_ini, A).compute_expectation(counts)
                tvi_temp_temp[k] = find_tvi(prob, sol)
                prob_sol_temp_temp[k] = find_prob_sol(G, counts)

            energy_temp[n] = np.min(energy_temp_temp)
            tvi_temp[n] = np.min(tvi_temp_temp)
            prob_sol_temp[n] = np.max(prob_sol_temp_temp)
            approx_ratio_temp[n] = -energy_temp[n]/i
            theta_opt = theta[np.argmax(prob_sol_temp_temp),:]

            counts = regular_qaoa(G, theta_opt, A).run_qaoa_circ()[1]
            num_sol[n] = counting(G, counts)
            err_sol_temp[n] = abs(num_sol[n]-true_num_sol[n])
        
        idx = np.argsort(prob_sol_temp)[len(prob_sol_temp)//2]
        ex_num_sol[i-numqubit[0],j-p[0]] = num_sol[idx]
        ex_true_num_sol[i-numqubit[0],j-p[0]] = true_num_sol[idx]
        ex_prob_sol[i-numqubit[0],j-p[0]] = prob_sol_temp[idx]

        energy[i-numqubit[0],j-p[0]] = np.mean(energy_temp)
        prob_sol[i-numqubit[0],j-p[0]] = np.mean(prob_sol_temp)
        tvi[i-numqubit[0],j-p[0]] = np.mean(tvi_temp)
        err_num_sol[i-numqubit[0],j-p[0]] = np.mean(err_sol_temp)
        approx_ratio[i-numqubit[0],j-p[0]] = np.mean(approx_ratio_temp)

for i in numqubit:
    print("Les probabilités maximales obtenues pour " + str(i) + " qubit sont :", list(prob_sol[i-numqubit[0],:]))

for i in numqubit:
    print("Les distances minimales pour "+ str(i) + " qubit sont :", list(tvi[i-numqubit[0],:]))

for i in numqubit:
    print("Les énergies minimales pour "+ str(i) + " qubit sont :", list(energy[i-numqubit[0],:]))

for i in numqubit:
    print("Les ratios d'approximation pour "+ str(i) + " qubit sont :", list(approx_ratio[i-numqubit[0],:]))

for i in numqubit:
    print("L'erreur sur le nombre de solutions pour "+ str(i) + " qubit est :", list(err_num_sol[i-numqubit[0],:]))

count_approx = []
count_true = []
for i in numqubit:
    count_approx.append(ex_num_sol[i-numqubit[0], np.argmax(ex_prob_sol, axis=1)[i-numqubit[0]]])
    count_true.append(ex_true_num_sol[i-numqubit[0], np.argmax(ex_prob_sol, axis=1)[i-numqubit[0]]])

print("For an example graph, the best approximative solution count is, for " + str(numqubit[0]) + " to " + str(numqubit[-1]) + " , ", count_approx)
print("For the same example graph, the true count is: ", count_true)
