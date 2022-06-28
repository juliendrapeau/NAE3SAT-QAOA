#################################################################
#                                                               #
#   FUNCTIONS FOR SOLVING #NAE3SAT USING QAOA ALGORITHM         #
#   ==========================================                  #
#   ( nae3sat_qaoa.py )                                         #
#   First instance: 20220517                                    #
#   Written by Julien Drapeau (julien.drapeau@usherbrooke.ca)   #
#                                                               #
#   Routines for solving the #NAE3SAT problem using the QAOA    #
#   algorithm. This module provides functions for the sampling  #
#   of random instances of bicubic graphs representing NAE3SAT  #
#   problems using qecstruct, finding an approximate solution   #
#   of the number of solutions using the QAOA algorithm, and    #
#   evaluating many performance statistics like the total       #
#   variation distance, the approximation ratio and others.     #
#   The solutions are benchmarked with the SAT solver Glucose4  #
#   included with pysat.                                        #
#                                                               #
#   DEPENDENCIES: numpy, scipy, sklearn, qesctruct, pysat,      #
#                 qiskit, igraph                                #
#                                                               #
#################################################################


import numpy as np
from scipy.optimize import minimize
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity
import qecstruct as qs
from pysat.solvers import Glucose4
from qiskit import QuantumCircuit
from qiskit import Aer
from qiskit.quantum_info import Statevector
import igraph as ig


class bicubic_graph:

    """
    This class instantiates a random bicubic graph representating a NAE3SAT problem using qecstruct. It then maps the bicubic graph to an Ising graph using the Ising formulation of the NA3SAT problem.
    """

    def __init__(self, numvar, numcau, vardeg, caudeg, seed):

        """
        Args:
        numvar: number of variables
        numcau: number of causes
        vardeg: variables degree
        caudeg: causes degree
        """

        #samples a random bicubic graph
        code = qs.random_regular_code(numvar, numcau, vardeg, caudeg, qs.Rng(seed))

        #write the 3SAT formula and find the edges of the ising graph
        cf = []
        edges = []
        for row in code.par_mat().rows():
            temp_cf = []
            for value in row:
                temp_cf.append(value)
            cf.append(temp_cf)
            edges.append([temp_cf[0],temp_cf[1]])
            edges.append([temp_cf[1],temp_cf[2]])
            edges.append([temp_cf[2],temp_cf[0]])

        #3SAT formula
        self.cf = np.array(cf)+1
        #NA3SAT formula
        self.cf_nae = np.vstack((self.cf, np.invert(self.cf)+1))
        #edges of the ising graph
        self.edges = np.array(edges)
        #number of variables
        self.numnodes = numvar

    def ising_view(self):
        """
        Ising formulation graph
        """
        return ig.Graph(n=self.numnodes, edges=self.edges)

    #Optional dependency needed: tensorcsp.py
    #def cf_view(self):
        """
        CNF formulation graph
        """
        return cnf_graph(self.cf)

class graph_from_cf:

    """
    This class instantiates a graph from a CNF formula representating a NAE3SAT problem. It then maps the graph to an Ising graph using the Ising formulation of the NA3SAT problem.
    """

    def __init__(self, cf, numvar):

        """
        Args:
        cf: CNF formula
        numqubit: number of variables
        """
        
        #find the edges of the Ising graph
        edges = []
        for i in range(np.shape(cf)[0]):
            edges.append([cf[i,0],cf[i,1]])
            edges.append([cf[i,1],cf[i,2]])
            edges.append([cf[i,2],cf[i,0]])   

        #3SAT formula
        self.cf = cf
        #NAE3SAT formula
        self.cf_nae = np.vstack((self.cf, np.invert(self.cf)+1))
        #edges of the Ising graph
        self.edges = np.array(edges)-1
        #number of variables
        self.numnodes = numvar

class regular_qaoa:

    """
    This class regroups the main methods of the regular QAOA algorithm applied to a NAE3SAT problem. It instantiates a QAOA object for a specific graph representing a NAE3SAT problem.
    """

    def __init__(self, G, theta_ini, A, shots=512, optimizer='COBYLA', simulator='aer_simulator'):

        """
        Args:
        G: graph object
        theta_ini: initial list of unitary parameters
        A: parameter of the ising model
        shots: number of circuit samples
        optimizer: scipy optimizer
        simulator: qiskit optimizer
        """

        self.G = G
        self.theta_ini = theta_ini
        self.A = A
        self.shots = shots
        self.optimizer = optimizer
        self.simulator = simulator

    def compute_loss(self, x):
    
        """
        Given a bitstring as a solution, this function return 
        the loss of the solution.
        
        Args:
            x: str
               solution bitstring
            
        Returns:
            loss: float
                 loss
        """
        
        loss = 0
        for i, j in self.G.edges:
            loss += self.A*(2*int(x[i])-1)*(2*int(x[j])-1)
        
        return loss

    def compute_expectation(self, counts):
        
        """
        Computes expectation value based on measurement results.
        
        Args:
            counts: dict
                    key as bitstring, val as count
            
        Returns:
            expectation: float
                         expectation value
        """
        
        avg = 0
        sum_count = 0
        for bitstring, count in counts.items():
            obj = self.compute_loss(bitstring)
            avg += obj*count
            
            sum_count += count
        
        return avg/sum_count
    
    def create_qaoa_circ(self, theta, measure=True):
        
        """
        Creates a parametrized qaoa circuit.
        
        Args:
            G: graph object
            theta: list of unitary parameters
            measure: if True, mesure all qubits at the end of the circuit
        
        Returns:
            qc: qiskit circuit
        """
        
        nqubits = self.G.numnodes
        p = len(theta)//2 #number of alternating unitaries
        qc = QuantumCircuit(nqubits)
        
        beta = theta[:p]
        gamma = theta[p:]
        
        #Initial state
        for i in range(0, nqubits):
            qc.h(i)
        
        qc.barrier()
        
        for irep in range(0,p):
            
            #problem unitary
            for pair in self.G.edges:
                qc.rzz(2*self.A*gamma[irep], pair[0], pair[1])
                qc.barrier()
            
            qc.barrier()
            
            #driver unitary
            for i in range(0, nqubits):
                qc.rx(2*beta[irep], i)
        
        if measure == True:
            qc.measure_all()
            
        return qc
    
    def get_expectation(self):
        
        """
        Runs parametrized circuit.
        """
        
        backend = Aer.get_backend('qasm_simulator')
        backend.shots = self.shots
        
        def execute_circ(theta):
            qc = self.create_qaoa_circ(theta)
            counts = backend.run(qc, nshots=512).result().get_counts()
            
            return self.compute_expectation(counts)
        
        return execute_circ

    def run_qaoa_circ(self):

        """
        Minimize the expectation value by finding the best parameters.
        Samples the circuit multiples time and get the probability distribution of the final state.

        Returns:
            prob: list
                  probability distribution of the final state
            counts: dict
                    key as bitstring, val as count
            res.x: list
                   list of final unitary parameters

        """

        expectation = self.get_expectation()
        res = minimize(expectation, self.theta_ini, method=self.optimizer)

        backend = Aer.get_backend(self.simulator)
        backend.shots = self.shots

        #reverse the bits for convention
        qc_res = self.create_qaoa_circ(res.x, measure=False).reverse_bits()
        prob = Statevector(qc_res).probabilities()

        #reverse the bits for convention
        qc_res = self.create_qaoa_circ(res.x).reverse_bits()
        counts = backend.run(qc_res, nshots=512).result().get_counts()

        return prob, counts, res.x

class multi_angle_qaoa:

    """
    This class regroups the main methods of the mutli-angle QAOA algorithm applied to a NAE3SAT problem. It instantiates a QAOA object for a specific graph representing a NAE3SAT problem.
    """

    def __init__(self, G, theta_ini, A, shots=512, optimizer='COBYLA', simulator='aer_simulator'):

        """
        Args:
        G: graph object
        theta_ini: initial list of unitary parameters
        A: parameter of the ising model
        shots: number of circuit samples
        optimizer: scipy optimizer
        simulator: qiskit optimizer
        """

        self.G = G
        self.theta_ini = theta_ini
        self.A = A
        self.shots = shots
        self.optimizer = optimizer
        self.simulator = simulator        

    def compute_loss(self, x):
    
        """
        Given a bitstring as a solution, this function return 
        the loss of the solution.
        
        Args:
            x: str
               solution bitstring
            
        Returns:
            loss: float
                 loss
        """
        
        loss = 0
        for i, j in self.G.edges:
            loss += self.A*(2*int(x[i])-1)*(2*int(x[j])-1)
        
        return loss

    def compute_expectation(self, counts):
        
        """
        Computes expectation value based on measurement results.
        
        Args:
            counts: dict
                    key as bitstring, val as count
            
        Returns:
            expectation: float
                         expectation value
        """
        
        
        avg = 0
        sum_count = 0
        for bitstring, count in counts.items():
            obj = self.compute_loss(bitstring)
            avg += obj*count
            sum_count += count
        
        return avg/sum_count
    
    def create_qaoa_circ(self, theta, measure=True):
        
        """
        Creates a parametrized qaoa circuit.
        
        Args:
            G: graph object
            theta: list of unitary parameters
            measure: if True, mesure all qubits at the end of the circuit
        
        Returns:
            qc: qiskit circuit
        """
    
        nqubits = self.G.numnodes
        numedge = len(self.G.edges)
        p = len(theta)//(2*numedge) #number of alternating unitaries
        
        qc = QuantumCircuit(nqubits)
        
        beta = np.array(theta[:p*numedge]).reshape((numedge,p))
        gamma = np.array(theta[p*numedge:]).reshape((numedge,p))
  
        #Initial state
        for i in range(0, nqubits):
            qc.h(i)
        
        qc.barrier()
        
        for irep in range(0,p):
            
            #problem unitary
            for iter,pair in enumerate(self.G.edges):
                qc.rzz(2*self.A*gamma[iter,irep], pair[0], pair[1])
                qc.barrier()
            
            qc.barrier()
            
            #driver unitary
            for i in range(0, nqubits):
                qc.rx(2*beta[i, irep], i)
        
        if measure == True:
            qc.measure_all()
            
        return qc
    
    def get_expectation(self):
        
        """
        Runs parametrized circuit.
        """
        
        backend = Aer.get_backend('qasm_simulator')
        backend.shots = self.shots
        
        def execute_circ(theta):
            qc = self.create_qaoa_circ(theta)
            counts = backend.run(qc, nshots=512).result().get_counts()
            
            return self.compute_expectation(counts)
        
        return execute_circ

    def run_qaoa_circ(self):

        """
        Minimize the expectation value by finding the best parameters.
        Samples the circuit multiples time and get the probability distribution of the final state.

        Returns:
            prob: list
                  probability distribution of the final state
            counts: dict
                    key as bitstring, val as count
            res.x: list
                   list of final unitary parameters

        """

        expectation = self.get_expectation()
        res = minimize(expectation, self.theta_ini, method=self.optimizer)

        backend = Aer.get_backend(self.simulator)
        backend.shots = self.shots

        #reverse the bits for convention
        qc_res = self.create_qaoa_circ(res.x, measure=False).reverse_bits()
        prob = Statevector(qc_res).probabilities()

        #reverse the bits for convention
        qc_res = self.create_qaoa_circ(res.x).reverse_bits()
        counts = backend.run(qc_res, nshots=512).result().get_counts()

        return prob, counts, res.x


def rand_theta_ini(p):

    """
    Creates a list of random initial unitary parameters for the QAOA algorithm.

    Args:
        p: depth of the QAOA circuit

    Returns:
        theta_ini: list of random unitary parameters
    """

    theta_ini = np.hstack((np.random.rand(p)*np.pi,np.random.rand(p)*np.pi*2))

    return theta_ini

def TQA_ini(p, G, A, shots=512, simulator='aer_simulator'):

    """
    Creates a list of initial unitary parameters for the QAOA algorithm. The parameters are initialized based on the Trotterized Quantum Annealing (TQA) strategy for initialization. See "Quantum Annealing Initialization of the Quantum Approximate Optimization Algorithm".

    Args:
        p: depth of the QAOA circuit
        G: graph object
        A: parameter of the Ising model
        shots: number of circuit samples
        simulator: qiskit simulator

    Returns:
        theta_ini: list of random unitary parameters
    """

    time = np.linspace(0.1, 4, 20)
    backend = Aer.get_backend(simulator)
    backend.shots = shots

    energy = []
    for t_max in time:  
        dt = t_max/p
        t = dt*(np.arange(1,p+1)-0.5)
        gamma = (t/t_max)*dt
        beta = (1-t/t_max)*dt
        theta = np.concatenate((beta,gamma))
        qc = regular_qaoa(G,theta,A).create_qaoa_circ(theta).reverse_bits()
        counts = backend.run(qc, nshots=512).result().get_counts()
        energy.append(regular_qaoa(G, theta, A).compute_expectation(counts))

    idx = np.argmin(energy)
    t_max = time[idx]

    dt = t_max/p
    t = dt * (np.arange(1, p+1)-0.5)
    gamma = (t/t_max)*dt
    beta = (1-t/t_max)*dt
    theta_ini = np.concatenate((beta,gamma))
    
    return theta_ini

def salvage_theta_ini(theta_opt):

    """
    Salvage the previous optimal unitary parameters found by the QAOA algorithm to create new initial unitary parameters for the QAOA algorithm. Since this function uses the previous optimal parameters, it needs to be used only when starting from a circuit depth of one. Then, this function can be reused as the depth increases gradually.

    Args:
        theta_opt: list of previous optimal unitary parameters

    Returns:
        theta_ini: list of new initial unitary parameters
    """

    #for a new circuit with a depth of one
    if len(theta_opt) == 0:
            theta_ini = np.hstack((np.random.rand(1)*np.pi,np.random.rand(1)*np.pi*2))

    #for a circuit with a depth higher than one
    else:
        temp = np.hstack((np.random.rand(1)*np.pi,np.random.rand(1)*np.pi*2))
        theta_ini = np.hstack((theta_opt, temp))
    
    return theta_ini

def gen_rand_3sat(numvar, numcau, caudeg=3):

    """
    Generates a random 3SAT formula based on the number of variables and causes specified.

    Args:
        numvar: number of variables
        numcau: number of clauses
        caudeg: degree of clauses

    Returns:
        cf: 3SAT formula
    """

    rng = np.random.default_rng()

    cf = []
    for i in range(numcau):
        cf.append(list(rng.choice(numvar, size=caudeg, replace=False)+1))

    return np.array(cf)

def test_3sat_state(x, cf):

    """
    Determines if a particular state is a solution of the NAE3SAT formula.
    
    Args:
        x: str
           state bistring
        cf: NAE3SAT formula

    Returns:
        sol: bool
    """

    solver = Glucose4()
    solver.append_formula(cf.tolist())

    state = np.zeros(len(x), dtype=int)
    for iter, bit in enumerate(x):
        if bit == str(1):
            state[iter] = iter+1

        else:
            state[iter] = -iter-1

    sol = solver.solve(assumptions=state.tolist())

    solver.delete()

    return sol

def solve_sharp_3sat(cf, numvar):

    """
    Solve the given #3SAT problem and give the solutions of the problem. It returns the probability distribution of the solutions assuming that each solution has an equal probability. It also returns a list of all the solution where each solutions is given under the integer form.  

    Args:
        cf: 3SAT formula
        numvar: number of variables
    
    Returns:
        sol_prob: list
                  probability distribution of the solutions
        sol_int: list
                 solutions under the integer form
    """

    solver = Glucose4()
    solver.append_formula(cf.tolist())

    sol = []
    sol_prob = np.zeros(2**numvar)

    for m in solver.enum_models():
        sol.append(m)

    solver.delete()

    sol_bool = np.greater(np.array(sol), 0)

    #for an unsatiable problem
    if len(sol_bool) == 0:
        sol_int = []
        sol_prob = np.zeros(2**numvar)

    #for a satiable problem
    else:
        #pack the bits from the boolean form to the interger form
        sol_int = sol_bool.dot(2**np.arange(np.shape(sol_bool)[1])[::-1])
        sol_prob[sol_int] = 1/len(sol_int)

    return sol_prob, sol_int

def counting(G, counts):

    """
    Approximate the number of solutions using the samples from the final QAOA circuit. This function need samples from the QAOA circuit after the end of the optimization process.

    Args:
        G: graph object
        counts: dict
                key as bitstring, val as count

    Returns:
        num_sol: approximative number of solutions
    """

    numqubit = G.numnodes
    bitstrings = list(counts.keys())
    num_leaf_subt = np.zeros(numqubit)
    choice_subt = ""
    count_sol = 0
    truth_arr = np.zeros(len(bitstrings))

    for iter, bitstring in enumerate(bitstrings):
        truth_arr[iter] = test_3sat_state(bitstring, G.cf_nae)

    for i in range(numqubit):
        count_one = 0
        count_zero = 0

        for iter, j in enumerate(bitstrings):

            #Consider only the solutions
            if truth_arr[iter] == True:

                if i == 0:
                    count_sol += counts[j]

                if j[0:i+1] == (choice_subt + str(1)):
                    count_one += counts[j]
                elif j[0:i+1] == (choice_subt + str(0)):
                    count_zero += counts[j]

        if count_one >= count_zero:
            num_leaf_subt[i] += count_one
            choice_subt += "1"
        else:
            num_leaf_subt[i] += count_zero
            choice_subt += "0"

    num_sol = count_sol/num_leaf_subt[0]
    for i in range(numqubit-1):
        num_sol *= num_leaf_subt[i]/num_leaf_subt[i+1]

    return num_sol

def find_prob_sol(G, counts):

    """
    Find the probability that the samples of the final QAOA circuit are solutions of the NAE3SAT problem. This function need samples from the QAOA circuit after the end of the optimization process.

    Args:
        G: graph object
        counts: dict
                key as bitstring, val as count

    Returns:
        prob_sol: probability that a QAOA state is a NAE3SAT solution
    """

    num_sol = 0
    count_sol = 0

    for bitstring, value in counts.items():
        if test_3sat_state(bitstring, G.cf_nae)==True:
            num_sol += 1
            count_sol += value
    
    prob_sol = count_sol/sum(counts.values())
    #ratio_sol = num_sol/true_num_sol

    return prob_sol

def find_tvi(exact_prob, approx_prob):

    """
    Calculate the total variation distance between the approximative probability distribution of the solutions found by the QAOA algorithm and the exact probability distribution of the solutions. This is equivalent to the distance between the probability distribution of the solutions and the uniform probability distribution.

    Args:
        exact_prob: exact probability distribution of the solutions
        approx_prob: approximative probability distribution of the solutions
    """

    return sum(abs(approx_prob - exact_prob))/2

def clustering_1D(values):

    """
    UNSTABLE. Creates clusters of values based on the euclidian distance between them using Kernel Density. Calculate the ratio of the number of value inside the smallest cluster over the number of all possible values. 

    Args:
        values: list of values to clusters

    Returns: 
        rate: ratio of the size of the smallest cluster on the number of
              possible values
    """

    temp = values.reshape(-1,1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(temp)
    s = np.linspace(0,1)
    e = kde.score_samples(s.reshape(-1,1))
    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]

    if len(s[mi]) != 0: 
        rate = len(temp[temp < s[mi][0]])/len(temp)*100
    else:
        rate = np.NaN

    return rate
