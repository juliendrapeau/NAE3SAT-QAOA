#################################################################
#                                                               #
#   SOLVING #NAE3SAT USING QAOA ALGORITHM                       #
#   ==========================================                  #
#   ( nae3sat-qaoa.py )                                         #
#   First instance: 20220517                                    #
#   Written by Julien Drapeau                                   #
#                                                               #
#   This script samples a random instance of a bicubic graph    #
#   representing a #NAE3SAT problem. Then, it finds an          #
#   approximate solution using the QAOA algorithm and           #
#   benchmark it with an exact solution given by a tensor       #
#   network contraction (tensorcsp.py).                         #
#                                                               #
#   DEPENDENCIES: tensorcsp.py, scipy, qiskit, matplotlib,      #
#                 qecstruct                                     #
#                                                               #
#################################################################


from timeit import default_timer
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit import Aer
from qiskit.visualization import plot_histogram
from tensorcsp import * 
import qecstruct as qs


class graph:

    """
    This class instantiates a random bicubic graph representating a NAE3SAT problem using qecstruct.
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

        #CNF formula
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

        self.cf = array(cf)+1
        self.cf_nae = vstack((self.cf, invert(self.cf)+1))
        self.edges = array(edges)
        self.numnodes = numvar

    def cf(self):
        """
        CNF formula of 3SAT
        """
        return self.cf

    def cf_nae(self):
        """
        CNF formula of NAE3SAT
        """
        return self.cf_nae

    def numnodes(self):
        return self.numnodes

    def edges(self):
        return self.edges

    def ising_view(self):
        """
        Ising formulation graph
        """
        return Graph(n=self.numnodes, edges=self.edges)

    def cf_view(self):
        """
        CNF formulation graph
        """
        return cnf_graph(self.cf)


def compute_loss(x, G, A):
    
    """
    Given a bitstring as a solution, this function return 
    the loss of the solution.
    
    Args:
        x: str
           solution bitstring
        G: graph object
        A: parameter of the ising model
        
    Returns:
        obj: float
             Objective
    """
    
    loss = 0
    for i, j in G.edges:
        loss += A*(2*int(x[i])-1)*(2*int(x[j])-1)
    
    return loss

def compute_expectation(counts, G, A):
    
    """
    Computes expectation value based on measurement results.
    
    Args:
        counts: dict
                key as bitstring, val as count
        G: graph object
        A: parameter of the ising model
        
    Returns:
        avg: float
             expectation value
    """
    
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        obj = compute_loss(bitstring, G, A)
        avg += obj*count
        sum_count += count
    
    return avg/sum_count
    
def create_qaoa_circ(G, theta, A):
    
    """
    Creates a parametrized qaoa circuit.
    
    Args:
        G: graph object
        theta: list of unitary parameters
    
    Returns:
        qc: qiskit circuit
    """
    
    nqubits = G.numnodes
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
        for pair in G.edges:
            qc.rzz(A*gamma[irep], pair[0], pair[1])
            qc.barrier()
        
        qc.barrier()
        
        #driver unitary
        for i in range(0, nqubits):
            qc.rx(2*beta[irep], i)
        
    qc.measure_all()
        
    return qc
    
def get_expectation(G, A, shots=512):
    
    """
    Runs parametrized circuit.
    
    Args:
        G: graph object
        A: parameter of the ising model
    """
    
    backend = Aer.get_backend('qasm_simulator')
    backend.shots = shots
    
    def execute_circ(theta):
        qc = create_qaoa_circ(G, theta, A)
        counts = backend.run(qc, seed_simulator=10, nshots=512).result().get_counts()
        
        return compute_expectation(counts, G, A)
    
    return execute_circ

def run_qaoa_circ(G, p, A, shots=512, optimizer='COBYLA', simulator='aer_simulator', seed_simulator=10):

    """
    Minimize the expectation value by finding the best parameters.
    Analyse the results with a histogram.

    Args:
        G: graph object
        p: int
           number of alternating unitairies
        A: parameter of the ising model
        
    Returns:
        obj: dict
             counts
    """
    
    expectation = get_expectation(G, A)
    res = minimize(expectation, random.randn(2*p)/2+1, method=optimizer)

    backend = Aer.get_backend(simulator)
    backend.shots = shots

    qc_res = create_qaoa_circ(G, res.x, A)
    counts = backend.run(qc_res, seed_simulator=seed_simulator).result().get_counts()

    return counts


#BICUBIC GRAPH SAMPLING TO REPRESENT RANDOM INSTANCES OF #NAE3SAT


numvar = 4    #number of variables
numcau = 4    #number of causes
vardeg = 3    #variables degree
caudeg = 3    #causes degree
seed = 666

#samples a random bicubic graph
G = graph(numvar, numcau, vardeg, caudeg, seed)

print('The NAE3SAT formula is: \n', G.cf_nae)
print()

#fig, ax = plt.subplots()
#plt.title('Ising formulation graph')
#plot(G.ising_view(), target=ax, vertex_label=array(G.ising_view().vs.indices))

#fig, ax = plt.subplots()
#plt.title('CNF formulation graph')
#plot(G.cf_view(), target=ax, vertex_label=array(G.cf_view().vs.indices))
#plt.show()


#SOLVING #NAE3SAT WITH QAOA


p = 25   #number of turns QAOA
A = 1    #Ising model parameter
shots = 1024

counts = run_qaoa_circ(G, p, A, shots = shots)
counts_values = array(list(counts.values()),dtype=int)

plot_histogram(counts, title='#NAE3SAT QAOA of a random bicubic graph\
with p = '+str(p))
plt.show()

k = 0
median_counts = median(counts_values)
for i in counts_values:
    if i > median_counts/2:
        k += 1

print('Solved with QAOA with very roughly', k, 'solutions.\
See the histogram for a more detailled view.')
print()


#BENCHMARKING WITH TENSORCSP


__ = cnf_write(G.cf_nae,"tmp.cnf")   
tg = cnf_tngraph(G.cf_nae,dtype=int)  

#solve using a greedy contraction algorithm:
start = default_timer()
md,sg = contract_greedy(tg,combine_attrs=dict(attr=attr_contract))
end   = default_timer()
sol   = sg.vs[0]["attr"][1]
print('Solved with greedy contraction in ',end-start,' seconds')
print('  #Solutions:',sol)
print('  Max degree:',md.max())
print()

#solve using METIS graph partitioning:
start = default_timer()
m = recursive_bipartition(tg,metis_bipartition)
md,sg = contract_dendrogram(tg,m,combine_attrs=dict(attr=attr_contract))
end   = default_timer()
sol   = sg.vs[0]["attr"][1]
print('Solved with METIS partitioning in ',end-start,' seconds')
print('  #Solutions:',sol)
print('  Max degree:',md.max())
print()

#solve using Girvan-Newman community detection:
start = default_timer()
d = tg.community_edge_betweenness()
m = d.merges
md,sg = contract_dendrogram(tg,m,combine_attrs=dict(attr=attr_contract))
end   = default_timer()
sol   = sg.vs[0]["attr"][1]
print('Solved with Girvan-Newman in ',end-start,' seconds')
print('  #Solutions:',sol)
print('  Max degree:',md.max())
print()