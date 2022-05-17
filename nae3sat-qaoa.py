#################################################################
#                                                               #
#   SOLVING NAE3SAT USING QAOA ALGORITHM                        #
#   ==========================================                  #
#   ( nae3sat-qaoa.py )                                         #
#   First instance: 20220517                                    #
#   Written by Julien Drapeau                                   #
#                                                               #
#   This script sample a random instance of a bicubic graph     #
#   representing a #NAE3SAT problem. Then, it finds an          #
#   approximate solution using the QAOA algorithm and           #
#   benchmark it with an exact solution given by a tensor       #
#   network contraction (tensorcsp.py).                         #
#                                                               #
#   DEPENDENCIES: tensorcsp.py, scipy, qiskit, matplotlib,      #
#                 bipartite-graph-sampling                      #
#                                                               #
#################################################################


from readline import get_history_item
import sys  
import subprocess           
sys.dont_write_bytecode = True
from timeit import default_timer
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit import Aer
from qiskit.visualization import plot_histogram
from tensorcsp import * 


def loss(x, G, A):
    
    """
    Given a bitstring as a solution, this function return 
    the loss of the solution.
    
    Args:
        x: str
           solution bitstring
        G: igraph graph
        A: parameter of the ising model
        
    Returns:
        obj: float
             Objective
    """
    
    loss = 0
    for i, j in G.get_edgelist():
        loss += A*(2*int(x[i])-1)*(2*int(x[j])-1)
    
    return loss

def compute_expectation(counts, G, A):
    
    """
    Computes expectation value based on measurement results
    
    Args:
        counts: dict
                key as bitstring, val as count
        G: igraph graph
        A: parameter of the ising model
        
    Returns:
        avg: float
             expectation value
    """
    
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        obj = loss(bitstring, G, A)
        avg += obj*count
        sum_count += count
    
    return avg/sum_count
    
def create_qaoa_circ(G, theta):
    
    """
    Creates a parametrized qaoa circuit
    
    Args:
        G: igraph graph
        theta: list of unitary parameters
    
    Returns:
        qc: qiskit circuit
    """
    
    nqubits = len(G.vs.indices)
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
        for pair in list(G.get_edgelist()):
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
    Runs parametrized circuit
    
    Args:
        G: igraph graph
        A: parameter of the ising model
    """
    
    backend = Aer.get_backend('qasm_simulator')
    backend.shots = shots
    
    def execute_circ(theta):
        qc = create_qaoa_circ(G, theta)
        counts = backend.run(qc, seed_simulator=10, nshots=512).result().get_counts()
        
        return compute_expectation(counts, G, A)
    
    return execute_circ


#BICUBIC GRAPH SAMPLING TO REPRESENT RANDOM INSTANCES OF #NAE3SAT


numvar = 5   #number of variables
numcst = 5   #number of causes
vardeg = 3   #variables degree
cstdeg = 3   #causes degree
seed = 66666

#use bipartite-graph-sampling to sample a graph 
#(to run in \bipartite-graph-sampling-master\cli directory)
subprocess.run(['cargo', 'run', '--',
                '-n', str(numvar),
                '-m', str(numcst),
                '-c', str(cstdeg),
                '-v', str(vardeg),
                '-r', str(seed),
                '-o', 'tmp.txt'])

with open('tmp.txt') as file:
    graph = eval(file.read())

#CNF formula of 3SAT
cf = array(graph['graph']['constraint_neighbors'])+1
#CNF formula of NAE3SAT
cf_nae = vstack((cf, invert(cf)+1))

print('The NAE3SAT formula is: \n',cf_nae)
print()

edges = []
for i in cf:
    edges.append([i[0],i[1]])
    edges.append([i[1],i[2]])
    edges.append([i[2],i[0]])

#ising formulation graph
G_ising = Graph(n=graph['number_of_variables'], edges=array(edges)-1)
#CNF formulation graph
G_cnf = cnf_graph(cf)

#fig, ax = plt.subplots()
#plot(G_ising, target=ax, vertex_label=array(G_ising.vs.indices))

#fig, ax = plt.subplots()
#plot(G_cnf, target=ax, vertex_label=array(G_cnf.vs.indices))


#SOLVING #NAE3SAT WITH QAOA


p = 20    #number of turns QAOA
A = 1    #Ising model parameter
shots = 512

expectation = get_expectation(G_ising, A)
res = minimize(expectation, ones(2*p), method='COBYLA')

backend = Aer.get_backend('aer_simulator')
backend.shots = shots

qc_res = create_qaoa_circ(G_ising, res.x)
counts = backend.run(qc_res, seed_simulator=10).result().get_counts()
counts_values = array(list(counts.values()),dtype=int)

plot_histogram(counts, title='#NAE3SAT QAOA of a random bicubic graph with p = '+str(p))

k = 0
median_counts = median(counts_values)
for i in counts_values:
    if i > median_counts/2:
        k += 1

print('Solved with QAOA with very roughly', k, 'solutions. See the histogram for a more detail view.')
print()


#BENCHMARKING WITH TENSORCSP


__ = cnf_write(cf_nae,"tmp.cnf")   
tg = cnf_tngraph(cf_nae,dtype=int)  

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

plt.show()