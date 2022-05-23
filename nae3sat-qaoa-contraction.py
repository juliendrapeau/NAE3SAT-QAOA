#################################################################
#                                                               #
#   SOLVING #NAE3SAT USING QAOA CIRCUIT CONTRACTION             #
#   ==========================================                  #
#   ( nae3sat-qaoa.py )                                         #
#   First instance: 20220517                                    #
#   Written by Julien Drapeau                                   #
#                                                               #
#   This script samples a random instance of a bicubic graph    #
#   representing a #NAE3SAT problem. Then, it finds an          #
#   approximate solution by contracting the tensor network      #
#   of the QAOA circuit and benchmark it with an exact solution #
#   given by a tensor network contraction (tensorcsp.py).       #
#                                                               #
#   DEPENDENCIES: tensorcsp.py, qecstruct, matplotlib,          #
#                 contengra, quimb, skopt, qiskit, tqdm         #
#                                                               #
#################################################################


from timeit import default_timer
import tqdm
import matplotlib.pyplot as plt
import quimb as qu
import quimb.tensor as qtn
import cotengra as ctg
from skopt import Optimizer
from skopt.plots import plot_convergence, plot_objective
from tensorcsp import * 
import qecstruct as qs
from qiskit.visualization import plot_histogram


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


def energy(x):

    """
    Find the expectation value of the problem Hamiltonian with the circuit unitary parameters.

    Args:
        x: list of unitary parameters  
    """

    p = len(x) // 2   #number of alternating unitaries
    gammas = x[:p]
    betas = x[p:]
    circ = qtn.circ_qaoa(terms, p, gammas, betas)

    ZZ = qu.pauli('Z') & qu.pauli('Z')
    ens = [
        circ.local_expectation(weight * ZZ, edge, optimize=opt)
        for edge, weight in terms.items()
    ]

    return sum(ens).real

def minimize_energy_w_rehearse(G, p, A):

    """
    Minimize the expectation value of the problem Hamiltonian. The actual computation is rehearsed - the contraction widths and costs of each energy term is computed. 

    Args:
        G: graph object
        p: number of alternating unitaries
        A: parameter of the Ising model
    """

    gammas = qu.randn(p)
    betas = qu.rand(p)

    opt = ctg.ReusableHyperOptimizer(
    reconf_opts={},
    max_repeats=16,
    parallel=True
    )

    terms = {(i,j): A for i, j in G.edges}

    circ_ex = qtn.circ_qaoa(terms, p, gammas, betas)

    #rehearsal of the optimization
    ZZ = qu.pauli('Z') & qu.pauli('Z')
    local_exp_rehs = [
        circ_ex.local_expectation_rehearse(weight*ZZ, edge, optimize=opt, backend='torch') 
        for edge, weight in tqdm.tqdm(list(terms.items()))
        ]

    eps = 1e-6
    bounds = array(
        [(0.0        + eps, qu.pi / 2 - eps)] * p +
        [(-qu.pi / 4 + eps, qu.pi / 4 - eps)] * p
    )

    bopt = Optimizer(bounds)

    #minimize the energy
    for i in tqdm.trange(100):
        x = bopt.ask()
        res = bopt.tell(x, energy(x))

    return res

def minimize_energy_wo_rehearse(p):

    """
    Minimize the expectation value of the problem Hamiltonian. The actual computation is not rehearsed - the contraction widths and costs of each energy term are not pre-computed. 

    Args:
        p: number of alternating unitaries
    """

    gammas = qu.randn(p)
    betas = qu.rand(p)

    opt = ctg.ReusableHyperOptimizer(
        reconf_opts={},
        max_repeats=16,
        parallel=True
        )

    eps = 1e-6
    bounds = array(
        [(0.0        + eps, qu.pi / 2 - eps)] * p +
        [(-qu.pi / 4 + eps, qu.pi / 4 - eps)] * p
    )

    bopt = Optimizer(bounds)

    #minimize the energy
    for i in tqdm.trange(100):
        x = bopt.ask()
        res = bopt.tell(x, energy(x))

    return res

def run_qaoa_circ(G, p, res):

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

    terms = {(i,j): A for i, j in G.edges}

    circ_ex = qtn.circ_qaoa(terms, p, res.x[:p], res.x[p:])

    counts = circ_ex.simulate_counts(1024, reverse=True)

    return counts


#BICUBIC GRAPH SAMPLING TO REPRESENT RANDOM INSTANCES OF #NAE3SAT


numvar = 4    #number of variables
numcau = 4    #number of causes
vardeg = 3    #variables degree
caudeg = 3    #causes degree
seed = 666

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


#SOLVING #NAE3SAT WITH QAOA CIRCUIT CONTRACTION


A = 1   #Ising model parameter
p = 4   #number of turns QAOA
gammas = qu.randn(p)
betas = qu.rand(p)

opt = ctg.ReusableHyperOptimizer(
    reconf_opts={},
    max_repeats=16,
    parallel=True,
    )

terms = {(i,j): A for i, j in G.edges}

circ_ex = qtn.circ_qaoa(terms, p, gammas, betas)

circ_ex.psi.draw(color=['PSI0', 'H', 'RZZ', 'RX'])
circ_ex.get_rdm_lightcone_simplified([0]).draw(color=['PSI0', 'H', 'RZZ', 'RX'], highlight_inds=['k0', 'b0'])

#rehearse the computation of the tensor network contraction
ZZ = qu.pauli('Z') & qu.pauli('Z')
local_exp_rehs = [
    circ_ex.local_expectation_rehearse(weight*ZZ, edge, optimize=opt, backend='torch') 
    for edge, weight in tqdm.tqdm(list(terms.items()))
    ]

fig, ax1 = plt.subplots()
ax1.plot([rehs['W'] for rehs in local_exp_rehs], color='green')
ax1.set_ylabel('contraction width, $W$, [log2]', color='green')
ax1.tick_params(axis='y', labelcolor='green')

ax2 = ax1.twinx()
ax2.plot([rehs['C'] for rehs in local_exp_rehs], color='orange')
ax2.set_ylabel('contraction cost, $C$, [log10]', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
plt.show()

#minimize the energy
res = minimize_energy_wo_rehearse(p)

plot_convergence(res)
plot_objective(
    res,
    cmap='RdYlBu_r',
    dimensions=[f'$\\gamma_{i}$' for i in range(p)] + [f'$\\beta_{i}$' for i    in range(p)],
)

counts = run_qaoa_circ(G, p, res)

plot_histogram(counts, title='#NAE3SAT QAOA of a random bicubic graph\
with p = '+str(p))
plt.show()


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

plt.show()