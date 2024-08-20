import numpy as np # tested with v.1.26.4
import pulp # tested with v.2.7.0
import networkx as nx # tested with v.2.8.8
import pandas as pd # tested with v.2.1.4
import sys
import time
from scipy import sparse # tested with v.1.11.4
import gc

# ClearingVector class 
# 
# function solve performs full calculation of clearing vector.
# Class includes following options for calculations:
# 1. Debt contracts only (Eisenberg-Noe) - nominal liability matrix should be provided
# 2. Bankruptcy costs (Jackson-Pernoud) - to enable bankrupt_costs variable should be set to True
# 3. Equity links (Jackson-Pernoud) - a network with equity links should be provided

class ClearingVector:

    def __init__(self, liab_matrix, equity_network=0, bankrupt_costs=False, costs_value=1.5e6):
        self.bankrupt_costs = bankrupt_costs
        self.costs_value = costs_value
        self.liab_matrix = liab_matrix
        self.n_num = liab_matrix.shape[0]
        self.equity_network = equity_network

        self.nodes_debt = np.asarray(self.liab_matrix.sum(axis=1))
        self.nodes_assets = np.asarray(self.liab_matrix.sum(axis=0).T)
        self.liab_matrix_tmp = liab_matrix.toarray()
        
        # make relative liabilities matrix
        self.relative_liab_tmp = np.divide(
            self.liab_matrix_tmp, self.nodes_debt, out=np.zeros_like(self.liab_matrix_tmp),
            where=self.nodes_debt!=0
        )
        
        # free-up memory
        del self.liab_matrix_tmp
        gc.collect()
        
        # transform relative liability matrix into sparse data format
        self.relative_liab = sparse.csc_matrix(self.relative_liab_tmp).tocsc(copy=True)

        # free-up memory
        del self.relative_liab_tmp
        gc.collect()

        # vector depending on liabilities to assets ratio
        self.cf_vector = self.create_cf_vector(self.n_num)

# cashflow vector generation
# external cashflow is calculated so that ratio of external CF + assets / liabilities is ca. 0.93 on average
# based on normal distribution with mean of 0.93 and std 0.26

    def create_cf_vector(self, num):
            result_vector = np.ndarray(num)
            for i in range(num):
                ratio = np.random.normal(loc=0.93, scale=0.26, size=1)
                
                total_assets = self.nodes_debt[i] / ratio

                external_cf = total_assets - self.nodes_assets[i]

                result_vector[i] = np.maximum(20e5, external_cf)[0]

            return(result_vector)

# calculation of clearing vector using linear programming methods (puLP package)
# main function to run the model

    def clearing_vector_calc(self):
        ext_earn_vector = self.cf_vector
    # initialize model
        model = pulp.LpProblem("Clearing_vector", pulp.LpMaximize)

    # define decision variables with lower and upper bounds set so that
    # payment can't exceed node's total debt and can't be less than 0
        P = [pulp.LpVariable(f"P_{x}", 0, self.nodes_debt[x]) for x in range(self.n_num)]

    # define objective function - sum of payments
        model += pulp.lpSum(P[i] for i in range(self.n_num)), "Sum of payments"

    # define contraints
        start_time = time.time()

        # problem with equity links
        if self.equity_network != 0:
            for i in range(self.n_num):
                slice = sparse.find(self.relative_liab[:, i])
                fire_sale_flag = self.fire_sale(i)
                model += P[i] <= self.equity_value(i) + fire_sale_flag * ext_earn_vector[i] + self.default_costs(i) + fire_sale_flag * pulp.lpSum([slice[2][j[0]] * P[j[1]] for j in enumerate(slice[0])]), f"Node {i} assets with equity constraint"

        # problem with realized failure costs
        elif self.bankrupt_costs == True:
            for i in range(self.n_num):
                slice = sparse.find(self.relative_liab[:, i])
                fire_sale_flag = self.fire_sale(i)
                model += P[i] <= fire_sale_flag * ext_earn_vector[i] + self.default_costs(i) + fire_sale_flag * pulp.lpSum([slice[2][j[0]] * P[j[1]] for j in enumerate(slice[0])]), f"Node {i} assets constraint"
        
        # debt only problem
        else:
            for i in range(self.n_num):
                slice = sparse.find(self.relative_liab[:, i])
                model += P[i] <= ext_earn_vector[i] + pulp.lpSum([slice[2][j[0]] * P[j[1]] for j in enumerate(slice[0])]), f"Node {i} payment constraint - assets + external vector"
        print(f'Model building {time.time() - start_time}')
        
        start_time = time.time()
        model.solve()
        print(f'Model solving {time.time() - start_time}')
    
    # save clearing vector
        output = [x.varValue for x in P]
        
        return(model.status, output)

# function defines discount for assets sale in case of bank default
    def fire_sale(self, i):
        
        if self.nodes_assets[i] + self.cf_vector[i] + self.equity_value(i) > self.nodes_debt[i]:
            output = 1
        
        else:
            output = 0.8
        
        return(output)
    
# function defines fixed part of failure costs
    def default_costs(self, i):
        
        if self.nodes_assets[i] + self.cf_vector[i] + self.equity_value(i) > self.nodes_debt[i]:
            output = -self.costs_value
        
        else:
            output = 0

        return(output)

# function calculates value of given node investments
    def equity_value(self, i):
        
        investment_value = 0
        
        # checking whether there is equity network. If not return 0.
        if self.equity_network == 0:
            
            return(investment_value)
        # checking whether the node has any connections. If not return 0.
        if self.equity_network.degree(i) == 0:
            return(investment_value)
        
        # calculate investments value
        else:
            for v in self.equity_network.neighbors(i):
        # extract shareholding
                edge_data = self.equity_network.get_edge_data(i, v)
                share = edge_data[0]['equity']
                
        # calculate target's net worth
                node_net_worth = np.maximum(
            (self.cf_vector[v] + self.nodes_assets[v] - self.nodes_debt[v]), 0
                )
                
                investment_value += round(share * node_net_worth[0], 2)
        return(investment_value)    
    
  
    
# generation of nominal liabilities matrix based on graph network
# class variables are:
# - n_num - number of nodes in the network
# - core_size - share of core nodes
# - equity - True for generation of equity network, False for liability matrix
class GenNetwork:
    
    def __init__(self, n_num=10, core_size=0.1, equity=False):
        self.n_num = n_num
        self.core_size = core_size
        self.equity_flag = equity


# function to generate core to periphery network
# needs following inputs:
# 1. number of core nodes (n_core)
# 2. number of periphery nodes (n_periphery)
# 3. probabilities of core-to-core, core-to-periphery and intraperiphery connections (p_cc, p_cp and p_pp respectively)

    def generate_network(self, n_core, n_periphery, p_cc=0.8, p_cp=0.6, p_pp=0.2):
        G = nx.MultiDiGraph()

        # Add core and periphery nodes
        core_nodes = range(n_core)
        periphery_nodes = range(n_core, n_core + n_periphery)
        G.add_nodes_from(core_nodes, bipartite=0)
        G.add_nodes_from(periphery_nodes, bipartite=1)

    # limits for debt generation
        upperbound = 1e9
        lowerbound = 20e6

    # Connect core nodes
        for i in core_nodes:
            for j in core_nodes:
                if i != j and np.random.random() < p_cc and G.has_edge(j, i) is False:
                    G.add_edge(i, j, debt=round(np.random.uniform(low=lowerbound, high=upperbound), 2))

    # Connect core and periphery nodes
        for i in core_nodes:
            for j in periphery_nodes:
                if np.random.random() < p_cp and G.has_edge(j, i) is False:
                    G.add_edge(i, j, debt=round(np.random.uniform(low=lowerbound/2, high=upperbound/2), 2))

            # Connect periphery nodes
        for i in periphery_nodes:
            for j in periphery_nodes:
                if i != j and np.random.random() < p_pp and G.has_edge(j, i) is False:
                    G.add_edge(i, j, debt=round(np.random.uniform(low=lowerbound/4, high=upperbound/4), 2))

   # sanity check. If there are nodes not connected to any other, make random connections
    
        for node in G.nodes():
            if G.degree(node) == 0:
                node_list = np.random.choice(
                    list(np.arange(0, self.n_num, dtype='int')), size=np.maximum(1, int(self.n_num*0.05)), replace=False
                )
                for j in node_list:
                    G.add_edge(node, j, debt=round(np.random.uniform(low=lowerbound, high=upperbound, size=1)[0], 2))

        return G

    # function to generate netwok of equity links. links_pp variable sets share of nodes with links
    def generate_equity_network(self, links_pp=0.2):
        
        # define number of nodes to be connected
        connected_nodes_num = int(np.maximum(2, self.n_num * links_pp))
        
        if connected_nodes_num % 2 != 0:
            connected_nodes_num += 1
        
        all_nodes = [x for x in range(self.n_num)]
        
        # randomly select nodes to be connected
        selected_nodes = np.random.choice(all_nodes, size=connected_nodes_num, replace=False)
        
        # split the list of nodes into shareholders and targets
        shareholders = selected_nodes[0:int(len(selected_nodes)/2)]
        targets = selected_nodes[int(len(selected_nodes)/2):]
        
        G = nx.MultiDiGraph()
        
        G.add_nodes_from(range(0, self.n_num))
        
        for u, v in zip(shareholders, targets):
            # adding connection between nodes. Edge value represents shareholding randomly generated,
            G.add_edge(u, v, equity=round(np.random.uniform(low=0.1, high=1), 2))
        
        return G
        
    
    
    # transform graph network to scipy sparse array
    def graph_to_matrix(self, network):

        matrix = np.zeros((self.n_num, self.n_num))

    # chosing data type to extract values        
        if self.equity_flag == 1:
            data_type = "equity"
        else:
            data_type = "debt"
        
        for u, v, keys, data in network.edges(data=data_type, keys=True):
            matrix[v][u] = data

        sparse_matrix = sparse.csc_matrix(matrix).tocsc()
        del matrix
        gc.collect()

        return(sparse_matrix)

    # main function of the class. Creates network and transforms in into matrix form. Returns the matrix and number of edges
    def create_matrix_network(self):
        n_core = int(np.minimum(1, self.n_num * self.core_size))
        n_periphery = self.n_num - n_core
        
        if self.equity_flag == 0:        
            network = self.generate_network(n_core, n_periphery)
        else:
            network = self.generate_equity_network()
        
        matrix = self.graph_to_matrix(network)

        return (matrix, network.number_of_edges())


# function for running performance tests    
def testing():

# settings
# number of test runs for each network size
    number_of_iterations = 3
    lower_bound_for_network = 500
    upper_bound_for_network = 1000
    network_growth_step = 500

# dataframe to store results
    results = pd.DataFrame(columns=['problem', 'network_size', 'exec_time', 'edges'])

    
    for net_size in range(lower_bound_for_network, upper_bound_for_network+network_growth_step, network_growth_step):
        
        print(f'Network size: {net_size}')

        for i in range(number_of_iterations):
        
            (net, edges_number) = GenNetwork(n_num=net_size).create_matrix_network()
            G = GenNetwork(n_num=net_size, equity=True).generate_equity_network()

# Eisenber-Noe         
            print(f'Iteration: {i} - Debt only')
            cv = ClearingVector(liab_matrix=net, bankrupt_costs=False)
            start_time = time.time()
        
            cv.clearing_vector_calc()
        
            end_time = time.time()
            
            results.loc[len(results)] = ['Debt-only', net_size, (end_time - start_time), edges_number]

# with bankruptcy costs
            print(f'Iteration: {i} - with failure costs')
            cv = ClearingVector(liab_matrix=net, bankrupt_costs=True)
            start_time = time.time()
        
            cv.clearing_vector_calc()
        
            end_time = time.time()
            
            results.loc[len(results)] = ['With costs', net_size, (end_time - start_time), edges_number]

# Equity            

            print(f'Iteration: {i} - with equity links')
            cv = ClearingVector(liab_matrix=net, equity_network=G, bankrupt_costs=True)
            start_time = time.time()
        
            cv.clearing_vector_calc()
        
            end_time = time.time()
            
            results.loc[len(results)] = ['With equity', net_size, (end_time - start_time), edges_number]

        gc.collect()
            
    return(results)

# one sample run of the algorithm with selected type of problem and network size
# by default - debt contracts only. Network size is 500
# output clearing vector is saved into a csv file
def random_network_calculation(choice=0, net_size=500):

    (net, edges_number) = GenNetwork(n_num=net_size).create_matrix_network()

    if choice == 0:
        print(f'Debt only contracts. Network size {net_size}')
        cv = ClearingVector(liab_matrix=net)
    elif choice == 1:
        print(f'With realized failure costs. Network size {net_size}')
        cv = ClearingVector(liab_matrix=net, bankrupt_costs=True)
    else:
        print(f'With equity links. Network size {net_size}')
        G = GenNetwork(n_num=net_size, equity=True).generate_equity_network()
        cv = ClearingVector(liab_matrix=net, equity_network=G, bankrupt_costs=True)

    (status, vector) = cv.clearing_vector_calc()
    pd.DataFrame(vector).to_csv('output_vector.csv')



# perform automated test with various network sizes. Results are saved into csv-file
#testing().to_csv('test_results.csv', header=True)

# sample run of the solver
# Set choice variable to change the problem type: 0 - for debt only, 1 - for realized failure costs, 2 for equity links
# net_size sets the number of nodes in the network
random_network_calculation(choice=0, net_size=500)