import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from itertools import combinations

class USsupremeCourt(): 
    def __init__(self):
        pass

    def load_data(self, file_path): 
        """load time series data and return the time interval"""
        with open(file_path, "r") as f:
            data = f.readlines()
        cleaned_data = [line.strip() for line in data]

        return cleaned_data
    
    def clean_data(self, file_path_s, file_path_h, file_path_J): 
        self.US_data = self.load_data(file_path_s)
        self.s_mat = np.array([[int(char) for char in line] for line in self.US_data])
        self.s_mat = 2*self.s_mat-1
        self.s_mat_set = np.unique(self.s_mat, axis=0)

        h_data = self.load_data(file_path_h)
        self.h_vec = np.array([float(x) for x in h_data])
        self.n = len(self.h_vec)
        
        J_data = self.load_data(file_path_J)
        Jij_floats = [float(x) for x in J_data]
        index = 0
        self.J_mat= np.zeros((self.n,self.n))
        for i in range(self.n):
            for j in range(i+1, self.n):
                self.J_mat[i,j] = Jij_floats[index]
                self.J_mat[j,i] = Jij_floats[index]  
                index += 1
        
        self.s_list = np.array(list(itertools.product([-1, 1], repeat=self.n)))

    def calculate_empirical_average_s(self, matrix): 
        s_D = np.mean(matrix, axis=0)
        return s_D

    def calculate_empirical_average_ss(self, matrix): 
        ss_D = np.dot(matrix.T, matrix) / matrix.shape[0]
        return ss_D

    def calculate_energy(self, s_mat, h_vec, J_mat): 
        energy = -np.dot(s_mat, h_vec) -0.5 * np.einsum('ij,jk,ik->i', s_mat, J_mat, s_mat)
        return energy
    
    def calculate_partition_function(self, h_vec, J_mat): 
        energies = self.calculate_energy(self.s_list, h_vec, J_mat)
        self.Z = np.sum(np.exp(-energies))
    
    def calculate_empirical_probability(self, matrix, state_set):
        prob_empirical = []
        for element in state_set:
            matches = np.all(matrix == element, axis=1)
            prob_empirical.append(float(np.sum(matches) / matrix.shape[0]))
        return prob_empirical
    
    def calculate_analytical_probability(self, h_vec, J_mat, state_set): 
        self.calculate_partition_function(h_vec, J_mat)
        energies = self.calculate_energy(state_set, h_vec, J_mat) 
        prob_anal = np.exp(-energies) / self.Z                  
        return prob_anal
    
    def matrix_to_vector(self, mat):
        assert mat.shape == (9, 9), "Input must be 9x9 matrix"
        triu_indices = np.triu_indices(9, k=1) 
        return mat[triu_indices]
    
    def calculate_voting_rate(self, mat): 
        return np.mean(mat == 1, axis=0)
    
    def calculate_voting_prob_from_no_coupling_model(self, p, k):
        indices = range(len(p))
        prob = 0.0
        for S in combinations(indices, k):
            prod = 1.0
            for i in indices:
                if i in S:
                    prod *= p[i]
                else:
                    prod *= (1 - p[i])
            prob += prod
        print(f"P({k} conservative vote from no coupling model) = {float(prod):.4f}")
        return float(prob)
    
    def calculate_voting_prob_from_data(self, mat, k): 
        num_ones_per_row = np.sum(mat == 1, axis=1)
        prob_k = np.mean(num_ones_per_row == k)
        print(f"P({k} conservative vote from data) = {prob_k:.4f}")
        return prob_k
    
    def calculate_voting_prob_from_ising_model(self, h_vec, J_mat, k): 
        prob_ana_all = self.calculate_analytical_probability(h_vec, J_mat, self.s_list)
        num_ones_per_state = np.sum(self.s_list == 1, axis=1)
        prob_k = np.sum(prob_ana_all[num_ones_per_state == k])
        print(f"P({k} conservative vote{'s' if k != 1 else ''} from Ising model) = {prob_k:.4f}")
        return float(prob_k)

def plot_scatter(x, y, color='b', Q="Q6_3", label=None, file_name=None): 
    plt.figure(figsize=(7, 6), dpi=300)
    plt.scatter(x, y, color=color, label=label)

    if Q == "Q6_3": 
        plt.xlabel('i', fontsize=14)
        plt.ylabel(fr'$\langle s_i \rangle_D$', fontsize=14)
        plt.title(fr'Scatter Plot of i vs. $\langle s_i \rangle_D$', fontsize=16)
    elif Q == "Q6_4":
        plt.xlabel('Reordered i', fontsize=14) 
        plt.ylabel(fr'$h_i$', fontsize=14)
        plt.title(fr'Scatter Plot of i vs. $h_i$', fontsize=16)
    elif Q == "Q6_5": 
        plt.plot([0, 0.25], [0, 0.25], color='r', linestyle='--', alpha=0.7, linewidth=2, label='reference line')
        plt.xlabel(r'$p_g(s)$', fontsize=14)
        plt.ylabel(r'$p_D(s)$', fontsize=14)
        plt.title(fr'Cross-validation: model probability ($p_g(s)$) vs. empirical probability ($p_D(s)$)' )
        plt.grid()
    elif Q == "Q6_6": 
        max_num = max(x)+0.05
        min_num = min(x)-0.05
        plt.plot([min_num, max_num], [min_num, max_num], color='r', linestyle='--', alpha=0.7, linewidth=2, label='reference line')
        plt.title(fr'Checking the fit: model average of $s_i$ ($\langle s_i \rangle$) vs. empirical average of $s_i$ ($\langle s_i \rangle_D$) ' )
        plt.xlabel(fr'$\langle s_i \rangle$', fontsize=14)
        plt.ylabel(fr'$\langle s_i \rangle_D$', fontsize=14)
        plt.grid()
    elif Q == "Q6_6_2": 
        max_num = max(x)+0.05
        min_num = min(x)-0.05
        plt.plot([min_num, max_num], [min_num, max_num], color='r', linestyle='--', alpha=0.7, linewidth=2, label='reference line')
        plt.title(fr'Checking the fit: model average of $s_is_j$ ($\langle s_is_j \rangle$) vs. empirical average of $s_is_j$ ($\langle s_is_j \rangle_D$) ' )
        plt.xlabel(fr'$\langle s_is_j \rangle$', fontsize=14)
        plt.ylabel(fr'$\langle s_is_j \rangle_D$', fontsize=14)
        plt.grid()
    elif Q == "Q6_7": 
        plt.grid()   

    plt.legend()

    if file_name: 
        plt.savefig(file_name, dpi=300)
    else: 
        plt.show()

def plot_heatmap(mat, cmap="Greys", Q="Q6_3", file_name=None): 
    # labels = ['JS', 'RG', 'DS', 'SB', 'SO', 'AK', 'WR', 'AS', 'CT']
    labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

    plt.figure(figsize=(7, 6), dpi=300)
    if Q == "Q6_3": 
        sns.heatmap(
            mat,
            cmap=cmap,            
            annot=False,           
            fmt=".2f",            
            square=True,
            cbar_kws={'label': 'Value'},
            xticklabels=labels,
            yticklabels=labels
        )
    else: 
        sns.heatmap(
            mat,
            cmap=cmap,            
            annot=False,           
            fmt=".2f",            
            square=True,
            cbar_kws={'label': 'Value'},
            xticklabels=labels,
            yticklabels=labels, 
            vmin=-1,    
            vmax=1 
        )
     
    if Q=="Q6_3":
        plt.title(fr'Correlation matrix $\langle s_is_j \rangle_D$ of ideological votes', fontsize=14)
    else: 
        plt.title(r'Effective interactions $J_{ij}$', fontsize=14)
    plt.xlabel("i", fontsize=14)
    plt.ylabel("i", fontsize=14)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if file_name: 
        plt.savefig(file_name, dpi=300)
    else: 
        plt.show()

def plot_line(y, color='b', label=None): 
    plt.plot(range(len(y)), y, marker='o', linestyle='-', color=color, label=label)
    plt.xlabel('k conservative notes', fontsize=14)
    plt.ylabel('P(k)', fontsize=14)
    plt.legend()
    plt.title("Probability of k conservative votes", fontsize=16)
    plt.grid(True)

if __name__ == "__main__": 
    file_path_s = "../data/US_SupremeCourt_n9_N895.txt"
    file_path_h = "../data/hi_ussc_unsorted.txt"
    file_path_J = "../data/Jij_ussc_unsorted.txt"

    Q3 = USsupremeCourt()
    Q3.clean_data(file_path_s, file_path_h, file_path_J)
    n = Q3.n

#     # Q6.1
#     print("n", n)
#     print(fr"$2^n$", 2**n)
#     print("N", len(Q3.US_data))
#     print("N_max", len(set(Q3.US_data)))

    # Q6.3
    s_D = Q3.calculate_empirical_average_s(Q3.s_mat)
    new_index = np.argsort(s_D)
    s_mat = Q3.s_mat[:,new_index]
    ss_D = Q3.calculate_empirical_average_ss(s_mat)

    # plot_scatter(range(len(s_D)), s_D)
    # plot_scatter(range(len(s_D[new_index])), s_D[new_index])
    # plot_heatmap(ss_D)

#     # Q6.4
    h_vec = Q3.h_vec[new_index]
    J_mat = Q3.J_mat[new_index][:, new_index]
#     plot_scatter(range(len(h_vec)), h_vec, Q="Q6_4")
#     plot_heatmap(J_mat, cmap="coolwarm", Q="Q6_4")

#     # Q6.5
#     prob_emp = Q3.calculate_empirical_probability(s_mat, Q3.s_mat_set[:,new_index])
#     prob_ana = Q3.calculate_analytical_probability(h_vec, J_mat, Q3.s_mat_set[:,new_index])
#     plot_scatter(prob_ana, prob_emp, Q="Q6_5", label="datapoints")

#     # Q6.6
    # s_list = Q3.s_list
    # p_g = Q3.calculate_analytical_probability(h_vec, J_mat, s_list)
    # mean_si = np.sum(s_list * p_g[:, None], axis=0)  
    # mean_sisj = np.sum(
    #     (s_list[:, :, None] * s_list[:, None, :]) * p_g[:, None, None],
    #     axis=0
    # )  
    # print(mean_si)
    # print(np.mean(s_mat, axis=0))
    # plot_scatter(mean_si, np.mean(s_mat, axis=0), Q="Q6_6", label="datapoints")
    # plot_scatter(Q3.matrix_to_vector(mean_sisj), Q3.matrix_to_vector(ss_D), Q="Q6_6", label="datapoints")

#     # Q6.7
#     prob_cons = Q3.calculate_voting_rate(s_mat)
#     P = [Q3.calculate_voting_prob_from_no_coupling_model(prob_cons, k) for k in range(n+1)]

#     plt.figure(figsize=(7, 5))
#     plot_line(P, label=fr"$P_I(k)$")
#     plt.show()

#     # Q6.8
#     P_data = [Q3.calculate_voting_prob_from_data(s_mat, k) for k in range(n+1)]
#     print(max(P_data))

#     plt.figure(figsize=(7, 5))
#     plot_line(P, label=fr"$P_I(k)$")
#     plot_line(P_data, color="green", label=fr"$P_D(k)$")
#     plt.show()

#     # Q6.9
#     P_ising = [Q3.calculate_voting_prob_from_ising_model(h_vec, J_mat, k) for k in range(n+1)]
#     plt.figure(figsize=(7, 5))
#     plot_line(P, label=fr"$P_I(k)$")
#     plot_line(P_data, color="green", label=fr"$P_D(k)$")
#     plot_line(P_ising, color="r", label=fr"$P_P(k)$")
#     plt.show()
