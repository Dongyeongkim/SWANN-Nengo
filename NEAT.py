import time
import numpy as np
import random as rd


def translate_gene_into_nengo_param(gene_pool):
    gene_pool_list = []
    for gene in gene_pool:
        gene_node_list = []
        for nuc in gene:
            gene_node_list.append(nuc[0])
            gene_node_list.append(nuc[1])
        gene_node_list.sort()
        gene_connection_list = gene.copy()
        gene_pool_list.append([gene_node_list, gene_connection_list])
        f = open('gene/'+str(time.strftime('%c', time.localtime(time.time())))+'.txt', 'w')
        f.write(str(gene_pool_list))
        f.close()

    return gene_pool_list


def generate_first_generation(gene_pool_size, input_neuron, output_neuron):
    gene_pool_list = []
    for _ in range(gene_pool_size):
        gene_list = []
        gene_list.append([rd.randint(0, input_neuron - 1), rd.randint(0, output_neuron - 1)])
        connecting_num = rd.randint(1, input_neuron * output_neuron)
        for _ in range(connecting_num):
            rand = [rd.randint(0, input_neuron - 1), rd.randint(0, output_neuron - 1)]
            gene_list.append(rand)
        gene_tuple = set([tuple(gene_list) for gene_list in gene_list])
        gene_list = []
        for gene in gene_tuple:
            gene_list.append([gene[0], gene[1]])

        gene_pool_list.append(gene_list)

    return gene_pool_list


def mutate(gene_pool, adding_node_probability, adding_connection_probability):
    gene_pool_list = []
    mutated_gene_pool_list = []
    for gene in gene_pool:
        gene_node_list = []
        for nuc in gene:
            gene_node_list.append(nuc[0])
            gene_node_list.append(nuc[1])
        gene_node_list = list(set(gene_node_list))
        gene_node_list.sort()
        gene_connection_list = gene.copy()
        gene_pool_list.append([gene_node_list, gene_connection_list])
    for gene_info in gene_pool_list:
        node_mutated_gene = []
        current_num = gene_info[0][-1]
        for i in gene_info[1]:
            if rd.random() < adding_node_probability:
                node_mutated_gene.append([i[0], current_num + 1])
                node_mutated_gene.append([current_num + 1, i[1]])
            else:
                node_mutated_gene.append(i)
        fully_mutated_gene = node_mutated_gene.copy()
        mnuc_node_list = []
        for mnuc in node_mutated_gene:
            mnuc_node_list.append(mnuc[0])
            mnuc_node_list.append(mnuc[1])
        mnuc_node_list = list(set(mnuc_node_list))
        if rd.random() < adding_connection_probability:
            try:
                while True:
                    rand_connection = rd.sample(mnuc_node_list, 2)
                    fully_mutated_gene.index(rand_connection)
            except ValueError:
                fully_mutated_gene.append(rand_connection)
        else:
            pass
        mutated_gene_pool_list.append(fully_mutated_gene)
    return mutated_gene_pool_list


def crossover(gene_pool, prob_list):

    check_list = list(range(len(gene_pool)))
    checked = np.random.choice(check_list, size=len(gene_pool), p=prob_list)

    adapted_genes = []
    for check in checked:
        adapted_genes.append(gene_pool[check])
    rd.shuffle(adapted_genes)
    choiced_list = []
    crossovered_gene_pool_list = []
    for i in range(int(len(gene_pool) / 2)):
        choiced_list.append(adapted_genes[2 * i:2 * i + 2])
    gene1 = []
    gene2 = []
    for choiced in choiced_list:
        t = rd.randint(0, 1)
        if t == 0:
            gene1.append(choiced[0])
            gene2.append(choiced[1])
        else:
            gene1.append(choiced[1])
            gene2.append(choiced[0])
    crossovered_gene_pool_list.append(gene1)
    crossovered_gene_pool_list.append(gene2)
    return crossovered_gene_pool_list




