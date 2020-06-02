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


def mutate(gene_pool, adding_node_probability, adding_connection_probability,
           initial_active_connection_prob,inp,outp):
  
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
        current_num = max(gene_info[0])
        for i in gene_info[1]:
            if rd.random() < adding_node_probability:
                node_mutated_gene.append([i[0], current_num+1])
                node_mutated_gene.append([current_num+1, i[1]])
            else:
                node_mutated_gene.append(i)
        fully_mutated_gene = node_mutated_gene.copy()
        mnuc_node_list = []
        for mnuc in node_mutated_gene:
            mnuc_node_list.append(mnuc[0])
            mnuc_node_list.append(mnuc[1])
        num_of_node = len(fully_mutated_gene)
        mnuc_node_list = list(set(mnuc_node_list))
        mnuc_node_list.sort()
        if rd.random() < adding_connection_probability:
            rand_connection = [rd.randint(0,mnuc_node_list[-1]),rd.randint(0,mnuc_node_list[-1])]
            fully_mutated_gene.append(rand_connection)
            fully_mutated_gene_tup = set([tuple(fully_mutated_gene) for fully_mutated_gene in fully_mutated_gene])
            fmg_list = []
            for gene in fully_mutated_gene_tup:
                fmg_list.append([gene[0], gene[1]])
            fully_mutated_gene = fmg_list

        else:
            pass
        if rd.random() < initial_active_connection_prob:
            if len(fully_mutated_gene) == 1:
                pass
            else:
                rand_num = rd.randint(0,len(fully_mutated_gene)-1)
                if (fully_mutated_gene[rand_num][0]>=inp) and (fully_mutated_gene[rand_num][1]>=outp):
                    del fully_mutated_gene[rand_num]
                else:
                    pass
        else:
            pass
        #for b,n in enumerate(fully_mutated_gene):
            #if not n:
                #del fully_mutated_gene[b]
        mutated_gene_pool_list.append(fully_mutated_gene)
    return mutated_gene_pool_list


def crossover(gene_pool, score_list,tournament_size):
    selected_genenum_list = []
    sco_list = [score_list[i:i+tournament_size] for i in range(0, len(score_list), tournament_size)]
    for i,prob in enumerate(sco_list):
        selected_genenum_list.append(i*tournament_size+prob.index(max(prob)))
    adapted_genes = []
    for selected in selected_genenum_list:
        adapted_genes.append(gene_pool[selected])
    adapted_genes = adapted_genes * int(len(gene_pool)/tournament_size)
    rd.shuffle(adapted_genes)
    choiced_list = []
    crossovered_gene_pool_list = []
    for i in range(int(len(gene_pool) / 2)):
        choiced_list.append(adapted_genes[2 * i:2 * i + 2])

    for choiced in choiced_list:
        gene1 = []
        gene2 = []
        if len(choiced[0]) > len(choiced[1]):
            for i in range(len(choiced[1])):
                t = rd.randint(0, 1)
                if t == 0:
                    gene1.append(choiced[0][i])
                    gene2.append(choiced[1][i])
                else:
                    gene1.append(choiced[1][i])
                    gene2.append(choiced[0][i])
            for h in choiced[0][len(choiced[0]):len(choiced[1])]:
                gene1.append(h)
        elif len(choiced[0]) < len(choiced[1]):
            for j in range(len(choiced[0])):
                t = rd.randint(0, 1)
                if t == 0:
                    gene1.append(choiced[0][j])
                    gene2.append(choiced[1][j])
                else:
                    gene1.append(choiced[1][j])
                    gene2.append(choiced[0][j])
            for r in choiced[1][len(choiced[0]):len(choiced[1])]:
                gene2.append(r)
        else:
            for k in range(len(choiced[0])):
                t = rd.randint(0,1)
                if t == 0:
                    gene1.append(choiced[0][k])
                    gene2.append(choiced[1][k])
                else:
                    gene1.append(choiced[1][k])
                    gene2.append(choiced[0][k])
        crossovered_gene_pool_list.append(gene1)
        crossovered_gene_pool_list.append(gene2)
    return crossovered_gene_pool_list



