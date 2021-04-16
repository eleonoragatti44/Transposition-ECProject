from random import random, randint, sample, uniform
from operator import itemgetter
import numpy as np

'''
REPRESENTATION
'''
def cromo_len(max_domain, prec):
    approx_max_domain = int(max_domain * 10**(prec))
    max_domain_bin = bin(approx_max_domain).lstrip("0b") 
    len_max = len(max_domain_bin)
    return len
    
def float_to_bin(number, max_domain, prec):
    # Defining the lenght of the cromosome
    approx_max_domain = int(max_domain * 10**(prec))
    max_domain_bin = bin(approx_max_domain).lstrip("0b") 
    len_max = len(max_domain_bin)

    sign = 0
    # Use prec to define max lenght for decimals
    approx_number = int(number * 10**(prec))
    if number < 0:
        sign += 1
        approx_number *= -1
    number_bin_str = bin(approx_number).lstrip("0b") 
    
    # We add the zeros at the beginning of the bin represented number
    len_numb = len(number_bin_str)
    diff_len = len_max - len_numb
    if diff_len > 0:
        for _ in range(diff_len):
            number_bin_str = '0' + number_bin_str
    if sign == 0:
        number_bin_str = '0' + number_bin_str
    else:
        number_bin_str = '1' + number_bin_str
        
    return(number_bin_str)

def bin_to_float(numb, prec):
    if numb[0] == 0:
        sign = 1
    else:
        sign = -1
    numb = numb[1:]
    numb = ''.join(str(i) for i in numb)
    approx_numb = int(numb,2)/(10**prec)
    return(approx_numb * sign)

def phenotype(geno, dimension, precision):
    pheno = []
    len_mono_cromo = int(len(geno)/dimension)
    for i in range(0, len(geno), len_mono_cromo):
        pheno.append(bin_to_float(geno[i:i+len_mono_cromo], precision))
    return pheno

'''
STEP 0 - FITNESS

def fitness(geno, f):
    return f(phenotype(indiv))
'''


'''
STEP 1 - INITIALIZE POPULATION
'''

# Initialize population
def gera_pop(max_domain, precision, dimension):
    def pop(size_pop, size_cromo):
        return [(gera_indiv(max_domain, precision, dimension),0) for _ in range(size_pop)]
    return pop

def gera_indiv(max_domain, precision, dimension):
    indiv = ''
    for _ in range(dimension):
        # random initialization
        x = np.random.uniform(-max_domain, max_domain)
        # convert to bin
        indiv += float_to_bin(x, max_domain, precision)
    indiv = [int(indiv[i]) for i in range (len(indiv))]
    return indiv


'''
STEP 2 - PARENTS SELECTION
'''

# Parents Selection: tournament
def tour_sel(t_size):
    def tournament(pop):
        size_pop= len(pop)
        mate_pool = []
        for i in range(size_pop):
            winner = one_tour(pop,t_size)
            mate_pool.append(winner)
        return mate_pool
    return tournament

def one_tour(population,size):
    """Maximization Problem. Deterministic"""
    pool = sample(population, size)
    pool.sort(key=itemgetter(1))
    return pool[0]

'''
STEP 3 - VARIATION OPERATOR
'''

# Binary mutation
def muta_bin(prob_muta,max_domain,dimension,prec):
    def mutation(indiv, prob_muta):
        # Mutation by gene
        cromo = indiv[:]
        temp = indiv[:]
        for i in range(len(indiv)):
            temp[i] = muta_bin_gene(temp[i],prob_muta)
            pheno = phenotype(temp,dimension,prec)
            # We control that the mutated cromosome doesn not correspond to a float value out from the domain
            for x in pheno:
                if x < max_domain: 
                    cromo[i] = temp[i]
        return cromo
    return mutation


def muta_bin_gene(gene, prob_muta):
    g = gene
    value = random()
    if value < prob_muta:
        g ^= 1
    return g


# New version with different boundary process
# Uniform crossover 
def uniform_cross(prob_cross, max_domain, precision, dimension):
    def crossover(indiv_1, indiv_2, prob_cross):
        max_domain_bin = float_to_bin(max_domain, max_domain, precision) # binary string representing the domain boundary
        max_domain_bin =[int(max_domain_bin[i]) for i in range (len(max_domain_bin))] # binary list
        min_domain_bin = max_domain_bin
        min_domain_bin[0] = 1
        value = random()
        if value < prob_cross:
            cromo_1 = indiv_1[0]
            cromo_2 = indiv_2[0]
            f1 = []
            f2 = []
            for i in range(0,len(cromo_1)):
                if random() < 0.5:
                    f1.append(cromo_1[i])
                    f2.append(cromo_2[i])
                else:
                    f1.append(cromo_2[i])
                    f2.append(cromo_1[i])
            pheno1 = phenotype(f1, dimension, precision)  
            pheno2 = phenotype(f2, dimension, precision)
            for i in pheno1:
                if(i > max_domain):
                    pheno1[i] = max_domain_bin
                if(i < - max_domain):
                    pheno1[i] = min_domain_bin
            for i in pheno2:
                if(i > max_domain):
                    pheno2[i] = max_domain_bin
                if(i < - max_domain):
                    pheno2[i] = min_domain_bin
            
            return ((f1,0),(f2,0))
        else:
            return (indiv_1,indiv_2)
    return crossover

""" 
# old version with rejecting of the "outsider" (values created out from the domain).
# Uniform crossover 
def uniform_cross(prob_cross, max_domain, precision, dimension):
    def crossover(indiv_1, indiv_2, prob_cross):
        value = random()
        if value < prob_cross:
            cromo_1 = indiv_1[0]
            cromo_2 = indiv_2[0]
            while True:
                f1 = []
                f2 = []
                for i in range(0,len(cromo_1)):
                    if random() < 0.5:
                        f1.append(cromo_1[i])
                        f2.append(cromo_2[i])
                    else:
                        f1.append(cromo_2[i])
                        f2.append(cromo_1[i])
                pheno1 = phenotype(f1, dimension, precision)  
                pheno2 = phenotype(f2, dimension, precision)
                #print(pheno1,'ph1\n')
                #print(pheno2,'ph2\n')
                pheno = pheno1+pheno2
                pheno.sort()
                if(pheno[-1] <= max_domain):
                    return ((f1,0),(f2,0))
        else:
            return (indiv_1,indiv_2)
    return crossover
"""

# Transposition
'''
def transposition(flanking):

'''

'''
STEP 4 - SURVIVALS SELECTION
'''

# Survivals Selection: elitism
def sel_survivors_elite(elite):
    def elitism(parents,offspring):
        size = len(parents)
        comp_elite = int(size* elite)
        offspring.sort(key=itemgetter(1))
        parents.sort(key=itemgetter(1))
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        return new_population
    return elitism

def best_pop(populacao):
    populacao.sort(key=itemgetter(1))
    return populacao[0]


'''
EVOLUTIONARY ALGORITHM
'''

# Binary Evolutionary Algorithm
def ea(numb_generations, size_pop, prob_mut, prob_cross,
       sel_parents, recombination, mutation, sel_survivors,
       fitness_func, dimension, max_domain, precision):
    
    # inicialize population: indiv = (cromo,fit)
    populacao = gera_pop(size_pop, max_domain, precision, dimension)
    # evaluate population
    populacao = [(indiv[0], fitness_func(indiv[0], dimension, precision)) for indiv in populacao]
    for i in range(numb_generations):
        # sparents selection
        mate_pool = sel_parents(populacao)
    # Variation
    # ------ Crossover
        progenitores = []
        for i in  range(0,size_pop-1,2):
            indiv_1= mate_pool[i]
            indiv_2 = mate_pool[i+1]
            filhos = recombination(indiv_1,indiv_2, prob_cross,max_domain,precision,dimension)
            progenitores.extend(filhos) 
        # ------ Mutation
        descendentes = []
        for cromo,fit in progenitores:
            novo_indiv = mutation(cromo,prob_mut,max_domain,dimension,precision)
            descendentes.append((novo_indiv,fitness_func(novo_indiv, dimension, precision)))
        # New population
        populacao = sel_survivors(populacao,descendentes)
        # Evaluate the new population
        populacao = [(indiv[0], fitness_func(indiv[0], dimension, precision)) for indiv in populacao]     
    return best_pop(populacao)


'''
!!!!!INTOCCABILE!!!!!
'''

def sea(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination, mutation, sel_survivors, fitness_func, gera_pop):
    # inicialize population: indiv = (cromo,fit)
    populacao = gera_pop(size_pop,size_cromo)
    # evaluate population
    populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
    for i in range(numb_generations):
        print(i, end='\r')
        # sparents selection
        mate_pool = sel_parents(populacao)
    # Variation
        # ------ Crossover
        progenitores = []
        for i in  range(0,size_pop-1,2):
            indiv_1= mate_pool[i]
            indiv_2 = mate_pool[i+1]
            filhos = recombination(indiv_1,indiv_2, prob_cross)
            progenitores.extend(filhos) 
        # ------ Mutation
        descendentes = []
        for cromo,fit in progenitores:
            print(cromo)
            novo_indiv = mutation(cromo,prob_mut)
            descendentes.append((novo_indiv,fitness_func(novo_indiv)))
        # New population
        populacao = sel_survivors(populacao,descendentes)
        # Evaluate the new population
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]     
    return best_pop(populacao)