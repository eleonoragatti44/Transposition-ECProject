from random import random, randint, sample, uniform, randrange
from operator import itemgetter
import numpy as np


'''
REPRESENTATION
'''

# return the lenght of the cromosome of a problem with a precise domain and precision
def cromo_len(max_domain, prec):
    approx_max_domain = int(max_domain * 10**(prec))
    max_domain_bin = bin(approx_max_domain).lstrip("0b") 
    len_max = len(max_domain_bin)
    return len_max
    
    
# Convert a float number into a binary string. 
# It multiplies the the float for 10^(precision), cuts the decimal part, and then applies the binary conversion.
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

# convert a binary list to float.
def bin_to_float(numb, prec):
    if numb[0] == 0:
        sign = 1
    else:
        sign = -1
    numb = numb[1:]
    numb = ''.join(str(i) for i in numb)
    approx_numb = int(numb,2)/(10**prec)
    return(approx_numb * sign)

# return a list of float correspoding to the coordinate of the binary input element (geno).
def phenotype(geno, dimension, precision):
    pheno = []
    len_mono_cromo = int(len(geno)/dimension)
    for i in range(0, len(geno), len_mono_cromo):
        pheno.append(bin_to_float(geno[i:i+len_mono_cromo], precision))
    return pheno

# return a binary list of the float coordinate input
def genotype(feno, max_domain,precision):
    geno_str = ''
    for i in feno:
        geno_str += float_to_bin(i,max_domain, precision)
    geno = [int(geno_str[i]) for i in range(len(geno_str))]
    return geno
    
    
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
def muta_bin(max_domain,dimension,prec):
    def mutation(indiv, prob_muta):
        # Mutation by gene
        old = indiv[:]
        cromo = indiv[:]
        len_cromo = int(len(indiv)) # cromosome length
        len_mono_cromo = int(len_cromo/dimension) # length of the single coordinate in bit
        
        # jump from one mono_cromo to the next one
        # and applied the mutation to the last len_cromo/3 bits
        for j in range(0, len_cromo, len_mono_cromo):
            for i in range(1, round(len_mono_cromo/2)):
                cromo[len_mono_cromo+j-i] = muta_bin_gene(cromo[len_mono_cromo+j-i],prob_muta)
                pheno = phenotype(cromo,dimension,prec)
                # We control that the mutated cromosome does not correspond to a float value out from the domain
                for x in pheno:
                    if x > max_domain: 
                        cromo[i] = old[i]
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
            for i, val in enumerate(pheno1):
                if(val > max_domain):
                    pheno1[i] = max_domain
                if(val < - max_domain):
                    pheno1[i] = - max_domain
            for i, val in enumerate(pheno2):
                if(val > max_domain):
                    pheno2[i] = max_domain
                if(val < - max_domain):
                    pheno2[i] = - max_domain
            f1 = genotype(pheno1, max_domain,precision)
            f2 = genotype(pheno2, max_domain,precision)
            return ((f1,0),(f2,0))
        else:
            return (indiv_1,indiv_2)
    return crossover



# Transposition
def transposition(flank_size):
    def transpose(indiv_1, indiv_2):
        end = False
        start = False
        f1 = indiv_1
        f2 = indiv_2
        j = 0
        
        # define a random point in indiv_1
        size_indiv = len(indiv_1)
        rnd_index = randrange(flank_size, size_indiv)
        flanking = indiv_1[rnd_index-flank_size:rnd_index]
        print('rnd_index', rnd_index)
        print('flanking', flanking)
        
        while end == False:
            # if flanking==slice of indiv_1
            if (indiv_1[ (rnd_index+j)%size_indiv : (rnd_index+j+flank_size)%size_indiv ] == flanking):
                end = True
                end_indiv_1 = (rnd_index+j+flank_size)%size_indiv
                print('end indiv_1: ', end_indiv_1)
            else:
                j += 1
                
        j = 0
        while start == False:
            if (indiv_2[j : j+flank_size] == flanking):
                start = True
                start_indiv_2 = j+flank_size
                print('start indiv_2: ', start_indiv_2)
            else:
                j += 1
            
        if end_indiv_1 == rnd_index or start == False:
            print("trasposizione non avvenuta")
            return indiv_1, indiv_2
                
        # now the transposition happens
        i = 0
        
        #print('indiv1:\t', indiv_1)
        #print('indiv2:\t', indiv_2)
        #print('f1:\t', f1)
        #print('f2:\t', f2)
        
        # repeat len(transposone) times
        while (rnd_index+i)%size_indiv != end_indiv_1:
            f1[(rnd_index+i)%size_indiv] = indiv_2[(start_indiv_2+i)%size_indiv]
            f2[(start_indiv_2+i)%size_indiv] = indiv_1[(rnd_index+i)%size_indiv]
            # è come se le due righe qui sopra non stessero assegnando, come se f1 e f2 non venissero mai modificati
            print('f1:\t', f1)
            print('f2:\t', f2)
            i += 1
            print("trasposizione in corso")
        return f1, f2
    return transpose
        
        
        
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
    for j in range(numb_generations):
        print(j, end='\r')

        # sparents selection
        mate_pool = sel_parents(populacao)
    # Variation
        # ------ Crossover
        progenitores = []
        for i in range(0,size_pop-1,2):
            indiv_1= mate_pool[i]
            indiv_2 = mate_pool[i+1]
            filhos = recombination(indiv_1,indiv_2, prob_cross)
            progenitores.extend(filhos) 
        # ------ Mutation
        descendentes = []
        for cromo,fit in progenitores:
            novo_indiv = mutation(cromo,prob_mut)
            descendentes.append((novo_indiv,fitness_func(novo_indiv)))
        # New population
        populacao = sel_survivors(populacao,descendentes)
        # Evaluate the new population
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]     
    return best_pop(populacao)

'''
Best over all algorithm
'''

# return the average of the population
def average_pop(populacao):
    return sum([fit for cromo,fit in populacao])/len(populacao)

# Simple [Binary] Evolutionary Algorithm 
# Return the best plus, best by generation, average population by generation
def sea_for_plot(numb_generations,size_pop, size_cromo, prob_mut,prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func,gera_pop):
    # inicializa população: indiv = (cromo,fit)
    populacao = gera_pop(size_pop,size_cromo)
    # avalia população
    populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
    
    # para a estatística
    stat = [best_pop(populacao)[1]]
    stat_aver = [average_pop(populacao)]
    
    for j in range(numb_generations):
        print(j, end='\r')

        # selecciona progenitores
        mate_pool = sel_parents(populacao)
	# Variation
	# ------ Crossover
        progenitores = []
        for i in  range(0,size_pop-1,2):
            cromo_1= mate_pool[i]
            cromo_2 = mate_pool[i+1]
            filhos = recombination(cromo_1,cromo_2, prob_cross)
            progenitores.extend(filhos) 
        # ------ Mutation
        descendentes = []
        for indiv,fit in progenitores:
            novo_indiv = mutation(indiv,prob_mut)
            descendentes.append((novo_indiv,fitness_func(novo_indiv)))
        # New population
        populacao = sel_survivors(populacao,descendentes)
        # Avalia nova _população
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao] 
	
	# Estatística
        stat.append(best_pop(populacao)[1])
        stat_aver.append(average_pop(populacao))
	
    return best_pop(populacao),stat, stat_aver


# return the best over all and the best average over all
def sea_boa(numb_repetitions,numb_generations,size_pop, size_cromo, prob_mut,prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func):
    results_stat = [] # stat(s) of every rep.The index specifies the repetition of the sea alg.
    results_stat_av = [] #sat_av(s) of every rep. The index specifies the repetition of the sea alg.
    boa = []  # best over all
    baoa = [] # best average over all
    for i in range(numb_repetitions):
        best, stat, stat_av = sea_for_plot(numb_generations,size_pop, size_cromo, prob_mut,prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func)
        results_stat.append(stat)
        results_stat_av.append(stat_av)
    for g in range(numb_generations):
        boa.append(results_stat[0][g])
        baoa.append(results_stat_av[0][g])
        
        for r in range(numb_repetitions):
            if (boa[g] < results_stat[r][g]):
                boa[g] = results_stat[r][g]
            if (baoa[g] < results_stat_av[r][g]):
                baoa[g] = results_stat_av[r][g]
                
    return(boa, baoa)
