#import pandas_datareader.data as web
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as sci
import random
import seaborn as sns
import scipy.optimize as sco
import time
import scipy.stats as scis
#from skmultiflow.drift_detection import PageHinkley
from scipy.stats import norm, zscore
#from operators import *


#set_stock_prices_1=['T','V','MCD','INTC','AAPL','AMZN','KO','PFE','GOOG','AFL','EBAY','MO','JNJ','WFC','PEP','KHC','ABBV','AMD','BAC','GS']
#set_stock_prices_1=['KO','GOOG']
#data=web.DataReader(set_stock_prices_1,data_source="yahoo",start='08/01/2015',end='08/01/2020')['Adj Close']
#data=pd.read_csv('C:\\Users\\nonloso\\Desktop\\codici_ga\\FTSE_MIB1.CSV',header=None)

# 1. Read, parse dates and set index
data = pd.read_csv(
    'FTSE-100.csv',
    index_col=0,
    parse_dates=[0],
)

data = data.apply(pd.to_numeric, errors='coerce')
data.dropna(inplace=True)


#dataframe ripulito
#data=data.resample('M').apply(lambda x: x[-1])
data.sort_index(inplace=True)
n_stocks=len(data.columns)
rendimenti_set_1=data.pct_change().dropna()
rendimento_medio_set_1=rendimenti_set_1.mean() 
matrice_covarianza_1=rendimenti_set_1.cov()
matrice_correlazione_1=rendimenti_set_1.corr()


def rendimento(pop):
    rendimento_portafoglio=np.dot(rendimento_medio_set_1,pop.T)
    pesi=pop
    rendimento_portafoglio=np.zeros(len(pesi))
    for i in range(len(pesi)):
        rendimento_portafoglio[i] = np.sum(rendimento_medio_set_1 * pesi[i])
    return rendimento_portafoglio


def vol(pop):
    matrice_covarianza_1=rendimenti_set_1.cov()
    #matrice_correlazione_1=rendimenti_set_1.corr()
    pesi=pop
    std_dev_portafoglio=np.zeros(len(pesi)) 
    for i in range(len(pop)):
        std_dev_portafoglio[i]=np.sqrt(np.dot(pesi[i].T,np.dot(matrice_covarianza_1,pesi[i])))
    return std_dev_portafoglio

def risk_parity(pop):
    pesi=pop
    fRP=np.zeros(np.shape(pesi))
    portvar=vol(pesi)**2
    Cx=(np.dot(matrice_covarianza_1,pesi.T))
    for j in range(len(pesi.T)):
            #fRP[:,j]=(((pesi[:,j]*Cx[j,:])/portvar)-(1/geni))**2
            #fRP[:,j]=(((pesi[:,j]*Cx[j,:])/portvar)-(1/geni))**2
            fRP[:,j]=np.abs(((pesi[:,j]*Cx[j,:])/portvar)-(1/geni))
    rp=-np.sum(fRP,axis=1)
    return rp


    
def semivar(pop):
    pesi=pop
    rendimento_portafoglio_mensile=rendimenti_set_1@pesi.T
    semivar=(np.var(np.minimum(rendimento_portafoglio_mensile.mean()-rendimento_portafoglio_mensile,0),axis=0))
    return semivar

def mean_absolute_deviation(pop):
    pesi=pop
    rendimento_portafoglio_mensile=rendimenti_set_1@pesi.T
    mad=np.mean(np.abs(rendimento_portafoglio_mensile-np.mean(rendimento_portafoglio_mensile)))
    return mad

def omega_ratio(pop):
    pesi=pop
    rendimento_portafoglio_mensile=rendimenti_set_1@pesi.T
    omega=(np.sum(np.minimum(rendimento_portafoglio_mensile,0),axis=0)/np.sum(np.maximum(rendimento_portafoglio_mensile,0),axis=0))
    return omega

def nuova_rm(pop):
    delta=0.5
    perdita=-0.02
    gain=0.05
    pesi=pop
    conta='meno'
    rendimento_portafoglio_mensile=rendimenti_set_1@pesi.T
    if conta=='meno':
        contameno=np.count_nonzero(rendimento_portafoglio_mensile<perdita,axis=0)/len(rendimento_portafoglio_mensile)
        costo=delta*rendimento(pop)-(1-delta)*(contameno)
        #costo=-contameno
    elif conta=='piu':
        contapiu=np.count_nonzero(rendimento_portafoglio_mensile>gain,axis=0)/rendimento_portafoglio_mensile
        costo=rendimento(pop)+delta*(contapiu)
    return costo
    
def twosided(pop):
    a=0.25
    pesi=pop
    rendimento_portafoglio_mensile=rendimenti_set_1@pesi.T
    twoside=np.zeros(len(rendimento_portafoglio_mensile.T))
    #upside=rendimento_portafoglio_mensile
    #downside=rendimento_portafoglio_mensile
    upside=np.maximum(rendimento_portafoglio_mensile-rendimento_portafoglio_mensile.mean(),0)
    downside=np.maximum(rendimento_portafoglio_mensile.mean()-rendimento_portafoglio_mensile,0)
    ##    for i in range(len(rendimento_portafoglio_mensile.T)):
    ##	    for j in range(len(rendimento_portafoglio_mensile)):
    ##		    if rendimento_portafoglio_mensile.iloc[j,i]-rendimento_portafoglio_mensile.iloc[:,i].mean()<0:
    ##			    upside.iloc[j,i]=0
    ##    for i in range(len(rendimento_portafoglio_mensile.T)):
    ##	    for j in range(len(rendimento_portafoglio_mensile)):
    ##		    if rendimento_portafoglio_mensile.iloc[:,i].mean()-rendimento_portafoglio_mensile.iloc[j,i]<0:
    ##			    downside.iloc[j,i]=0
    #upside=np.max((rendimento_portafoglio_mensile-rendimento_portafoglio_mensile.mean()),0)
    #downside=np.max((rendimento_portafoglio_mensile.mean()-rendimento_portafoglio_mensile),0)
    for z in range(len(rendimento_portafoglio_mensile.T)):
        twoside[z]=-a*np.linalg.norm(upside.iloc[:,z],ord=1)-(1-a)*np.linalg.norm(downside.iloc[:,z],ord=2)+rendimento_portafoglio_mensile.mean()[z]
    #twosided=a*np.linalg.norm(upside,ord=1)+(1-a)*np.linalg(downside,ord=2)-rendimento_portafoglio_mensile
    
    return (twoside)

def sortino_ratio(pop):
    pesi=pop
    rendimento_portafoglio_mensile=rendimenti_set_1@pesi.T
    semivar=(np.var(np.minimum(rendimento_portafoglio_mensile.mean()-rendimento_portafoglio_mensile,0),axis=0))
    sortino=np.mean(rendimento_portafoglio_mensile,axis=0)/semivar
    return sortino

def sharpe(pop):
    return rendimento(pop)/vol(pop)

def mean_variance(pop):
    lambda_1=0.5
    delta=2.0
    return -lambda_1*vol(pop)+(1-lambda_1)*rendimento(pop)-delta*np.sum(abs(pop),axis=1)
    
def mean_semivariance(pop):
    lambda_1=0.5
    return -lambda_1*semivar(pop)+(1-lambda_1)*rendimento(pop)

def mean_mad(pop):
    lambda_1=0.5
    return -lambda_1*mean_absolute_deviation(pop)+(1-lambda_1)*rendimento(pop)

def minimax(pop):##minimax
    lambda_1=0.5
    minimax=np.zeros(len(pop))
    for i in range(len(pop)):
	#rendimento_portafoglio[i] = rendimento_medio_set_1 * pesi[i] * 12
	#np.sum(np.abs(rendimenti_set_1-*pesi[i],axis=1))
        minimax[i]=-lambda_1*min((1-lambda_1)*(rendimento_medio_set_1 * pop[i])-(np.mean(np.abs(rendimenti_set_1-rendimento_medio_set_1))*pop[i]))
    return minimax

def variance_with_skewness(pop):
    lambda_1=0.5
    std_dev_portafoglio=np.zeros(len(pop))
    rendimento_portafoglio=rendimento_medio_set_1@pop.T
    rendimento_portafoglio_mensile=rendimenti_set_1@pop.T
    for i in range(len(pop)):
        std_dev_portafoglio[i]=np.sqrt(np.dot(pop[i].T,np.dot(matrice_covarianza_1,pop[i])))
    dsr2=np.array(np.mean((rendimento_portafoglio_mensile-np.mean(rendimento_portafoglio_mensile))**2))
    dsr3=np.array(np.mean((rendimento_portafoglio_mensile-np.mean(rendimento_portafoglio_mensile))**3))
    ccef=-lambda_1*dsr2+(1-lambda_1)*np.mean(rendimento_portafoglio_mensile)+(dsr3/(dsr2)**(3/2))
    return ccef

def value_at_risk(pop):
    alpha=0.05
    rend=rendimento(pop)
    stdev=vol(pop)
    var=norm.ppf(alpha,rend,stdev)*np.sqrt(21)
    #var=(norm.ppf(1-alpha)*stdev-rend)*np.sqrt(21)
    return var

def expected_shortfall(pop):
    alpha=0.05
    rend=rendimento(pop)
    std=vol(pop)
    es=-(alpha**-1*norm.pdf(norm.ppf(alpha))*std - rend)*np.sqrt(21)
    return es



def calcola_entropy(pop):
    aux=np.zeros((cromosomi,geni))
    for i in range(len(pop)):
        for j in range(geni):
            aux[i,j]=-(pop[i,j]/cromosomi)*(np.log(pop[i,j]/cromosomi))/(np.log(2)*geni)
    aux1=np.sum(aux,axis=1)
    aux2=np.sum(aux1,axis=0)
    return aux2


def calcola_pop_fitness(pop):
##    obiettivo=sharpe(pop)
##    obiettivo=sortino_ratio(pop)
    obiettivo=omega_ratio(pop)
##    obiettivo=risk_parity(pop)
##    obiettivo=nuova_rm(pop)
##    obiettivo=twosided(pop)
##    obiettivo=mean_variance(pop)
##    obiettivo=mean_semivariance(pop)
##    obiettivo=mean_mad(pop)
##    obiettivo=variance_with_skewness(pop)
##    obiettivo=minimax(pop)
##    obiettivo=value_at_risk(pop)
##    obiettivo=expected_shortfall(pop)
    return obiettivo



def elitist_selection(pop, fitness, num_parents):  ##elitist selection (tasso=5%)
    ##nel loop si individua la soluzione con fitness massima
    ##il primo parent viene selezionata dalla pop con fit max
    ##la fitness viene posta molto negativa per non essere riselezionata
    parents=np.zeros((num_parents, pop.shape[1])) #inizializz dim
    for i in range(num_parents):
        posizione_max_fitness = np.where(fitness == np.max(fitness)) ##posizione idx con fitness massima
        posizione_max_fitness = posizione_max_fitness[0][0]
        parents[i,:] = pop[posizione_max_fitness, :]
        fitness[posizione_max_fitness] = -10000
    return parents

def selection(pop, fitness, num_parents):  ##elitist selection (tasso=5%)
    ##nel loop si individua la soluzione con fitness massima
    ##il primo parent viene selezionata dalla pop con fit max
    ##la fitness viene posta molto negativa per non essere riselezionata
    parents=np.zeros((num_parents, pop.shape[1])) #inizializz dim
    #for i in range(num_parents):
        #posizione_max_fitness = np.where(fitness == np.max(fitness)) ##posizione idx con fitness massima
        #posizione_max_fitness = posizione_max_fitness[0][0]
        #parents[i,:] = pop[posizione_max_fitness, :]
    parents = pop
        #fitness[posizione_max_fitness] = -10000
    return parents

#tournament selection
def tournament_selection(pop,fitness, num_parents):
    parents=np.zeros((num_parents, pop.shape[1])) #inizializz dim
    for i in range(num_parents):
        tournament=np.array(random.choices(range(len(fitness)),k=2))
        fitness_tournament=fitness[tournament]
        stack=np.vstack((tournament,fitness_tournament))
        if stack[1,0]>stack[1,1]:
            posizione=stack[0,0]
        else:
            posizione=stack[0,1]
        parents[i,:]=pop[int(posizione),:]
    return parents



def crossover(parents, offspring_size): #Goldberg (1975)
    offspring=np.zeros(offspring_size)
    crossover_point=int(offspring_size[1]/2) ##len colonne/2 (k_point crossover con k=1, crossover deterministico a meta' cromosoma)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0] ##resto genitore 1 (0, poi 1,2,3,...)
        parent2_pos = (i+1)%parents.shape[0] ##resto genitore 2 (all'ultimo step riparte da zero) 1-2-3-...-0
        offspring[i, :crossover_point] = parents[parent1_pos, :crossover_point] ##meta' dei geni dal primo genitore
        offspring[i, crossover_point:] = parents[parent2_pos, crossover_point:] ##meta' dei geni dal secondo genitore
        ##completato il primo loop, viene generato il primo figlio
    return offspring


def two_point_crossover(parents,offspring_size): #Goldberg (1975), Muehlenberger (1993)
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]):
        parent1_pos=i%parents.shape[0]
        parent2_pos=(i+1)%parents.shape[0]
        crossover_point_1=np.random.randint(1,geni-1)
        crossover_point_2=np.random.randint(crossover_point_1,geni-1)
        if i%2==0:
            offspring[i,:crossover_point_1]=parents[parent1_pos,:crossover_point_1]
            offspring[i,crossover_point_1:crossover_point_2]=parents[parent2_pos,crossover_point_1:crossover_point_2]
            offspring[i,crossover_point_2:]=parents[parent1_pos,crossover_point_2:] 
        else:
            offspring[i,:crossover_point_1]=parents[parent2_pos,:crossover_point_1]
            offspring[i,crossover_point_1:crossover_point_2]=parents[parent1_pos,crossover_point_1:crossover_point_2]
            offspring[i,crossover_point_2:]=parents[parent2_pos,crossover_point_2:] 
    return offspring

def three_point_crossover(parents,offspring_size): #Muehlenberger (1993)
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]):
        parent1_pos=i%parents.shape[0]
        parent2_pos=(i+1)%parents.shape[0]
        crossover_point_1=int((offspring_size[1]-1)/3)
        crossover_point_2=int((2*offspring_size[1]-1)/3)
        crossover_point_3=int((3*offspring_size[1]-1)/3)
        #crossover_point_1=np.random.randint(1,geni-1)
        #crossover_point_2=np.random.randint(crossover_point_1,geni-1)
        #crossover_point_3=np.random.randint(crossover_point_2,geni-1)
        if i%2==0:
            offspring[i,:crossover_point_1]=parents[parent1_pos,:crossover_point_1]
            offspring[i,crossover_point_1:crossover_point_2]=parents[parent2_pos,crossover_point_1:crossover_point_2]
            offspring[i,crossover_point_2:crossover_point_3]=parents[parent1_pos,crossover_point_2:crossover_point_3]
            offspring[i,crossover_point_3:]=parents[parent2_pos,crossover_point_3:] 
        else:
            offspring[i,:crossover_point_1]=parents[parent2_pos,:crossover_point_1]
            offspring[i,crossover_point_1:crossover_point_2]=parents[parent1_pos,crossover_point_1:crossover_point_2]
            offspring[i,crossover_point_2:crossover_point_3]=parents[parent2_pos,crossover_point_2:crossover_point_3]
            offspring[i,crossover_point_3:]=parents[parent1_pos,crossover_point_3:] 
    return offspring

def crossover_uniforme(parents, offspring_size): #Spears, De Jong (1991)
    offspring=np.zeros(offspring_size)
    ###al 50% (p) il peso viene ereditato da un genitore, al 50% dall'altro (1-p), si completa un loop e si passa al figlio successivo
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0] ##resto genitore 1 (0, poi 1,2,3,...)
        parent2_pos = (i+1)%parents.shape[0] ##resto genitore 2 (all'ultimo step riparte da zero) 1-2-3-...-0
        for j in range(geni):
            r=np.random.uniform(0,1)
            if r>0.5:
                offspring[i,j] = parents[parent1_pos,j] 
            else:
                offspring[i,j] = parents[parent2_pos,j]
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def arnone_crossover(parents, offspring_size): #Arnone Loraschi Tettamanzi (1993)
    offspring=np.zeros(offspring_size)
    ###al 50% (p) il peso viene ereditato da un genitore, al 50% dall'altro (1-p), si completa un loop e si passa al figlio successivo
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0] ##resto genitore 1 (0, poi 1,2,3,...)
        parent2_pos = (i+1)%parents.shape[0] ##resto genitore 2 (all'ultimo step riparte da zero) 1-2-3-...-0
        for j in range(geni):
            r=np.random.uniform(0,1)
            if r>0.5:
                offspring[i,j] =min(parents[parent1_pos,j],parents[parent1_pos,j]*((np.sum(parents[parent1_pos])+np.sum(parents[parent2_pos]))/(2*np.sum(parents[parent1_pos]))))
            else:
                offspring[i,j] =min(parents[parent2_pos,j],parents[parent2_pos,j]*((np.sum(parents[parent1_pos])+np.sum(parents[parent2_pos]))/(2*np.sum(parents[parent2_pos]))))
        ##completato il primo loop, viene generato il primo figlio
    return offspring


def crossover_uniforme_globale(parents, offspring_size): #Dan Simon (2013)
    offspring=np.zeros(offspring_size)
    #la probabilità di scelta non è più al 50% tra un genitore e l'altro, ma 1/N:
    #si sceglie alla i-esima posizione uno dei N-esimi geni di tutta la popolazione (di genitori)
    for i in range(offspring_size[0]): ##loop per riga
        for j in range(geni):
            offspring[i,j]=random.choice(parents[:,j])
    return offspring

def flat_crossover(parents, offspring_size): #Herrera (1998)
    ##i geni del figlio derivano da j=geni estrazioni random (unif.) comprese tra i valori più piccoli
    ##dei genitori ed i valori massimi dei medesimi, riferite allo stesso gene j-esimo dei genitori
    #(es per j=0, si prendono i geni più piccoli e grandi dei genitori, a seconda di dove si collocano)
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
                offspring[i,j] =np.random.uniform(min(parents[parent1_pos,j],parents[parent2_pos,j]),max(parents[parent1_pos,j],parents[parent2_pos,j]))
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def blend_crossover(parents, offspring_size): #Houst (1995) & Herrera (1998)
    #E' una modifica del flat crossover, con alpha=0 e' equivalente. Il parametro user-defined alpha rappresenta
    #un mix tra exploration ed exploitation. Herrera (1998) propone alpha=0.5
    alpha=0.5
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
                xmin=min(parents[parent1_pos,j],parents[parent2_pos,j])
                xmax=max(parents[parent1_pos,j],parents[parent2_pos,j])
                deltax=xmax-xmin
                offspring[i,j]=np.abs(np.random.uniform(xmin-(alpha*deltax),xmax+(alpha*deltax)))
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def average_crossover(parents, offspring_size): #Nomura (1997)
    #si prende -per ogni gene del figlio- la media dei j-esimi geni dei rispettivi genitori
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
                offspring[i,j] =(parents[parent1_pos,j]+parents[parent2_pos,j])/2
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def multi_parent_average_crossover(parents, offspring_size): #Nomura (1997)
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0]
        parent3_pos = (i+2)%parents.shape[0] 
        for j in range(geni):
                offspring[i,j] =(parents[parent1_pos,j]+parents[parent2_pos,j]+parents[parent3_pos,j])/3
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def gene_pool_crossover(parents,offspring_size): #Muhlenbein (1993)
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0]
        parent3_pos = (i+2)%parents.shape[0]
        gene_pool=np.hstack((parents[parent1_pos],parents[parent2_pos],parents[parent3_pos]))
        for j in range(geni):
           estrazione=np.random.choice(gene_pool) 
           offspring[i,j]=estrazione
           gene_pool=np.delete(gene_pool,np.where(gene_pool==estrazione)[0][0])
    return offspring

def heuristic_crossover(parents, offspring_size): #Wright (1990)
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
            w=np.random.uniform(0,1)
            if parents[parent1_pos,j]>parents[parent2_pos,j]:
                offspring[i,j] =parents[parent2_pos,j]+w*(parents[parent1_pos,j]-parents[parent2_pos,j])
            else:
                offspring[i,j] =parents[parent1_pos,j]+w*(parents[parent2_pos,j]-parents[parent1_pos,j])
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def arithmetic_crossover(parents, offspring_size): #Michalewicz (1996)
    #media pesata dei geni secondo parametri user-defined
    beta=0.7
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
            if j%2==0:
                offspring[i,j]=beta*parents[parent1_pos,j]+(1-beta)*parents[parent2_pos,j]
            else:
                offspring[i,j]=beta*parents[parent2_pos,j]+(1-beta)*parents[parent1_pos,j]
         ##completato il primo loop, viene generato il primo figlio
    return offspring

def linear_crossover(parents, offspring_size): #Wright (1990)
    #selezione di un offspring all'interno di una matrice 3xgeni secondo fitness
    #i tre offspring sono generati secondo tre diff. relazioni lineari
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos=i%parents.shape[0]
        parent2_pos=(i+1)%parents.shape[0]
        offspring1=np.zeros(geni)
        offspring2=np.zeros(geni)
        offspring3=np.zeros(geni)
        for j in range(geni):
            offspring1[j]=np.abs(0.5*parents[parent1_pos,j]+0.5*parents[parent2_pos,j])
            offspring2[j]=np.abs(1.5*parents[parent1_pos,j]-0.5*parents[parent2_pos,j])
            offspring3[j]=np.abs(-0.5*parents[parent1_pos,j]+1.5*parents[parent2_pos,j])
        offspring_matrix=np.stack((offspring1,offspring2,offspring3))
        offspring_fitness=calcola_pop_fitness(offspring_matrix)
        pos_offspring=np.where(offspring_fitness==max(offspring_fitness))[0][0]
        offspring[i]=offspring_matrix[pos_offspring]
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def simulated_binary_crossover(parents,offspring_size): #Deb and Agrawal (1995)
    #idea di fondo è quella di estrarre i figli da una distribuzione empirica
    #secondo un parametro mu che e' un elastico tra exploration ed exploitation
    #si avranno figli più o meno o simili ai genitori a seconda di beta (e mu)
    #con beta=1 -->c.d 'stationary crossover'
    mu=0.05
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]):
        parent1_pos = i%parents.shape[0] ##resto genitore 1 (0, poi 1,2,3,...)
        parent2_pos = (i+1)%parents.shape[0] 
        r=np.random.uniform(0,1)
        if r<0.5:
            beta=(2*r)**(1/(mu+1))
        else:
            beta=(2-2*r)**(-1/(mu+1))
        if i%2==0:
            offspring[i]=np.abs(0.5*((1-beta)*parents[parent1_pos]+(1+beta)*parents[parent2_pos]))
        else:
            offspring[i]=np.abs(0.5*((1+beta)*parents[parent1_pos]+(1-beta)*parents[parent2_pos]))           
         ##completato il primo loop, viene generato il primo figlio
    return offspring

def shuffle_crossover(parents,offspring_size): ##Eshelman (1989)
    offspring=np.zeros(offspring_size)
    crossover_point=np.random.randint(1,geni-1)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0] ##resto genitore 1 (0, poi 1,2,3,...)
        parent2_pos = (i+1)%parents.shape[0] ##resto genitore 2 (all'ultimo step riparte da zero) 1-2-3-...-0
        parents[parent1_pos]=np.random.permutation(parents[parent1_pos])
        parents[parent2_pos]=np.random.permutation(parents[parent2_pos])
        offspring[i, :crossover_point] = parents[parent1_pos, :crossover_point] ##meta' dei geni dal primo genitore
        offspring[i, crossover_point:] = parents[parent2_pos, crossover_point:] ##meta' dei geni dal secondo genitore
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def ring_crossover(parents,offspring_size): #Yilmaz Kaya, Murat Uyar, Ramazan Tekin (2011)
    #il metodo seleziona una lista con tre passi:
    #si fondono i parents in un'unica lista
    #si sceg2lie un punto di taglio e si procede in senso orario/antiorario
    #selezionando n elementi
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]):
        parent1_pos=i%parents.shape[0]
        parent2_pos=(i+1)%parents.shape[0]
        aux=np.hstack((parents[parent1_pos],parents[parent2_pos]))
        start=np.random.randint(0,len(aux))
        offspring_aux=[]
        u=np.random.uniform(0,1)
        if u>0.5:
            for j in range(geni):
                offspring_aux.append(aux[(start+j)%aux.shape[0]]) #clockwise
        else:
            for j in range(geni):
                offspring_aux.append(aux[(start-j)%aux.shape[0]]) #anticlockwise
        offspring[i]=offspring_aux
    return offspring

def intermediate_crossover(parents, offspring_size): #Mühlenbein, H. and Schlierkamp-Voosen, D. (1993)
    #media pesata dei geni secondo parametri user-defined
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): ##loop per riga
        beta=np.random.uniform(-0.25,1.25)
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
            if j%2==0:
                offspring[i,j]=np.abs(beta*parents[parent1_pos,j]+(1-beta)*parents[parent2_pos,j])
            else:
                offspring[i,j]=np.abs(beta*parents[parent2_pos,j]+(1-beta)*parents[parent1_pos,j])
         ##completato il primo loop, viene generato il primo figlio
    return offspring

def line_crossover(parents, offspring_size): #Mühlenbein, H. and Schlierkamp-Voosen, D. (1993)
    #media pesata dei geni secondo parametri user-defined
    offspring=np.zeros(offspring_size)
    beta=np.random.uniform(-0.25,1.25)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
            if j%2==0:
                offspring[i,j]=np.abs(beta*parents[parent1_pos,j]+(1-beta)*parents[parent2_pos,j])
            else:
                offspring[i,j]=np.abs(beta*parents[parent2_pos,j]+(1-beta)*parents[parent1_pos,j])
         ##completato il primo loop, viene generato il primo figlio
    return offspring

def queen_bee_crossover(parents,offspring_size):  ##Karc (2004)
    ##queen bee crossover con doppio taglio
    ##si sele2ziona il vettore 'regina' (con maggiore fitness)
    ##si effettuano operazioni di crossover con gli altri parents tenendo fermo il vettore 'regina'
    offspring=np.zeros(offspring_size)
    queen_bee=np.where(calcola_pop_fitness(parents)==max((calcola_pop_fitness(parents))))[0][0]
    parent1_pos=queen_bee
    for i in range(offspring_size[0]):
        #parent1_pos=i%parents.shape[0]
        parent2_pos=(parent1_pos+(i+1))%parents.shape[0]
        crossover_point_1=np.random.randint(1,geni-1)
        crossover_point_2=np.random.randint(crossover_point_1,geni-1)
        if i%2==0:
            offspring[i,:crossover_point_1]=parents[parent1_pos,:crossover_point_1]
            offspring[i,crossover_point_1:crossover_point_2]=parents[parent2_pos,crossover_point_1:crossover_point_2]
            offspring[i,crossover_point_2:]=parents[parent1_pos,crossover_point_2:] 
        else:
            offspring[i,:crossover_point_1]=parents[parent2_pos,:crossover_point_1]
            offspring[i,crossover_point_1:crossover_point_2]=parents[parent1_pos,crossover_point_1:crossover_point_2]
            offspring[i,crossover_point_2:]=parents[parent2_pos,crossover_point_2:] 
    return offspring

def laplace_crossover(parents,offspring_size): #Deep and Thakur (2007a)
    #crossover è basato su insieme di estrazioni random dalla distribuzione
    #di Laplace. Come per SBX-alpha, idea di fondo è usare paramatri (a&b)
    #per muoversi tra exploration ed exploitation
    offspring=np.zeros(offspring_size)
    a=0
    b=5.0
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        alpha=np.random.uniform(0,1)
        if alpha>0.5:
            beta=a-b*np.log(alpha)
        else:
            beta=a+b*np.log(alpha)
        if i%2==0:
            offspring[i]=np.abs(parents[parent1_pos]+beta*np.abs(parents[parent1_pos]-parents[parent2_pos]))
        else:
            offspring[i]=np.abs(parents[parent2_pos]+beta*np.abs(parents[parent1_pos]-parents[parent1_pos]))
    return offspring

def parent_centric_crossover(parents, offspring_size): #Garcia Martinez et al (2008)
    #versione modificata del blend crossover (BLX-alpha), Deb et al.
    #notano che la parametrizzazione determina come sempre mix tra
    #exploitation ed exploration, producendo soluzioni piu' o meno
    #vicine ai genitori
    alpha=0.5
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]):
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
                a=np.random.uniform(0,min(parents[parent1_pos,j],parents[parent2_pos,j]))
                b=np.random.uniform(max(parents[parent1_pos,j],parents[parent2_pos,j]),1)
                I=np.abs(parents[parent1_pos,j]-parents[parent2_pos,j])
                l=max(a,parents[parent1_pos,j]-I*alpha)
                u=max(b,parents[parent2_pos,j]+I*alpha)
                offspring[i,j]=np.random.uniform(l,u)
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def direction_based_crossover(parents, offspring_size): #Arumugam et al (2005)
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): ##loop per riga
        r=np.random.uniform(0,1)
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0]
        vstack=np.vstack((parents[parent1_pos],parents[parent2_pos]))
        #fitness=np.where(calcola_pop_fitness(vstack,0.5)==max((calcola_pop_fitness(vstack,0.5))))[0][0]
        if calcola_pop_fitness(vstack)[0]>=calcola_pop_fitness(vstack)[1]:
            offspring[i]=r*(np.abs(parents[parent1_pos]-parents[parent2_pos]))+parents[parent2_pos]
        else:
            offspring[i]=r*(np.abs(parents[parent2_pos]-parents[parent1_pos]))+parents[parent1_pos]
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def geometrical_crossover(parents, offspring_size): #Michalewicz et al.(1996)
    offspring=np.zeros(offspring_size)
    ###al 50% (p) il peso viene ereditato da un genitore, al 50% dall'altro (1-p), si completa un loop e si passa al figlio successivo
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0] ##resto genitore 1 (0, poi 1,2,3,...)
        parent2_pos = (i+1)%parents.shape[0] ##resto genitore 2 (all'ultimo step riparte da zero) 1-2-3-...-0
        for j in range(geni):
            offspring[i,j] =np.sqrt(parents[parent1_pos,j]*parents[parent2_pos,j])
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def sphere_crossover(parents, offspring_size): #Michalewicz et al.(1996)
    alpha=0.5
    offspring=np.zeros(offspring_size)
    ###al 50% (p) il peso viene ereditato da un genitore, al 50% dall'altro (1-p), si completa un loop e si passa al figlio successivo
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0] ##resto genitore 1 (0, poi 1,2,3,...)
        parent2_pos = (i+1)%parents.shape[0] ##resto genitore 2 (all'ultimo step riparte da zero) 1-2-3-...-0
        for j in range(geni):
            offspring[i,j] =np.sqrt(alpha*parents[parent1_pos,j]+(1-alpha)*parents[parent2_pos,j])
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def simplex_crossover(parents,offspring_size):
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]):
        parent1_pos=i%parents.shape[0]
        parent2_pos=(i+1)%parents.shape[0]
        parent3_pos=(i+2)%parents.shape[0]
        vstack=np.vstack((parents[parent1_pos],parents[parent2_pos],parents[parent3_pos]))
        pos_peggiore_fitness=np.where(calcola_pop_fitness(vstack)==min(calcola_pop_fitness(vstack)))[0][0]
        pos_migliore_fitness=np.where(calcola_pop_fitness(vstack)==max(calcola_pop_fitness(vstack)))[0][0]
        best_parents=np.delete(vstack,(pos_peggiore_fitness),axis=0)
        centroid=np.sum(best_parents,axis=0)/(len(vstack)-1)
        offspring[i]=centroid+(np.abs(centroid-vstack[pos_peggiore_fitness]))
    return offspring

def fuzzy_crossover(parents,offspring_size): #Voigt 1995
    offspring=np.zeros(offspring_size)
    d=0.5
    for i in range(offspring_size[0]):
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0]
        for j in range(geni):
            if parents[parent1_pos,j]<parents[parent2_pos,j]:
                phi_1=random.triangular(parents[parent1_pos,j]-d*np.abs(parents[parent2_pos,j]-parents[parent1_pos,j]),parents[parent1_pos,j]+d*np.abs(parents[parent2_pos,j]-parents[parent1_pos,j]),parents[parent1_pos,j])
                phi_2=random.triangular(parents[parent2_pos,j]-d*np.abs(parents[parent2_pos,j]-parents[parent1_pos,j]),parents[parent2_pos,j]+d*np.abs(parents[parent2_pos,j]-parents[parent1_pos,j]),parents[parent2_pos,j]) 
            else:
                phi_2=random.triangular(parents[parent1_pos,j]-d*np.abs(parents[parent2_pos,j]-parents[parent1_pos,j]),parents[parent1_pos,j]+d*np.abs(parents[parent2_pos,j]-parents[parent1_pos,j]),parents[parent1_pos,j])
                phi_1=random.triangular(parents[parent2_pos,j]-d*np.abs(parents[parent2_pos,j]-parents[parent1_pos,j]),parents[parent2_pos,j]+d*np.abs(parents[parent2_pos,j]-parents[parent1_pos,j]),parents[parent2_pos,j]) 
            offspring[i,j]=random.choice([np.abs(phi_1),np.abs(phi_2)])
    return offspring

def unimodal_crossover(parents,offspring_size): #Ono 1997
    offspring=np.zeros(offspring_size)
    std1=0.25
    std2=0.05
    for i in range(offspring_size[0]):
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0]
        parent3_pos = (i+2)%parents.shape[0]
        #x_p=0.5*(parents[parent1_pos]+parents[parent2_pos])
        g=0.5*(parents[parent1_pos]+parents[parent2_pos])
        d1=parents[parent1_pos]-g
        d2=parents[parent2_pos]-g
        d3=parents[parent3_pos]-g
        e1=d1/np.abs(d1)
        e2=d2/np.abs(d2)
        e3=d3/np.abs(d3)
        d=parents[parent2_pos]-parents[parent1_pos]
        aux=(np.random.normal(0,std1)*e1*np.abs(d1))+(np.random.normal(0,std1)*e2*np.abs(d2))
        D=np.linalg.norm(parents[parent3_pos]-g)
        offspring[i,:]=np.abs(g+aux+np.random.normal(0,std2)*D*e3)
        #D=(1-(np.dot(parents[parent3_pos]-parents[parent1_pos].T,parents[parent2_pos]-parents[parent1_pos])/np.dot(np.abs(parents[parent3_pos]-parents[parent1_pos]),np.abs(parents[parent2_pos]-parents[parent1_pos])))**2)**0.5
        #D=np.dot(np.abs(parents[parent3_pos]-parents[parent1_pos]),D)
        #offspring[i,:]=np.abs(x_p+np.random.normal(0,0.25)*d+((d/np.abs(d))*D*np.random.normal(0,0.1)))
    return offspring

def parent_centric_normal_crossover(parents,offspring_size):
    offspring=np.zeros(offspring_size)
    eta=0.05
    for i in range(offspring_size[0]):
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
            w=np.random.uniform(0,1)
            if w<0.5:
                offspring[i,j]=np.abs(np.random.normal(parents[parent1_pos,j],np.abs(parents[parent2_pos,j]-parents[parent1_pos,j])/eta))
            else:
                offspring[i,j]=np.abs(np.random.normal(parents[parent2_pos,j],np.abs(parents[parent2_pos,j]-parents[parent1_pos,j])/eta))
    return offspring

def mutation(offspring_crossover,prob_mutation):
    for i in range(offspring_crossover.shape[0]):  
        for j in range(offspring_crossover.shape[1]):
            estraz_outer=random.uniform(0,1)
            if estraz_outer<(prob_mutation): ##probab pari a lunghezza cromosoma
                estraz_inner=random.uniform(0,1)
                if estraz_inner>0.5:
                    offspring_crossover[i,j]=offspring_crossover[i,j]*1.1
                else:
                    offspring_crossover[i,j]=offspring_crossover[i,j]*0.9
    return offspring_crossover

def gaussian_mutation(offspring_crossover,prob_mutation): #gaussian mutation centrata rispetto all'offspring corrente
    for i in range(offspring_crossover.shape[0]):         #search domain (0,1)  
        for j in range(offspring_crossover.shape[1]):
            estraz_outer=random.uniform(0,1)
            if estraz_outer<(prob_mutation):
                offspring_crossover[i,j]=max(min(1,np.random.normal(offspring_crossover[i,j],1)),0)               
    return offspring_crossover

def uniform_mutation(offspring_crossover,prob_mutation):
    for i in range(offspring_crossover.shape[0]):
        for j in range(offspring_crossover.shape[1]):
            estraz_outer=random.uniform(0,1)
            if estraz_outer<(prob_mutation):
                offspring_crossover[i,j]=np.random.uniform(0,1) #uniform mutation centrata nel search domain (0,1)           
    return offspring_crossover





def lista_crossover(parents,offspring_size):
    c1=crossover(parents,offspring_size)
    c2=crossover_uniforme(parents,offspring_size)
    c3=crossover_uniforme_globale(parents,offspring_size)
    c4=flat_crossover(parents,offspring_size)
    c5=blend_crossover(parents,offspring_size)
    c6=average_crossover(parents,offspring_size)
    c7=simulated_binary_crossover(parents,offspring_size)
    #c8=shuffle_crossover(parents,offspring_size)
    c9=intermediate_crossover(parents,offspring_size)
    c10=geometrical_crossover(parents,offspring_size)
    c11=arithmetic_crossover(parents,offspring_size)
    c12=laplace_crossover(parents,offspring_size)
    c13=two_point_crossover(parents,offspring_size)
    c14=queen_bee_crossover(parents,offspring_size)
    c16=ring_crossover(parents,offspring_size)
    c17=parent_centric_crossover(parents,offspring_size)
    c18=heuristic_crossover(parents,offspring_size)
    c19=three_point_crossover(parents,offspring_size)
    c20=line_crossover(parents,offspring_size)
    c21=sphere_crossover(parents,offspring_size)
    c22=multi_parent_average_crossover(parents,offspring_size)
    c23=gene_pool_crossover(parents,offspring_size)
    c24=linear_crossover(parents,offspring_size)
    c25=simplex_crossover(parents,offspring_size)
    c26=arnone_crossover(parents,offspring_size)
    return c1,c2,c3,c4,c5,c6,c7,c10,c11,c12,c13,c14,c18,c19,c20
    #return c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c25

def lista_crossover2(parents,offspring_size):
    aux_list=[crossover(parents,offspring_size),crossover_uniforme(parents,offspring_size),heuristic_crossover(parents,offspring_size),
              laplace_crossover(parents,offspring_size),queen_bee_crossover(parents,offspring_size),two_point_crossover(parents,offspring_size),
                arithmetic_crossover(parents,offspring_size),geometrical_crossover(parents,offspring_size),
                simulated_binary_crossover(parents,offspring_size),average_crossover(parents,offspring_size),
                blend_crossover(parents,offspring_size),flat_crossover(parents,offspring_size),
                crossover_uniforme_globale(parents,offspring_size),three_point_crossover(parents,offspring_size),
                linear_crossover(parents,offspring_size),direction_based_crossover(parents,offspring_size),unimodal_crossover(parents,offspring_size),
              fuzzy_crossover(parents,offspring_size),simplex_crossover(parents,offspring_size),parent_centric_normal_crossover(parents,offspring_size)]
    return aux_list

listone=[crossover,crossover_uniforme,heuristic_crossover,
              laplace_crossover,queen_bee_crossover,two_point_crossover,
                arithmetic_crossover,geometrical_crossover,
                simulated_binary_crossover,average_crossover,
                blend_crossover,flat_crossover,
                crossover_uniforme_globale,three_point_crossover,
                linear_crossover,direction_based_crossover,unimodal_crossover,
              fuzzy_crossover,simplex_crossover,parent_centric_normal_crossover]

geni=len(data.columns)  ##geni appartenenti all'i-esima soluz (cromosoma) 2*cromosomi (es per funzione in due variabili)

cromosomi=20 #numero soluzioni generate
num_parents=5 ##numero di genitori scelti per elitist selection
simulazioni=1
num_generazioni=1000
#dimensione popolazione
pop_size=(cromosomi,geni)

#####
#offspring_size=(pop_size[0]-parents.shape[0]


######


contatore_fitness_max=[]
contatore_fitness_media=[]
contatore_entropy=[]


#def aggregated_criteria_computation():
    #calcolo della variazione fitness media e determinazione di fwin che registra impatto applicazione criteri qualita'/diversita'
#    delta_fitness_media=df_contatore_fitness_media.diff().dropna()
#    delta_entropy=df_contatore_fitness_media.diff().dropna()
#    aux1=delta_fitness_media.mean()
#    aux2=delta_entropy.mean()
#    Fwin1=pd.DataFrame(aux1)
#    Fwin2=pd.DataFrame(aux2)
#    return Fwin1, Fwin2



start=time.time()
window=10
num_crossover=20

fitness_modulo1=np.zeros((num_generazioni,num_crossover))
entropy_modulo1=np.zeros((num_generazioni,num_crossover))
deltafitness_modulo1=np.zeros((num_generazioni-window,num_crossover))
deltaentropy_modulo1=np.zeros((num_generazioni-window,num_crossover))
credit_reward_list=np.zeros((num_generazioni-window,num_crossover))
credit_reward_aggregation=np.zeros((num_generazioni-2*window,num_crossover))
ap_selection=np.zeros((num_generazioni-2*window,num_crossover))
ap_selection[0,:]=np.ones(num_crossover)/num_crossover
#ph=PageHinkley(min_instances=400, delta=0.00015, threshold=0.0001, alpha=0.9999)

def aggregated_criteria_computation(fitness_modulo1,entropy_modulo1):
    #calcolo della variazione fitness media e determinazione di fwin che registra impatto applicazione criteri qualita'/diversita'
    if i>=window:
        delta_fitness_media=pd.DataFrame(fitness_modulo1[i-window:i,:]).diff()
        delta_entropy=pd.DataFrame(entropy_modulo1[i-window:i,:]).diff()
        Fwin1=delta_fitness_media.mean()
        Fwin2=delta_entropy.mean()
        deltafitness_modulo1[i-window,:]=Fwin1
        deltaentropy_modulo1[i-window,:]=Fwin2
        return Fwin1, Fwin2, deltafitness_modulo1

def reward_computation(theta):
    store_reward=np.zeros(num_crossover)
    if i>=window:
        for s in range(len(store_reward)):
            origine=[0,0]
            Fwin1=aggregated_criteria_computation(fitness_modulo1,entropy_modulo1)[0][s] #input procedura
            Fwin2=aggregated_criteria_computation(fitness_modulo1,entropy_modulo1)[1][s] #input procedura
            x=np.linspace(0,np.max(1.5*Fwin1),100)
            m=np.tan(theta) #slope
            y=m*x
            #inizializzazione
            #dp=np.zeros(len(Fwin1)) 
            #dpp=np.zeros(len(Fwin1))
            #reward=np.zeros(len(Fwin1))
            ##calcolo distanze
            #for i in range(len(Fwin1)):
            dp=abs((Fwin2-(m*Fwin1)))/(np.sqrt(1+m**2)) #retta in forma esplicita, distanza perpendicolare
            dpp=np.sqrt((Fwin1-origine[0])**2+(Fwin2-origine[1])**2) #distanza tra due punti
            reward=np.sqrt(dpp**2-dp**2)
            store_reward[s]=reward
        return store_reward

prova=np.zeros(1000)


def increasing_strategy():
    angle=0
    if i>=window:
        if i<=num_generazioni/4:
            angle=0
        elif i>=num_generazioni/4 and i<=num_generazioni/2:
            angle=np.pi/6
        elif i>=num_generazioni/2 and i<=num_generazioni*(3/4):
            angle=np.pi/3
        elif i>=num_generazioni*(3/4):
            angle=np.pi/2
    return angle

def decreasing_strategy():
    angle=np.pi/2
    if i>=window:
        if i<=num_generazioni/4:
            angle=np.pi/2
        elif i>=num_generazioni/4 and i<=num_generazioni/2:
            angle=np.pi/3
        elif i>=num_generazioni/2 and i<=num_generazioni*(3/4):
            angle=np.pi/6
        elif i>=num_generazioni*(3/4):
            angle=0
    return angle

def always_moving_strategy():
    angle=np.pi/2
    if i>=window:
        if i<=num_generazioni/5:
            angle=np.pi/2
        elif i>=num_generazioni*(1/5) and i<=num_generazioni*(2/5):
            angle=0
        elif i>=num_generazioni*(2/5) and i<=num_generazioni*(3/5):
            angle=np.pi/2
        elif i>=num_generazioni*(3/5) and i<=num_generazioni*(4/5):
            angle=0
        elif i>=num_generazioni*(4/5):
            angle=np.pi/2
    return angle


def reactive_moving_strategy():
    angle=0
    if i>=window:
        if  ((entropy_list[i-1]-entropy_list[i-window])/(entropy_list[i-window]))<(-1/100):
            angle=0
        elif np.abs((fitness_media[i-1]-fitness_media[i-window])/(fitness_media[i-window]))<(1/100):
            angle=np.pi/2
        else:
            angle=0
    return angle




def credit_assignment(strategy):
    if i>=window:
        angle=strategy
        store_reward=reward_computation(angle)
        prova[i]=store_reward[19]
        credit_reward_list[i-window,:]=store_reward
    if i>=2*window:
        credit_reward_aggregation[i-2*window,:]=np.mean(pd.DataFrame(credit_reward_list[i-2*window:i-window,:]).dropna())
    return credit_reward_aggregation



def operator_selection(credit_function): #Probability Matching (PM)
    choose_op='pm'
    if choose_op=='pm':
        p_min=0.01
        K=num_crossover
        idx=np.random.randint(0,19)
        credito=credit_function
        if i>=2*window:
            #credito=credit_assignment()
            wheel_selection=p_min+(1-K*p_min)*(credito[i-2*window,:]/(np.sum(credito[i-2*window,:])))
            print(wheel_selection)
            wheel_selection=np.cumsum(wheel_selection)
            u=np.random.uniform(0,1)
            for c in range(len(wheel_selection)):
                if u<wheel_selection[c]:
                    idx=c
                    #print(idx)
                    break
    elif choose_op=='mab':
        idx=i
        credito=credit_function
        C=0.00001
        aux=np.sum(matrice_memoria_operatori,axis=1)
        print(aux)
        if i>=2*window:
            print(credito[i-2*window,:])
            #credito=credit_assignment()
            mab_selection=credito[i-2*window,:]+C*np.sqrt(np.log(np.sum(aux))/aux)
            c=np.where(mab_selection==max(mab_selection))
            idx=c[0][0]
            for j in range(num_crossover):
                ph.add_element(credito[i-2*window,j])
                if ph.detected_change():
                    print('restart')
    elif choose_op=='ap':
        beta=0.5
        idx=np.random.randint(0,19)
        p_min=0.01
        p_max=1-(num_crossover-1)*p_min
        credito=credit_function
        #aux=np.sum(matrice_memoria_operatori,axis=1)
        #print(aux)
        if i>=2*window and i<num_generazioni:
            best_ip=np.where(credito[i-2*window,:]==max(credito[i-2*window,:]))
            best_ip=best_ip[0][0]
            best_ip_succ=np.where(credito[i-2*window+1,:]==max(credito[i-2*window+1,:]))
            best_ip_succ=best_ip_succ[0][0]
            print(best_ip)
            print(best_ip_succ)
            #credito=credit_assignment()
            for j in range(num_crossover):
                ap_selection[i-2*window+1,j]=ap_selection[i-2*window,j]+beta*(p_min-ap_selection[i-2*window,j])
            ap_selection[i-2*window+1,best_ip_succ]=ap_selection[i-2*window,best_ip]+beta*(p_max-ap_selection[i-2*window,best_ip])
            print(ap_selection[i-2*window,:])
            #wheel_selection=np.cumsum(ap_selection)
            #u=np.random.uniform(0,1)
            #for c in range(len(wheel_selection)):
            #    if u<wheel_selection[c]:
            #        idx=c
    return idx


##def multi_armed_bandit(credit_function): #MAB (Multi armed bandit)
##    idx=np.random.randint(0,19)
##    credito=credit_function
##    C=5
##    aux=np.sum(matrice_memoria_operatori,axis=1)
##    if i>=2*window:
##        #credito=credit_assignment()
##        mab_selection=credito[i-2*window,:]+C*np.sqrt(np.log(aux)/np.sum(aux))
##        c=np.where(mab_selection==max(mab_selection))
##        idx=c[0][0]
##    return idx
        

selection='elitist'
strategy='always'

for w in range(1):

    fitness_max_esterno=[]
    fitness_media_esterno=[]
    entropy_list_esterno=[]
    idx=0
    matrice_memoria_operatori=np.zeros((num_crossover,4))
    memoria_angolo=np.zeros(num_generazioni)
    fitness_individui=np.zeros((num_generazioni,cromosomi))
    contatore=0

    for k in range(simulazioni):

        #popolazione (oggetto di selezione->crossover(per generare 95 figli a partire da 5 genitori)->mutazione
        nuova_pop=np.random.uniform(low=0, high=1, size=pop_size)
        

        fitness_max=[]
        fitness_media=[]
        entropy_list=[]

        for i in range(num_generazioni):
            fitness=calcola_pop_fitness(nuova_pop)
            fitness_individui[i,:]=-fitness
            entropy=calcola_entropy(nuova_pop)
            average_fitness=np.sum(fitness)/cromosomi
            if selection=='elitist':
                parents=elitist_selection(nuova_pop,fitness,num_parents)
                #idx=15
                offspring_crossover=listone[idx](parents,offspring_size=(pop_size[0]-parents.shape[0], geni))
                ################################adaptive operator selection
                #soluzioni_crossover_15=listone[0:num_crossover](parents,offspring_size=(pop_size[0]-parents.shape[0], geni))
                soluzioni_crossover_15=np.zeros((num_crossover,cromosomi,data.shape[1]))
                for opti in range(num_crossover):
                    soluzioni_crossover_15[opti,:,:]=listone[opti](parents,offspring_size=(pop_size[0], geni))
            ################################adaptive operator selection
            #soluzioni_crossover_15=lista_crossover2(parents,offspring_size=(pop_size[0]-parents.shape[0], geni))[0:num_crossover]
            #average_fitness_15=np.zeros(num_crossover)
            #entropy_15=np.zeros(num_crossover)
            #for u in range(num_crossover):
            #    average_fitness_15[u]=np.sum(calcola_pop_fitness(soluzioni_crossover_15[u]))/cromosomi
            #    entropy_15[u]=calcola_entropy(soluzioni_crossover_15[u])
            average_fitness_15=np.array([np.sum(calcola_pop_fitness(soluzioni_crossover_15[u]))/cromosomi for u in range(num_crossover)])
            entropy_15=np.array([calcola_entropy(soluzioni_crossover_15[u]) for u in range(num_crossover)])
            fitness_modulo1[i,:]=average_fitness_15
            entropy_modulo1[i,:]=entropy_15
            if strategy=='reactive':
                angolo=reactive_moving_strategy()
                memoria_angolo[i]=angolo
                idx=operator_selection(credit_assignment(reactive_moving_strategy()))
            elif strategy=='always':
                angolo=always_moving_strategy()
                memoria_angolo[i]=angolo
                idx=operator_selection(credit_assignment(always_moving_strategy()))
            elif strategy=='decreasing':
                angolo=decreasing_strategy()
                memoria_angolo[i]=angolo
                idx=operator_selection(credit_assignment(decreasing_strategy()))
            elif strategy=='increasing':
                angolo=increasing_strategy()
                memoria_angolo[i]=angolo
                idx=operator_selection(credit_assignment(increasing_strategy()))
            print(idx)
            if i>0:
                if contatore==0:
                    matrice_memoria_operatori[idx,contatore]=matrice_memoria_operatori[idx,contatore]+1
                if memoria_angolo[i]!=memoria_angolo[i-1]:
                    contatore=contatore+1
                if contatore==1:
                    matrice_memoria_operatori[idx,contatore]=matrice_memoria_operatori[idx,contatore]+1
                if contatore==2:
                    matrice_memoria_operatori[idx,contatore]=matrice_memoria_operatori[idx,contatore]+1
                if contatore==3:
                    matrice_memoria_operatori[idx,contatore]=matrice_memoria_operatori[idx,contatore]+1
             ######
##            if angolo==0:
##                matrice_memoria_operatori[idx,0]=matrice_memoria_operatori[idx,0]+1
##            elif angolo==np.pi/6:
##                matrice_memoria_operatori[idx,1]=matrice_memoria_operatori[idx,1]+1
##            elif angolo==np.pi/3:
##                matrice_memoria_operatori[idx,2]=matrice_memoria_operatori[idx,2]+1
##            elif angolo==np.pi/2:
##                matrice_memoria_operatori[idx,3]=matrice_memoria_operatori[idx,3]+1
            ################################
            #offspring_crossover=lista_crossover2(parents,offspring_size=(pop_size[0], geni))[w]
            #offspring_crossover=arnone_crossover(parents,offspring_size=(pop_size[0]-parents.shape[0], geni))
            #offspring_crossover=lista_crossover(parents,offspring_size=(pop_size[0]-parents.shape[0], geni))[w]
            offspring_mutation=mutation(offspring_crossover,0.25)
            if selection=='elitist':
                nuova_pop[:parents.shape[0], :] = parents
                nuova_pop[parents.shape[0]:, :] = offspring_mutation
            #nuova_pop=offspring_crossover
            #nuova_pop[0][nuova_pop[0].argsort()[:5]]=0
            for j in range(len(nuova_pop)):
                #nuova_pop[j][nuova_pop[j].argsort()[:5]]=0
                nuova_pop[j]=nuova_pop[j]/np.sum(nuova_pop[j])
            pos_fitness_migliore=np.where(fitness == np.max(fitness))
            pos_fitness_migliore=pos_fitness_migliore[0][0]
            #fitness_max.append(fitness[pos_fitness_migliore])
            fitness_media.append(average_fitness)
            entropy_list.append(entropy)
        #fitness_max_esterno.append(fitness_max)
        fitness_media_esterno.append(fitness_media)
        entropy_list_esterno.append(entropy_list)
        
        
    #fitness=calcola_pop_fitness(nuova_pop,0.5) ##popolazione finale-->se ne calcola la fitness 'definitiva'
    #pos_fitness_migliore=np.where(fitness == np.max(fitness))
    #pos_fitness_migliore=pos_fitness_migliore[0][0]
    #print("Migliore soluzione ammissibile (feasible) : ", nuova_pop[pos_fitness_migliore, :])
    #print("Fitness : ", fitness[pos_fitness_migliore])


    
    #df_fitness_max=pd.DataFrame(np.transpose(fitness_max_esterno))
    #media_simulazioni_fitness_max=np.mean(df_fitness_max,axis=1)
    df_fitness_media=pd.DataFrame(np.transpose(fitness_media_esterno))
    media_simulazioni_fitness_media=np.mean(df_fitness_media,axis=1)
    df_entropy=pd.DataFrame(np.transpose(entropy_list_esterno))
    media_simulazioni_entropy=np.mean(df_entropy,axis=1)
    #contatore_fitness_max.append(media_simulazioni_fitness_max)
    contatore_fitness_media.append(media_simulazioni_fitness_media)
    contatore_entropy.append(media_simulazioni_entropy)
    #fig,ax = plt.subplots()
    #ax.plot(np.mean(df_fitness_media.iloc[1:,:],axis=1),c='b')
    #ax.set_ylabel('Fitness media')
    #ax.legend(['Fitness media'])
    #ax2=ax.twinx()
    #ax2.plot(np.mean(df_entropy.iloc[1:,:],axis=1),c='r')
    #ax2.set_ylabel('Entropy')
    #ax2.legend(['Entropy'])
    #plt.show()

    
    print(media_simulazioni_fitness_media)

    end=time.time()
    print(end-start)



#df_contatore_fitness_max=pd.DataFrame(np.transpose(contatore_fitness_max))
df_contatore_fitness_media=pd.DataFrame(np.transpose(contatore_fitness_media))
df_contatore_entropy=pd.DataFrame(np.transpose(contatore_entropy))
#fig,(ax1,ax2,ax3)=plt.subplots(1,3)
#ax1.plot(df_contatore_fitness_max.iloc[1:,:])
#ax1.set(ylabel='Sortino Ratio')
#ax1.set(ylabel='λE(R)-(1-λ)var(R)')
#ax1.set(xlabel='Generazioni')
#ax1.set_title('Max Fitness')
#ax2.plot(df_contatore_fitness_media.iloc[1:,:])
#ax2.set(xlabel='Generazioni')
#ax2.set_title('Fitness Media')
#ax3.plot(df_contatore_entropy.iloc[1:,:])
#ax3.set_title('Entropia')
#ax3.set(xlabel='Generazioni')
#plt.legend(['1-point crossover','uniform','global uniform',
#           'flat crossover','blend crossover','average crossover','SBX-alpha','shuffle crossover',
#           'intermediate crossover','geometrical crossover','arithmetic crossover',
#            'laplace crossover','two point crossover','queen_bee_crossover','line crossover',
#            'ring crossover','parent centric crossover','heuristic crossover'])
#plt.legend(['1-point crossover','uniform crossover','global uniform crossover',
#           'flat crossover','blend crossover','average crossover','SBX-alpha',
#            'geometrical crossover','arithmetic crossover','laplace crossover',
#            '2-point crossover','queen bee crossover','heuristic crossover',
#            '3-point crossover','line crossover'])
#plt.show()
#fig,(ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15)=plt.subplots(3,5)

lista_legenda=['1-point crossover','uniform crossover','heuristic crossover',
            'laplace crossover','queen bee crossover','two point crossover','arithmetic crossover',
            'geometrical crossover','SBX-alpha','average crossover',
            'blend crossover','flat crossover','uniform global crossover',
            '3-point crossover','line crossover']



#for i in range(5):
#    fig,(ax1,ax2,ax3)=plt.subplots(1,3)
#    fig.tight_layout(pad=4.0)
    #fig.suptitle('Risk measure: λE(R)-(1-λ)var')
#    fig.suptitle('Risk measure: Variance with Skewness')
#    ax1.plot(df_contatore_fitness_media.iloc[1:,3*i],c='b')
#    ax1.set_title(lista_legenda[3*i])
#    ax1.set_ylabel('Fitness media')
#    ax1.set_xlabel('Generazioni')
#    ax1.legend(['Fitness media'],loc=1)
#    ax11=ax1.twinx()
#    ax11.plot(df_contatore_entropy.iloc[1:,3*i],c='r')
#    ax11.set_ylabel('Entropy')
#    ax11.legend(['Entropy'],loc=4)
#    ax2.plot(df_contatore_fitness_media.iloc[1:,3*i+1],c='b')
#    ax2.set_title(lista_legenda[3*i+1])
#    ax2.set_ylabel('Fitness media')
#    ax2.set_xlabel('Generazioni')
#    ax2.legend(['Fitness media'],loc=1)
#    ax22=ax2.twinx()
#    ax22.plot(df_contatore_entropy.iloc[1:,3*i+1],c='r')
#    ax22.set_ylabel('Entropy')
#    ax22.legend(['Entropy'],loc=4)
#    ax3.plot(df_contatore_fitness_media.iloc[1:,3*i+2],c='b')
#    ax3.set_title(lista_legenda[3*i+2])
#    ax3.set_ylabel('Fitness media')
#    ax3.set_xlabel('Generazioni')
#    ax3.legend(['Fitness media'],loc=1)
#    ax33=ax3.twinx()
#    ax33.plot(df_contatore_entropy.iloc[1:,3*i+2],c='r')
#    ax33.set_ylabel('Entropy')
#    ax33.legend(['Entropy'],loc=4)
    #plt.show()
#ax4.plot(df_contatore_fitness_media.iloc[1:,3],c='b')
#ax4.set_ylabel('Fitness media')
#ax4.legend(['Fitness media'])
#ax44=ax4.twinx()
#ax44.plot(df_contatore_entropy.iloc[1:,3],c='r')
#ax44.set_ylabel('Entropy')
#ax44.legend(['Entropy'])
#ax5.plot(df_contatore_fitness_media.iloc[1:,4],c='b')
#ax5.set_ylabel('Fitness media')
#ax5.legend(['Fitness media'])
#ax55=ax5.twinx()
#ax55.plot(df_contatore_entropy.iloc[1:,4],c='r')
#ax55.set_ylabel('Entropy')
#ax55.legend(['Entropy'])
#plt.show()

##########plot probabilita selezione dell'i-esimo operatore###########
credito=credit_assignment(increasing_strategy())
roulette=np.zeros((num_generazioni-2*window,num_crossover))
p_min=0.01
for i in range(num_generazioni-2*window):
    roulette[i,:]=p_min+(1-num_crossover*p_min)*(credito[i-2*window,:]/(np.sum(credito[i-2*window,:])))
    
#fig, axs=plt.subplots(3,5)
#fig.tight_layout(pad=0.50)
#fig.suptitle('Probabilità di selezione per operatore di crossover. Misura di rischio: Sharpe Ratio',y=1.00)
#for i in range(5):
#    axs[0, i].plot(roulette[:,i])
#    axs[0, i].set_xlabel('generazioni')
#    axs[1, i].plot(roulette[:,i+5])
#    axs[1, i].set_xlabel('generazioni')
#    axs[2, i].plot(roulette[:,i+10])
#    axs[2, i].set_xlabel('generazioni')

fig, axs=plt.subplots(4,5)
fig.tight_layout(pad=0.50)
fig.suptitle('Probabilità di selezione. Omega + ALWAYS MOVING strategy',y=1.00)
for i in range(5):
    axs[0, i].plot(roulette[:,i])
    axs[0, i].set_xlabel('generazioni')
    axs[1, i].plot(roulette[:,i+5])
    axs[1, i].set_xlabel('generazioni')
    axs[2, i].plot(roulette[:,i+10])
    axs[2, i].set_xlabel('generazioni')
    axs[3, i].plot(roulette[:,i+15])
    axs[3, i].set_xlabel('generazioni')

#####################################################################

matrice_memoria_operatori=pd.DataFrame(matrice_memoria_operatori)
mmo=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
w=0.22
sns.set()
fig, axe=plt.subplots(4,1)
fig.tight_layout(pad=0.50)
axe[0].bar(mmo-2*w,matrice_memoria_operatori.iloc[:,0],width=w,align='center')
axe[0].bar(mmo-w,matrice_memoria_operatori.iloc[:,1],width=w,align='center')
axe[0].bar(mmo,matrice_memoria_operatori.iloc[:,2],width=w,align='center')
axe[0].bar(mmo+w,matrice_memoria_operatori.iloc[:,3]+w,width=w,align='center')
if strategy=='always':
    axe[0].legend(['π/2','0','π/2','0'],loc='upper left',prop={'size': 8})
elif strategy=='reactive':
    axe[0].legend(['0','π/2','0','π/2'],loc='upper left',prop={'size': 8})
elif strategy=='decreasing':
    axe[0].legend(['π/2','π/3','π/6','0'],loc='upper left',prop={'size': 8})
elif strategy=='increasing':
    axe[0].legend(['0','π/6','π/3','π/2'],loc='upper left',prop={'size': 8})
axe[0].set_xticks(mmo)
axe[0].set_xticklabels(['OPX','UX','HX','LX','QBX','TPX','AMX','GX','SBX','AVX','BLX-a','FX','GUX','TPX','LNX','DBX','UNDX','FR','SPX','PNX'],fontsize=6,rotation=90)
axe[0].set_ylabel('Operators Frequency')
axe[1].plot(media_simulazioni_entropy[1:],c='red')
axe[1].set_ylabel('Entropy')
axe[2].plot(memoria_angolo,c='r')
axe[2].set_ylabel('Angle')
for g in range(cromosomi):
	axe[3].scatter(np.linspace(1,num_generazioni,num_generazioni-1),fitness_individui[1:,g],s=5,c=fitness_individui[1:,g],cmap='RdYlBu')
axe[3].set_ylabel('Cost')




