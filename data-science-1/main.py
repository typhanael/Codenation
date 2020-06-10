#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[2]:


#%matplotlib inline
#from IPython import get_ipython
from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[3]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[4]:


# Sua análise da parte 1 começa aqui.

dataframe.isnull().sum()


# In[5]:


dataframe.columns


# In[6]:


dataframe.describe()


# In[7]:


dataframe.dtypes


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[8]:


def q1():
    # Retorne aqui o resultado da questão 1.
    
    [q1_norm, q2_norm,q3_norm,q1_binom, q2_binom,q3_binom]=[dataframe['normal'].quantile(q=0.25), dataframe['normal'].quantile(q=0.5),
                                                           dataframe['normal'].quantile(q=0.75),dataframe['binomial'].quantile(q=0.25),
                                                         dataframe['binomial'].quantile(q=0.5),dataframe['binomial'].quantile(q=0.75)]
    
    #val = (q1_norm - q1_binom, q2_norm - q2_binom, q3_norm - q3_binom)
    #arredondar para 3 casas decimais.
    
    return (round(q1_norm - q1_binom, ndigits=3),
            round(q2_norm - q2_binom, ndigits=3),
            round(q3_norm - q3_binom, ndigits=3))
    #pass
#q1()


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[52]:


def q2():
    # Retorne aqui o resultado da questão 2.
    #deve-se retornar um resultado float no return e não um float64...
    [media,desvio_padrao]=[dataframe['normal'].mean(),dataframe['normal'].std()]
    ecdf = ECDF(dataframe['normal'])
    result = ecdf(media+desvio_padrao)-ecdf(media-desvio_padrao)
    #fazendo cast do resultado formatado antes do envio...
    return float(round(result, ndigits=3))
#pass
#q2()


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[10]:


def q3():
    # Retorne aqui o resultado da questão 3.
    [m_binom,v_binom, m_norm,v_norm] = [dataframe['binomial'].mean(),dataframe['binomial'].var(),
                                         dataframe['normal'].mean(),dataframe['normal'].var()]

    return (round(m_binom - m_norm, ndigits = 3), 
            round(v_binom - v_norm, ndigits=3))
    #pass
#q3()


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[19]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[20]:


# Sua análise da parte 2 começa aqui.
stars.columns


# In[53]:


stars.head(10)


# In[55]:


stars.isnull().sum()


# In[56]:


stars.describe()


# In[60]:


stars.dtypes


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[88]:


def padronizer(data):
    med = data.mean()
    dsv = data.std()
    res = (data-med)/dsv
    return res

def q4():
    # Retorne aqui o resultado da questão 4.
    aux = [0.80, 0.90, 0.95]
    
    target_zero = stars[stars['target'] == 0]['mean_profile']
    false_pulsar_mean_profile_standardized= padronizer(target_zero)
    ecdf=ECDF(false_pulsar_mean_profile_standardized)
    
    #calculando os valores para cada quantil e aplicando ecdf já no resultado...
    arr_q = [ecdf(sct.norm.ppf(q, loc = false_pulsar_mean_profile_standardized.mean(), 
                          scale=false_pulsar_mean_profile_standardized.std())) for q in aux]
    
    result=tuple([float(round(x, ndigits=3)) for x in arr_q])    
    return result
    #pass
#q4()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[97]:


def q5():
    # Retorne aqui o resultado da questão 5.
    target_zero = stars[stars['target'] == 0]['mean_profile']
    false_pulsar_mean_profile_standardized= padronizer(target_zero)
    
    aux=[0.25, 0.5, 0.75]
    
    #quantis padronizados
    fp_padronizer = [np.percentile(false_pulsar_mean_profile_standardized,q*100) for q in aux]
    
    #calculando distribuição normal
    qs_norm = [sct.norm.ppf(q, loc=0, scale=1) for q in aux]
    
    #calculando a diferença entre o quantil padronizado e a distribuição normal.
    #arredondando cada valor calculado para 3 casas decimais, com o valor sendo em float.
    #tranformando o resultado em uma tupla de valores.
    qs_diff = tuple([float(round(fp_padronizer[p]-qs_norm[p], ndigits=3)) for p in range(len(qs_norm))])
    
    return qs_diff
    #pass

#q5()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
