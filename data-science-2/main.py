#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[34]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns


import statsmodels.api as sm


# In[3]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[4]:


athletes = pd.read_csv("athletes.csv")


# In[5]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[6]:


# Sua análise começa aqui.
athletes.isnull().sum()


# In[7]:


athletes.describe()


# In[8]:


athletes.shape


# In[9]:


athletes.columns


# In[10]:


athletes.nationality.unique()


# In[11]:


alpha=5/100


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[12]:


def q1():
    # Retorne aqui o resultado da questão 1.
    df=get_sample(df=athletes, col_name='height', n=3000)
    
    #Array retornado:
    #Estatistica de teste, valor do teste de hipotese
    test_shapiro = sct.shapiro(df)
    
    #Sendo o nivel de significancia igual a 5%
    # o valor de teste de hipotese devera ser maior que 0.05
    return True if test_shapiro[1] > alpha else False

q1()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# In[39]:


sns.distplot(athletes['height'], bins = 25);
height_mean = athletes['height'].mean()
plt.axvline(height_mean, color='red', linestyle='dashed', linewidth=2)
min_ylim, max_ylim = plt.ylim()
plt.text(height_mean*1.05, max_ylim*0.9, 'Height Mean: {:.2f}'.format(height_mean))
plt.show()


# In[38]:


column = 'height'
sample = get_sample(athletes, 'height', 3000)
  
sm.qqplot(sample, fit= True, line ='45') 
plt.show()


# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[13]:


def q2():
    # Retorne aqui o resultado da questão 2.
    df=get_sample(df=athletes, col_name='height', n=3000)
    
    #Array retornado:
    #Estatistica de teste, valor do teste de hipotese
    test_jarque = sct.jarque_bera(df)
    
    return True if test_jarque[1] > alpha else False

q2()


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# In[40]:



sct.skew(sample), sct.kurtosis(sample)


# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[14]:


def q3():
    # Retorne aqui o resultado da questão 3.
    df=get_sample(df=athletes, col_name='weight', n=3000)
    
    #[statistic, pvalue]
    test_pearson=sct.normaltest(df)
    

    return True if test_pearson[1] > alpha else False

q3()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# In[41]:


sns.distplot(athletes['weight'], bins = 25);
height_mean = athletes['weight'].mean()
plt.axvline(height_mean, color='red', linestyle='dashed', linewidth=2)
min_ylim, max_ylim = plt.ylim()
plt.text(height_mean*1.05, max_ylim*0.9, 'Weight Mean: {:.2f}'.format(height_mean))
plt.show()


# In[42]:


sns.boxplot(athletes['weight'])


# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[15]:


def q4():
    # Retorne aqui o resultado da questão 4.
    df=get_sample(df=athletes, col_name='weight', n=3000)
    
    #fazendo a tranformação logarítmica e logo em seguida, o test de normalidade.
    test_log = np.log(df)
    test_pearson=sct.normaltest(test_log)
    return True if test_pearson[1] > alpha else False

    #pass
q4()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# In[43]:


plt.figure(figsize = (12,8))
sample = get_sample(athletes, 'weight', 3000)

log_transform_sample = np.log(sample)
weight_mean = log_transform_sample.mean()
sns.distplot(log_transform_sample, bins = 25);
plt.axvline(weight_mean, color='red', linestyle='dashed', linewidth=2)
min_ylim, max_ylim = plt.ylim()
plt.text(weight_mean*1.05, max_ylim*0.9, 'Weight Mean: {:.2f}'.format(weight_mean))
plt.show()


# In[44]:


sm.qqplot(log_transform_sample, fit = True, line='45')
plt.show()


# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[16]:


def q5():
    # Retorne aqui o resultado da questão 5.
    #[df_br, df_can, df_usa] =[athletes[(athletes['nationality'] == 'BRA')]['height'].dropna(), athletes[athletes['nationality'] == 'CAN'], athletes[athletes['nationality'] == 'USA')]]
    
    #Pegando os indivíduos que possui nacionalidade brasileira, canadense ou americana
    #e em seguida, pegando apenas os valores das suas alturas, retirando os valores nulos.
    [df_alt_br, df_alt_can, df_alt_usa] = [athletes[(athletes['nationality'] == 'BRA')]['height'].dropna(),
                                           athletes[(athletes['nationality'] == 'CAN')]['height'].dropna(),
                                           athletes[(athletes['nationality'] == 'USA')]['height'].dropna()]
    
    hip_test = sct.ttest_ind(df_alt_br, df_alt_usa, equal_var=False)
    return True if hip_test[1] > alpha else False
q5()


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[17]:


def q6():
    # Retorne aqui o resultado da questão 6.
    #Pegando os indivíduos que possui nacionalidade brasileira, canadense ou americana
    #e em seguida, pegando apenas os valores das suas alturas, retirando os valores nulos.
    [df_alt_br, df_alt_can, df_alt_usa] = [athletes[(athletes['nationality'] == 'BRA')]['height'].dropna(),
                                           athletes[(athletes['nationality'] == 'CAN')]['height'].dropna(),
                                           athletes[(athletes['nationality'] == 'USA')]['height'].dropna()]
    
    hip_test = sct.ttest_ind(df_alt_br, df_alt_can, equal_var=False)
    return True if hip_test[1] > alpha else False

q6()


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[31]:


def q7():
    # Retorne aqui o resultado da questão 7.
    #Pegando os indivíduos que possui nacionalidade brasileira, canadense ou americana
    #e em seguida, pegando apenas os valores das suas alturas, retirando os valores nulos.
    [df_alt_br, df_alt_can, df_alt_usa] = [athletes[(athletes['nationality'] == 'BRA')]['height'].dropna(),
                                           athletes[(athletes['nationality'] == 'CAN')]['height'].dropna(),
                                           athletes[(athletes['nationality'] == 'USA')]['height'].dropna()]
    
    hip_test = sct.ttest_ind(df_alt_usa, df_alt_can, equal_var=False)
    return float(round(hip_test[1], 8))
q7()


# In[30]:


[df_alt_br, df_alt_can, df_alt_usa] = [athletes[(athletes['nationality'] == 'BRA')]['height'].dropna(),
                                       athletes[(athletes['nationality'] == 'CAN')]['height'].dropna(),
                                       athletes[(athletes['nationality'] == 'USA')]['height'].dropna()]
    
    
print('Mediana: Brasil | Canadá | USA = ',[df_alt_br.median(), df_alt_can.median(), df_alt_usa.median()])


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?

# In[20]:


[df_alt_br, df_alt_can, df_alt_usa] = [athletes[(athletes['nationality'] == 'BRA')]['height'].dropna(),
                                         athletes[(athletes['nationality'] == 'CAN')]['height'].dropna(),
                                         athletes[(athletes['nationality'] == 'USA')]['height'].dropna()]


# In[22]:


sns.distplot(df_alt_can)


# In[23]:


sns.distplot(df_alt_usa)


# In[24]:


sns.distplot(df_alt_br)


# In[25]:


sns.distplot(df_alt_usa, label = 'USA')
sns.distplot(df_alt_can, label = 'CAN')
sns.distplot(df_alt_br,  label = 'BR')
plt.legend()
plt.show()


# Como a média da alturas dos paises são próximas, aceitamos a Hipótese que são estatisticamente verdadeiras.

# In[ ]:




