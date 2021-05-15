import pandas as pd
import numpy as np
from datetime import timedelta


def obter_correlation(df, min_threshold=-0.8, max_threshold=0.8, tipo_retorno=2):
   
    """    
        Descrição: Função para obter correlação entre colunas dataframe.

        Params:
            df: Dados de entrada.

            tipo_retorno: criterio de retoro da função. (default: 2) 

            min_threshold: Valor mínimo correlação. (default: -0.8)

            max_threshold:  Valor máximo correlação. (default: 0.8)

        return: 
            tipo_retorno = 0 >> colunas com correlação alta.
            tipo_retorno = 1 >> colunas sem correlação.
            tipo_retorno = 2 >> dataframe sem correlação.
    """

    col_corr = []
    corr_matrix = df.corr()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] <= min_threshold or corr_matrix.iloc[i, j] >= max_threshold:
                colname = corr_matrix.columns[i]
                col_corr.append(colname)
    
    df_out = df.drop(columns=set(col_corr), inplace=False)
    
    if tipo_retorno == 0:
        return set(col_corr)
    elif tipo_retorno == 1:
        return df_out.columns.tolist()
    elif tipo_retorno == 2:
        return df_out

def outlier_quartis_df(df, percentual=1.5, q1=0.25, q3=0.75):
    
    """    
        Descrição: Função mostrar os limites inferiores e superiores para caada coluna de outlier em quartis.

        Params:
            df: Dados de entrada.

            percentual: Valor multiplicado ao IQR. (default: 1.5) 

            q1: Valor do quartil 1. (default: 0.25)

            q3:  Valor do quartil 3. (default: 0.75)

        return: vazio.
    """

    for i in df.columns:
        Q1 = df[i].quantile(q=q1)
        Q3 = df[i].quantile(q=q3)
        IQR = Q3-Q1
        Min = Q1 - percentual*IQR
        Max = Q3 + percentual*IQR
        print('Min: {:.2f}  -  Max: {:.2f}  -  {}'.format(Min, Max, i))

    return 

def outlier_quartis_janela(df, tags_outlier, delta=90, percentual=1.5, q1=0.25, q3=0.75):
    
    """
        Descrição: Função responsável para retirar outlier por quartis.

        Params:
            df: Dados de entrada.

            tags_outlier: Lista de tagpara análise.

            delta: Janela de tempo. (default: 90 dias)

            percentual: Valor multiplicado ao IQR. (default: 1.5) 

            q1: Valor do quartil 1. (default: 0.25)

            q3:  Valor do quartil 3. (default: 0.75)
        
        return: 
            dataframe sem outlier por quartil.
    """
    
    df_outlier_removed = pd.DataFrame()
    
    for tag in tags_outlier:
        print(tag)

        df_jan_quartis_out = pd.DataFrame()
        maximo = df.index.max()
        ini = df.index.min()
        fim = ini + timedelta(days=delta)
        df_jan_quartis = df.filter(items=[tag])
        
        while fim < maximo:

            fim = ini + timedelta(days=delta)
            df_jan_quartis_cut = df_jan_quartis[(df_jan_quartis.index>ini) & (df_jan_quartis.index<fim)]
            
            Q1 = df_jan_quartis_cut[tag].quantile(q=q1)
            Q3 = df_jan_quartis_cut[tag].quantile(q=q3)
            IQR = Q3-Q1
            Min = Q1 - IQR*percentual
            Max = Q3 + IQR*percentual
            df_jan_quartis_cut = df_jan_quartis_cut[(df_jan_quartis_cut[tag]>Min) & (df_jan_quartis_cut[tag]<Max)]

            ini = fim

            df_jan_quartis_out = df_jan_quartis_out.append(df_jan_quartis_cut)

        df_outlier_removed = pd.concat([df_outlier_removed, df_jan_quartis_out], axis=1, sort=False)            
    
    df_outlier_removed.dropna(axis=0, inplace=True)
    df_out = df[df.index.isin(df_outlier_removed.index)]
    
    return df_out

def montar_entrada_shift(df, janelas_ini_fim, tags_shift_x, valores_freq_x, drop=False, f='D'):
   
    """
        Descrição: Função paran criar variável para dataframe com shift por janela.

        Parâmetros:

            df:
                DataFrame 

            janelas_ini_fim:
                Janelas de corte onde serão aplicadas os shift separadamente.

            tags_shift_x:
                Lista das tags em que serão aplicadas os shift. 

            drop:
                Booleano para definir se os nans serão removidos. (default: False)

            f:
                Frequência dos pontos a serem considerados. (default: D) As únicas opções são: 
                    'P' - considerando os últimos pontos
                    'D' - considerando os últimos dias

        Return:
            uma dupla >> dataframe montado e uma lista com os nomes das variáveis criadas.
    """

    nomes_variaveis = set()
    freq = f if f else 'P'
    df_out = pd.DataFrame([])

    for ini, fim in janelas_ini_fim:

        df_filter = df[(df.index >= ini) & (df.index <= fim)]

        for tag in tags_shift_x:
            for shift in valores_freq_x:

                nome_variavel = tag + ' (T_' + freq.upper() + str(shift) + ')'
                df_filter[nome_variavel] = df_filter[tag].shift(shift, freq=freq)
                nomes_variaveis.add(nome_variavel)
        
        df_out = df_out.append(df_filter)
    
    nomes_variaveis = list(nomes_variaveis)
    nomes_variaveis.sort()
    if drop:
        df_out.dropna(axis=0, inplace=True) 
    
    return df_out, nomes_variaveis

def montar_entrada_media_ponderada(df, janelas_ini_fim, tags_media_x, tamanhos_janela, label_index='Date', drop=False, freq='D'):
 
    """
        Descrição: Função para criar variável para dataframe com media ponderada por janela.

        Parâmetros:

            df:
                DataFrame 

            janelas_ini_fim:
                Janelas de corte onde serão aplicadas as médias ponderadas separadamente

            tags_media_x:
                Lista das tags em que serão aplicadas as médias ponderadas 

            tamanhos_janela:
                Lista contendo as quantidades de pontos (anteriores ao ponto atual) a serem considerados na média

            label_index:
                Nome do index do dataframe. (default: 'Date')

            drop:
                Booleano para definir se os nans serão removidos. (default: False) 

            freq:
                Frequência dos pontos a serem considerados. (default: 'D') As únicas opções são: 
                    'P' - média considerando os últimos pontos
                    'D' - média considerando os últimos dias

        Return:
            dataframe montado por média ponderada.
    """

    df_out = pd.DataFrame([])
    for ini, fim in janelas_ini_fim:

        df_filter = df[(df.index >= ini) & (df.index <= fim)]
        df_mp = df_filter.reset_index()
        
        for tag in tags_media_x:
            for tam in tamanhos_janela:
                df_mp[tag + ' (MP_' + str(freq.upper()) + str(tam) + ')'] = df_mp.apply(lambda x: media_ponderada(x, df_filter, tam, tag, freq, label_index), axis=1)        
        df_out = df_out.append(df_mp)
   
    df_out.set_index(label_index, inplace=True)
    
    if drop:
        df_out.dropna(axis=0, inplace=True)
    
    return df_out  

def media_ponderada(row, df, tam_janela, nome_tag, freq, label_index):

    """
        Descrição: Função auxilar  de 'montar_entrada_media_ponderada'.
    """

    soma_total = 0
    soma_peso = 0
    
    if (freq.lower() == 'd'):
        ini = row[label_index] - timedelta(days=tam_janela)
        fim = row[label_index] - timedelta(days=1)
        pontos = df[ini : fim][nome_tag]
        if (len(pontos) > 0):
            data_ref = pontos.index[-1]

            for ponto in pontos.iteritems():
                peso = tam_janela - (data_ref - ponto[0]).days 
                soma_total += ponto[1]*peso
                soma_peso += peso
        else:
            return None
    
    elif(freq.lower() == 'p'):
        ini = row.name - tam_janela
        ini = ini if (ini > 0) else 0
        pontos = df.iloc[ini : row.name][nome_tag]
        if (len(pontos) > 0):
            idx_ref = len(pontos)
            
            for ponto in enumerate(pontos):
                peso = tam_janela - (idx_ref - ponto[0]) + 1
                soma_total += ponto[1]*peso
                soma_peso += peso
        else:
            return None
    else:
        raise(Exception("Os únicos valores aceitáveis para a frequência são:" 
         + " 'P' - média considerando os últimos pontos;" 
         + " 'D' - média considerando os últimos dias "))  
    
    return soma_total/soma_peso     

def resample_interpolate_por_janela(df, janelas_ini_fim, freq="D", drop=True, method='linear'):
    
    """
        Descrição: Função para resample e interpolação do dataframe por janela.

        Parâmetros:

            df:
                DataFrame 

            janelas_ini_fim:
                Janelas de corte onde serão aplicadas os calculos separadamente.

            method:
                Tipo do metodo para aplicar na interpolação. (default: 'linear')

            drop:
                Booleano para definir se os nans serão removidos. (default: False) 

            freq:
                Frequência dos pontos a serem considerados. (default: 'D') As únicas opções são: 
                    'P' - média considerando os últimos pontos
                    'D' - média considerando os últimos dias

        Return:
            dataframe montado por resamble e interpolação.
    """

    df_out = pd.DataFrame([])

    for ini, fim in janelas_ini_fim:
        df_filter = df[(df.index >= ini) & (df.index <= fim)]
        df_filter = df_filter.resample(freq).interpolate(method=method)
        df_out = df_out.append(df_filter)

    if drop:
        df_out.dropna(axis=0, inplace=True)
    
    df_out.sort_index(inplace=True)

    return df_out

def rolling_time_por_janela(df, janelas_ini_fim, rolling_time, drop=True):
    
    """
        Descrição: Função para montar media móvel por janela.

        Parâmetros:

            df:
                DataFrame 

            janelas_ini_fim:
                Janelas de corte onde serão aplicadas os calculos separadamente.

            rolling_time:
                Tipo dode valores para o metodo rolling.

            drop:
                Booleano para definir se os nans serão removidos. (default: False) 

        Return:
            dataframe montado por média móvel.
    """

    df_out = pd.DataFrame([])

    for ini, fim in janelas_ini_fim:
        df_filter = df[(df.index >= ini) & (df.index <= fim)]
        df_filter = df_filter.rolling(rolling_time).mean()
        df_out = df_out.append(df_filter)

    if drop:
        df_out.dropna(axis=0, inplace=True)
    
    df_out.sort_index(inplace=True)

    return df_out

def label_dias_ascendente(df, janelas_ini_fim, nome_tag='dias'):

    """
        Descrição: Função para criar um label de contagem de dias ascendentes por janela.

        Parâmetros:

            df:
                DataFrame 

            janelas_ini_fim:
                Janelas de corte onde serão aplicadas os calculos separadamente.

            nome_tag:
                Nome do lable criado. (default: dias) 

        Return:
            dataframe montado.
    """

    df_out = pd.DataFrame([])

    for ini, fim in janelas_ini_fim:
        df_filter = df[(df.index >= ini) & (df.index <= fim)]
        if(~df_filter.empty):
            df_filter[nome_tag] = df_filter.index
            df_filter[nome_tag] = (df_filter[nome_tag] - df_filter[nome_tag][0]).dt.days
            df_out = df_out.append(df_filter)
    
    df_out.sort_index(inplace=True)

    return df_out

def label_dias_descendente(df, janelas_ini_fim, media_tempo, nome_tag='dias'):

    """
        Descrição: Função para criar um label de contagem de dias descendentes por janela.

        Parâmetros:

            df:
                DataFrame 

            janelas_ini_fim:
                Janelas de corte onde serão aplicadas os calculos separadamente.

            nome_tag:
                Nome do lable criado. (default: dias) 
            
            media_tempo:
                Valor inteiro para media de tempo para geração.

        Return:
            dataframe montado.
    """

    df_out = pd.DataFrame([])

    for ini, fim in janelas_ini_fim:
        df_filter = df[(df.index >= ini) & (df.index <= fim)]
        df_filter[nome_tag] = df_filter.index
        df_filter[nome_tag] = (df_filter[nome_tag].max() - df_filter[nome_tag]).dt.days
        
        delta_dias = (pd.Timestamp(fim) - pd.Timestamp(ini)).days
        
        if (delta_dias < 0.9*media_tempo):
            df_filter[nome_tag] = df_filter[nome_tag] + (media_tempo - delta_dias)

        df_out = df_out.append(df_filter)
    
    df_out.sort_index(inplace=True)

    return df_out
