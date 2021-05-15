#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from random import *
import matplotlib.pyplot as plt
from matplotlib.dates import (DAILY,HOURLY,MINUTELY,DateFormatter,
                              rrulewrapper, RRuleLocator, drange)

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import ticker
import matplotlib.animation as animation
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR


from math import ceil
import datetime
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import time
from statsmodels.tsa.stattools import acf, pacf

cach_fit = ()
def plot_gif(df, var_previstas,var_real, start_date, end_date, min_var, max_var,min_2_var, model_fit=[],list_model=[],
 titulo = "Plot FIT",x_label="Data", figsize = (20,10), days_interval=60, minutes_interval=60, lim_y=(0.01,0.04), nome_arquivo="plot_gif.gif",
 velocidade_gif= 500,  frames = 50,interval_yticker=0.001, normalize_point=True, isFit=True, isFitHistorico=True,isLimiares=True,  yLabel="Dados", limpar_tela=True, fps=13) :
    '''
        Descrição: Função responsável por apresentar uma animação dos pontos de uma variável e suas predições.
        Será implementada com as seguintes características:

            - Possibilidade de passar uma lista de modelos, onde cada modelo irá prever um ponto para um determinado tempo na frente
            - Possibilidade de, para cada frame da animação, ter o ponto atual, suas determiandas previsões e a média(FIT desses pontos, FIT do último ponto previsto e o FIT médio dos anteriores)
            - Possibilidade de plotar a animação
            - Possiblidade de salvar em determinado formato    
            - Descrição dos FITS:
                - Um FIT com todos os pontos reais plotados (FIT Linear)
                - Um FIT com todos os pontos plotados (FIT linear_1)
                - Um FIT com todos os pontos plotados (FIT linear_1) [Com peso nas entradas]
                - Um FIT com todos pontos históricos (FIT SVR) [parametrizado]
                - Um FIT com equação da reta (y=ax+b)
                - Um FIT médio entre os 3

            - Limiares:
                - Máximo Teórico
                - Momento de Incrustação (1 mm)
                - Momento de Incrustação (2 mm)

        Parâmetros:

            df::pandas.DataFrame
                DataFrame com o conjunto de dados das entradas e saída

            var_previstas::list
                Lista com os nomes das variáveis previstas

            var_real:str
                String com o nome da TAG que foi prevista
            
            titulo::str
                Título do Gráfico
            yLabel::str
                Nome do eixo Y do gráfico

            x_label::str
                Nome do eixo X do gráfico

            list_model::list
                Lista de tuplas com cada modelo e o seu shift aprengado a saída quando foi treinado          

            start_date::str
                Início do período a ser plotado (obrigatório)

            end_date::str
                Fim do período a ser plotado (obrigatório)
            
            model_fit::tuple(modelo, color)
                Instância do modelo do FIT histórico [treinado com entradas sendo dias]
            
            min_var::float
                Limiar mínimo da variável a ser plotada e prevista

            max_var::float
                Limiar máximo da variável a ser plotada e prevista

            min_2_var::float
                Limiar entre o mínimo e o máximo da variável a ser plotada e prevista

            figsize::tuple
                Uma tupla com as dimensões do da figura

            lim_y::tuple
                Limiares para delimitar o figure do matplot
            
            nome_arquivo::str
                Nome do arquivo + path onde será salvo o arquivo

            velocidade_gif::int
                Velocidade de transição dos frames

            interval_yticker::float
                Intevalo no que será apresentado na referência do eixo y
            
            normalize_point::bool
                Controle normalizar dataframe
            
            isFit::bool
                Controle para plotar o FIT

            isFitHistorico::bool
                Controle para plotar o FitHistorico

    '''


    start_time = time.time()

    fig, ax = plt.subplots(figsize=figsize)

    print(f"Gerando GIF com {frames} frames")

    _config_plot( days_interval, minutes_interval, start_date, end_date, x_label, yLabel, titulo, lim_y ,max_var, interval_yticker, figsize) 
    df_filtrado = df[(df.index >= start_date) & (df.index <= end_date)]    

    scaler = StandardScaler()
    if(normalize_point):
        norm_x = scaler.fit_transform(df_filtrado.filter(items=var_previstas))
        df_norm = pd.DataFrame(data=norm_x, columns = var_previstas, index = df_filtrado.index)     
        df_norm[var_real] = df_filtrado[var_real]
    else:
        df_norm = df_filtrado.copy()
        #df_norm["dias"] = scaler.fit_transform( np.asarray(df_norm["dias"].values).reshape(-1, 1))

    
    index_original = df_norm.index
    if(isFitHistorico):
        predict_fit_historico = model_fit[0].predict(df_norm[model_fit[3]].values.reshape(-1, 1))
        df_norm.drop(columns=[model_fit[3]], inplace=True)

    data_range_index = pd.date_range(start=start_date, end=end_date, freq="D")
    df_norm_completo = pd.DataFrame(index=data_range_index, columns=df_norm.columns)
    df_norm_completo = df_norm_completo.combine_first(df_norm)

    df_norm_original = df_norm_completo.copy()

    if(df_norm_original.shape[0]<frames):
        raise ValueError("O número de Frames não pode ser maior que o tamanho do DataFrame filtrado.")

    for tag in df_norm_completo.columns:
        df_norm_completo[tag].interpolate(inplace=True)
    df_norm_completo.fillna(method="pad", inplace=True)

    df_norm = df_norm_completo
    
    list_y_real = []
    list_y_real_original=[]
    list_index_real = []
    list_index_original = [pd.Timestamp(index) for index in df_norm_original.index.values]
    list_index_real_original=[]
    list_index = [pd.Timestamp(index) for index in df_norm.index.values]

    def _init():
        plt.plot([],[])

    #Função para preparar o conteúdo de cada frame
    def _animate(i):
        print(f"Dia {i} de {frames}")
        #Limpando ambiente de plotagem
        if(limpar_tela):
            fig.clf()
            #Preparando configuração do ambiente de PLOT
            _config_plot(days_interval, minutes_interval, start_date, end_date, x_label, yLabel, titulo, lim_y,max_var, interval_yticker, figsize, dias_corridos=i)    

        list_legenda_limiares=[]

        if(isLimiares):
            legenda_min,legenda_min_2,legenda_max = _gerarLimiares(min_var, min_2_var, max_var)
            list_legenda_limiares.append(legenda_min)
            list_legenda_limiares.append(legenda_min_2)
            list_legenda_limiares.append(legenda_max)

        #Recuperando conteúdo index que será plotado
        index_frame = list_index[i]    
        df_frame = df_norm.loc[index_frame,:]

        #Recebendo as entradas e saídas para os modelos de previsão
        x_predict = df_frame.filter(items=var_previstas)
        y_real = df_frame[var_real]        
        list_y_real.append(y_real)
        list_index_real.append(index_frame)

        # Original
        #Recuperando conteúdo index que será plotado
        index_frame_original = list_index_original[i]    
        df_frame_original = df_norm_original.loc[index_frame_original,:]

        #Recebendo as entradas e saídas para os modelos de previsão
        x_predict_original = df_frame_original.filter(items=var_previstas)
        y_real_original = df_frame_original[var_real]        
        list_y_real_original.append(y_real_original)
        list_index_real_original.append(index_frame_original)


        #Plotando ponto real
        legend_ponto_real, = plt.plot(list_index_real_original, list_y_real_original, "o-",color="blue", label=f'{var_real} [Real]')
        #plt.plot(list_index_real, list_y_real, "o-",color="black", label=f'{var_real} [Real]')


        #Plotando FIT Histórico
        if(isFitHistorico):
            legend_fit_historico, = plt.plot(index_original,predict_fit_historico, "-", color = model_fit[1], label = f"FIT [histórico - [{model_fit[2]} dias]")

        list_y_previsto = list_y_real.copy()
        list_index_previsto = list_index_real.copy()

        list_y_previsto_original = list_y_real.copy()
        list_index_previsto_original = list_index_real.copy()
        list_legend_ponto_prevista = []
        list_legend_fit=[]

        #Gerando Pontos Previstos
        for model in list_model:   
            predict = model[0].predict([x_predict])   
            list_y_previsto.append(predict[0][0])
            index = index_frame + pd.DateOffset(days=model[1])                    
            list_index_previsto.append(index)            

            if(x_predict_original.sum()==0):
                predict_original = np.nan
            else:
                predict_original = model[0].predict([x_predict_original])[0]       
            index_original_previsto = index_frame_original + pd.DateOffset(days=model[1])                     
            legend_ponto_prevista, = plt.plot([index_original_previsto], predict_original, "go",color=model[2], label=f'{model[3]} [Previsto {model[1]}]')
            list_legend_ponto_prevista.append(legend_ponto_prevista)    
        
        list_legend_fit = _gerarPlotFit(list_index_real, list_y_real, list_index_previsto, list_y_previsto,list_index_real_original, x_predict_original,list_index_previsto_original, list_y_previsto_original, list_y_real_original,isFit,df_norm)

        #Gerando Legenda
        plt.legend(handles=[legend_ponto_real] + list_legenda_limiares + list_legend_ponto_prevista + list_legend_fit, loc='upper right')

    
    #Salvando GIF gerado
    ani = matplotlib.animation.FuncAnimation(fig, _animate, frames=frames, repeat=True, interval = velocidade_gif)
    ani.save(nome_arquivo, writer =  matplotlib.animation.PillowWriter(fps=fps), dpi=15) 

    #Marcando tempo
    print("--- %s seconds ---" % (time.time() - start_time))
   

def _gerarLimiares(min_var, min_2_var, max_var):
    #Gerando linhas de limiares
    plt.axhline(y= min_var, linewidth=3, color='darkred')
    legenda_min = mpatches.Patch(color="darkred", label="Limite Histórico Mínimo")
        
    plt.axhline(y= min_2_var, linewidth=3, color='darkorange')
    legenda_min_2 = mpatches.Patch(color="darkorange", label="Limite Histórico Mediano")

    plt.axhline(y= max_var, linewidth=3, color='g')
    legenda_max = mpatches.Patch(color='g', label="Limite Histórico Maximo")

    return (legenda_min,legenda_min_2,legenda_max )

def _gerarPlotFit(list_index_real, list_y_real, list_index_previsto, list_y_previsto,list_index_real_original, x_predict_original,list_index_previsto_original, list_y_previsto_original, list_y_real_original,isFit,df_norm):
    global cach_fit

    
    #Plotando FIT
    if(isFit):                
        x_fit_real = [x +1  for x in np.arange(len(list_index_real))]
        y_fit_real = list_y_real

        x_fit_previsto =  np.asarray([x + 1 for x in np.arange(len(list_index_previsto))], dtype=np.int32)
        y_fit_previsto = list_y_previsto   

        x_fit_previsto_original =  np.asarray([x + 1 for x in np.arange(len(list_index_previsto_original))], dtype=np.int32)
        y_fit_previsto_original = list_y_previsto_original      

        x_fit_real_original =  np.asarray([x + 1 for x in np.arange(len(list_index_real_original))], dtype=np.int32)
        y_fit_real_original = list_y_real_original      

        list_x= np.arange(len(df_norm.index))
        parcela_x =(0 if len(x_fit_real)==1 else ceil(len(x_fit_real)*0.4))
        #print(parcela_x)
        coefs_linear_reais = np.polyfit(x_fit_real,y_fit_real,1,)        
        coefs_linear_previsto = np.polyfit(x_fit_previsto,y_fit_previsto,1) 
        coefs_linear_previsto_parcela = np.polyfit(x_fit_previsto[parcela_x:len(x_fit_previsto)],y_fit_previsto[parcela_x:len(x_fit_previsto)],1) 
        coefs_linear_previsto_peso = np.polyfit(x_fit_previsto,y_fit_previsto,1, w= np.sqrt(x_fit_previsto[::-1])) 

        if(x_predict_original.sum()==0 and len(cach_fit)!=0):       
            ffit_reais = cach_fit[0]
            ffit_peso  = cach_fit[1]
            ffit = cach_fit[2]           
            fit_reta_previsto = cach_fit[3]            
            fit_svr = cach_fit[4] 
            fit_reta_previsto_parcela = cach_fit[5]     
            fit_svr_ply = cach_fit[6]
            list_x = cach_fit[7]          
        else:
            ffit_reais = np.poly1d(coefs_linear_reais)                  
            ffit_peso = np.poly1d(coefs_linear_previsto_peso)  
            ffit = np.poly1d(coefs_linear_previsto)  
            fit_reta_previsto_parcela = np.poly1d(coefs_linear_previsto_parcela)
            #FIT com Equação da Reta Reduzida [y = ax + b]
            fit_reta_previsto = [((y_fit_real_original[-1] - y_fit_real_original[0])/(x_fit_real_original[-1] - x_fit_real_original[0])) * (x - x_fit_real_original[0]) + x_fit_real_original[0] for x in list_x]
            
            svr_nu = NuSVR(kernel='linear', C=1, gamma='scale', nu=0.9)
            svr_nu_poly = NuSVR(kernel='rbf', C=1, gamma='scale', nu=0.9)
            svr_nu.fit((x_fit_previsto_original.reshape(-1,1)), y_fit_previsto_original)    
            svr_nu_poly.fit((x_fit_previsto_original.reshape(-1,1)), y_fit_previsto_original)       
            fit_svr = svr_nu.predict(list_x.reshape(-1,1))
            fit_svr_ply = svr_nu_poly.predict(list_x.reshape(-1,1))
            cach_fit = (ffit_reais,ffit_peso,ffit,fit_reta_previsto,fit_svr,fit_reta_previsto_parcela, fit_svr_ply, list_x)
            
            

       # legend_fit_real,= plt.plot(df_norm.index, ffit_reais(list_x), color="orange",  linestyle='--', label="FIT [pontos reais]")
       # legend_fit_previsto, = plt.plot(df_norm.index, ffit_peso(list_x), color="red",  linestyle='--', label= "FIT [pontos reais + último ponto previsto] PESO (SQRT)")
       # legend_fit_previsto_sem_peso, = plt.plot(df_norm.index, ffit(list_x), color="g",  linestyle='--', label= "FIT [pontos reais + último ponto previsto] Sem peso")
       # legend_fit_previsto_reta, = plt.plot(df_norm.index,fit_reta_previsto, color="chocolate",  linestyle='--', label= "FIT Equacao da Reta")
       # legend_fit_previsto_sem_peso_parcela, = plt.plot(df_norm.index, fit_reta_previsto_parcela(list_x), color="slategray",  linestyle='--', label= "FIT [pontos reais + último ponto previsto - parcela] Sem peso")

        legend_fit_previsto_svr, = plt.plot(df_norm.index,fit_svr, color="mediumvioletred",  linestyle='--', label= "FIT SVR [Linear]")
        #legend_fit_previsto_svr_poly, = plt.plot(df_norm.index,fit_svr_ply, color="red",  linestyle='--', label= "FIT SVR [Poly]")

       # list_legend_fit = [legend_fit_previsto, legend_fit_previsto_sem_peso, legend_fit_real, legend_fit_previsto_reta,legend_fit_previsto_svr,legend_fit_previsto_sem_peso_parcela]                
        list_legend_fit =  [legend_fit_previsto_svr]
        return list_legend_fit

def _config_plot( days_interval, minutes_interval, start_date, end_date, x_label, yLabel, titulo, lim_y,max_var, interval_yticker, figsize, dias_corridos = 0):
        ax = plt.gca()
        plt.xlim(start_date, end_date)
        plt.ylim(lim_y[0], lim_y[1])
        plt.xlabel(x_label,fontsize=20)
        plt.ylabel(yLabel,fontsize=20)
        plt.title(f'{titulo} [{dias_corridos} Dias]',fontsize=20)
        plt.grid(True)
    
        ax.yaxis.set_major_locator(ticker.MultipleLocator(interval_yticker))   
        
        rule = rrulewrapper(DAILY, interval=days_interval)
        minor_rule = rrulewrapper(MINUTELY, interval=minutes_interval)
        legenda_tag= []
        loc = RRuleLocator(rule)
        formatter = DateFormatter('%d-%m-%Y')
        minor_loc = RRuleLocator(minor_rule)
        minor_formatter = DateFormatter('%H:%M')        
        if(days_interval<2):
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_tick_params(which='major', rotation=90, pad=60)

            ax.xaxis.set_minor_locator(minor_loc)
            ax.xaxis.set_minor_formatter(minor_formatter)
            ax.xaxis.set_tick_params(which='minor', rotation=90)
        else:
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_tick_params(which='major', rotation=90, pad=0)

 
            
def plot_full(df, variables_array, start_date, end_date, **params):
    """
        Descrição: Função responsável por apresentar o gráfico com as seguintes características:

            - Delimitação de período a ser apresentado;
            - Escolha das variáveis;
            - Uma curva de tendência utilizando uma filtro de média móvel (personalizado);
            - Personalização das fontes;
            - Marcação de períodos com blocos personalizados;
            - Marcação de dias;
            - Plot do erro Absoluto (entre duas variaveis especificadas);
            - Salve do gráfico;
            - Salve do csv do dataframe delimitado;

        =====================================================================

        Parâmetros:

            df:
                DataFrame a ser plotado (obrigatório)

            variables_array:
                Variáveis do DataFrame a serem plotados (obrigatório)

            start_date:
                Início do período a ser plotado (obrigatório)

            end_date:
                Fim do período a ser plotado (obrigatório)

        ======================================================================

            **params (Foi implementado essa forma de parâmetro **kwargs onde terá a possibilidade de passar um dicionários com os parametros a seres escolhidos.
             Esse conjuntos de parametros são todos opcionais): 

                ylabel:
                    Descrição da label para o eixo y (Default: '')

                xlabel:  
                    Descrição da label para o eixo x (Default: "Data")

                titulo:
                    Título para o gráfico (Default: '') (É utilizado um combinação dos nomes das colunas))

                days_interval:
                    Intervalo de dias que será plotado (Default: 2)

                minutes_interval:
                    Intervalo de minutos que será plotado (Default: 180)

                save_figure_name:
                    Nome do arquivo da imagem do gráfico que será salvo (Default: '')

                error:
                    Erro entre duas variáveis (Default: False)

                abs_error
                    Erro absoluto entre duas variáveis (Default: False)

                rolling_time:
                    Janela do filtro de média móvel a ser calculada(Default: 0)

                figure_heigh:
                    Altura da figura do gráfico (Default: 10)

                figure_width:
                    Comprimento da figura do gráfico (Default: 30)

                ticks:
                    Ticks do eixo Y a serem personalizado (Default: []) 
                    Ex: [0, 1, 0.1] ==>> eixo y ==  [0. , 0.1, 0.2, ...  0.8, 0.9, 1. ]

                tendencia:
                    Plota a curva de cada variável e sua respectiva tendência (Default.: False)

                label_font_size:
                    Tamanho da fonte da label (eixos) (Default: 18)

                label_font_weight
                    Espessura da fonte do label (eixos) (Default: 'bold')

                title_font_weight:
                    Espessura da fonte do título  (Default: 'bold')

                tick_font_weight:
                    Espessura da fonte do tick  (Default: 'bold')

                tick_font_size:
                    Tamanho da fonte da tick  (Default: 14)

                legenda_font_size
                    Tamanho da fonte da legenda  (Default: 16)

                treino_true_data:
                    Janela do treino limpo para ser destacada (Ex.: [data_ini:str, data_fim:str])

                treino_true_color:
                    Cor do bloco de treino limpo (Ex.: 'green')

                treino_true_alpha:
                    Quantidade de transparencia do bloco de treino limpo (Default.: 0.3)

                treino_true_label:
                    Legenda do bloco de treino limpo (Default.: 'Período de Treino Limpo)

                treino_false_data:
                    Janelas do treino sujo para ser destacada (Ex.: [(data_ini:str, data_fim:str)])

                treino_false_color:
                    Cor do bloco de treino sujo (Ex.: 'red')

                treino_false_alpha:
                    Quantidade de transparência do bloco de treino sujo (Default.: 0.3)

                treino_false_label:
                    Legenda do bloco de treino sujo (Default.: 'Período de Treino Sujo)

                bloco_destaque_data:
                    Janelas do bloco extra para ser destacada (Ex.: [(data_ini:str, data_fim:str)])

                bloco_destaque_color:
                    Cor do bloco extra (Ex.: 'yellow')

                bloco_destaque_label:
                    Legenda do bloco extra (Default.: 'Período Extra')

                bloco_destaque_alpha:
                    Quantidade de transparencia do bloco extra (Default.: 0.3)

                dia_destaque_data:
                    Datas a serem destacada (Ex.:[data])

                dia_destaque_color:
                    Cor da data destacada (Default: 'red')

                dia_destaque_label:
                    Descrição da data destacada (Default: 'Dia Destaque')

                reta_tendencia_data:
                    Períodos das retas de tendências a serem plotadas (Ex.: [(data_ini, data_fim)])

                reta_tendencia_color:
                    Cores a serem definidas para cada reta de tendência

                reta_tendencia_label:
                    Descrição na legenda de cada reta de tendência (Default: Reta de Tendência (Tag))

                reta_tendencia_ordem:
                    Descrição da ordem do FIT de todas as linhas de tendência (Default: 1)

                interval_yticker:
                    Intervalo inteiro no qual será distribuído os valores do eixo Y 

                show_legenda:
                    Mostrar a legenda (Default:True)

                label_legenda:
                    Descrição da label para a legenda (Default: nome da variavel)
                
                linestyle:
                    Estilo da linha (Default: '-') Ex:'-' ==> ______ | ':' => ...... | '-.' ==> --.--. | '--' ==> ------
                
                marker:
                    Marcação do ponto no gráfico (Default:'o')  Ex: '.' ',' 's' '*'  ... 

                linewidth:
                    Espessura da linha (Default: 1.5)

                hotinzontal treino_true_data, treino_false_data,bloco_destaque_data:
                    adicione h na variavel --> treino_true_h_data 

    ==============================================================================

            Exemplo de uso (Exemplo passando apenas os parâmetros obrigatórios):
                params = {    
                    "treino_true_data": [(date_ini_limpo,date_fim_limpo )],
                    "treino_false_data":[(date_ini_sujo,date_fim_sujo )],
                    "bloco_destaque_data":[(date_ini_bloco,date_fim_bloco )], 
                    "dia_destaque_data":[date_ini_limpo],  
                    "reta_tendencia_data": [(data_reta_tendencia_ini, data_reta_tendencia_fim)],
                    "reta_tendencia_color": ['red']  
                
                }

                plt_variables_full(dataframe_tags, dataframe.columns, dataframe_tags.index.min(), dataframe.index.max(), **params)
                

    Obs.: o uso do **kwargs tem a possibilidade do uso de um dicionário com os parâmetros, ou também pela a forma convencional (passando cada um por vez)
    
    """
    #Recuperando parametros de **params
    ylabel = params.get("ylabel",'')
    xlabel = params.get( "xlabel","Data")
    titulo = params.get( "titulo",'')
    days_interval = params.get( "days_interval",2)
    minutes_interval = params.get("minutes_interval",180)
    save_figure_name = params.get("save_figure_name",'')
    error = params.get("error",False)
    rolling_time = params.get("rolling_time",0)
    abs_error = params.get("abs_error",False )
    figure_heigh = params.get("figure_heigh",10)
    figure_width = params.get( "figure_width",30)
    ticks = params.get( "ticks",[])
    tendencia = params.get( "tendencia",False)
    show_legenda = params.get( "show_legenda",True)
    label_legenda =  params.get("label_legenda",'')
    leg_labels = label_legenda
    
    #Recuperando parametros de estilização do gráfico
    label_font_size = params.get("label_font_size", 18)
    label_font_weight = params.get("label_font_weight", "bold")

    title_font_size = params.get("title_font_size", 20)
    title_font_weight = params.get("title_font_weight", "bold")

    tick_font_weight =  params.get("tick_font_weight", "bold")
    tick_font_size = params.get("tick_font_size", 14)

    legenda_font_size = params.get("legenda_font_size", 16)

    reta_tendencia_data = params.get("reta_tendencia_data", [])
    reta_tendencia_color = params.get("reta_tendencia_color",['darkmagenta'])
    reta_tendencia_label = params.get("reta_tendencia_label", ["Reta de Tendência"])
    reta_tendencia_ordem = params.get("reta_tendencia_ordem",[1])
    interval_yticker = params.get("interval_yticker")

    linestyle = params.get("linestyle",'-')
    marker = params.get("marker","o")
    linewidth = params.get("linewidth", 1.5)

    df_filtered = df[ (df.index > start_date) & (df.index < end_date) ]
    
    print(df_filtered.shape)
    for variable in variables_array:
        if 'MODE' in variable:
            df_filtered[variable] = df_filtered[variable].apply(lambda d: 1 if d=="CAS" else 0)

    rule = rrulewrapper(DAILY, interval=days_interval)
    minor_rule = rrulewrapper(MINUTELY, interval=minutes_interval)
    legenda_tag= []
    loc = RRuleLocator(rule)
    formatter = DateFormatter('%d-%m-%Y')
    minor_loc = RRuleLocator(minor_rule)
    minor_formatter = DateFormatter('%H:%M')

    fig, ax = plt.subplots()
    xs = {}
    series = {}
    smask = {}
    df_return = pd.DataFrame()

    if(error):
        if(label_legenda == ''):
            label_legenda = 'Erro'
        
        if(abs_error):
            error_abs = abs(df_filtered[variables_array[0]]-df_filtered[variables_array[1]])
            if(rolling_time>0):
                plot, =plt.plot_date(df_filtered.index, error_abs.rolling(rolling_time).mean(), linestyle=linestyle, marker=marker, linewidth=linewidth, label= f'{label_legenda}');
            else:
                plot, = plt.plot_date(df_filtered.index, error_abs, linestyle=linestyle, marker=marker, label="Erro");
            legenda_tag.append(plot)

        
        else:
            error_ = df_filtered[variables_array[0]]-df_filtered[variables_array[1]]
            if(rolling_time>0):
                plot_legenda, = plt.plot_date(df_filtered.index, error_.rolling(rolling_time).mean(), linestyle=linestyle, marker=marker, linewidth=linewidth, label="Erro Tendência");
                legenda_tag.append(plot_legenda)
            else:
                plot, = plt.plot_date(df_filtered.index, error_, linestyle=linestyle, marker=marker, linewidth=linewidth, label="Erro");
                legenda_tag.append(plot)

        
    else:
        k = 0
        for variable in variables_array:

            if(label_legenda == '' ):
                label_legenda = variable

            else:
                if (len(variables_array) == len(leg_labels)):
                    label_legenda = leg_labels[k]
                else:
                    if (len(variables_array) > 1 ):
                        label_legenda = variable
                    else:
                        label_legenda = leg_labels[k]
            k+=1

            if(tendencia):
                plot, = plt.plot_date(df_filtered.index, df_filtered[variable], linestyle=linestyle, marker=marker, linewidth=linewidth,label=label_legenda);
                legenda_tag.append(plot)
                if(rolling_time>0):
                    plt_tendencia, = plt.plot_date(df_filtered.index, df_filtered[variable].rolling(rolling_time).mean(), linestyle=linestyle, marker=marker, linewidth=linewidth, label=f'{label_legenda} (Tendência)');
                    legenda_tag.append(plt_tendencia)                    
            else:                            
                if(rolling_time>0):
                    plt_tendencia, = plt.plot_date(df_filtered.index, df_filtered[variable].rolling(rolling_time).mean(), linestyle=linestyle, marker=marker, linewidth=linewidth, label=f'{label_legenda}');
                    legenda_tag.append(plt_tendencia)                    
                else:
                    plot, = plt.plot_date(df_filtered.index, df_filtered[variable], linestyle=linestyle, marker=marker, linewidth=linewidth,label=label_legenda);
                    legenda_tag.append(plot)
                        


    if(len(titulo)>0):        
        plt.title(titulo, fontsize=title_font_size, fontweight= title_font_weight)
    elif(error):
        plt.title('Error', fontsize=title_font_size, fontweight=title_font_weight)
    else:
        plt.title(' - '.join(variables_array), fontsize=title_font_size, fontweight= title_font_weight)

    ax.tick_params(labelsize=tick_font_size)
    plt.rcParams['figure.titleweight'] = tick_font_weight
    plt.rcParams['font.weight'] = tick_font_weight
    
    # Adicionando  da Reta de tendência 
    legenda_reta_tendencia = []
    if(reta_tendencia_data):
        for col in variables_array:
            i=0
            for reta in reta_tendencia_data:
                ini = pd.Timestamp(reta[0])
                fim = pd.Timestamp(reta[1])

                if(ini>fim):
                    raise(ValueError (f'O o período para o fit da reda de tendência da variável {col} está com a Data Início maior que Data Fim.'))
                
                df_filtrado = df_filtered[(df_filtered.index > ini) & (df_filtered.index < fim)]            
                x = np.arange(len(df_filtrado)) 
                y = df_filtrado[col].values
                
                coefs = np.polyfit(x,y,reta_tendencia_ordem[i])
                ffit = np.poly1d(coefs) 
                
                df_filtrado['ffit'] = ffit(x)
                legenda, = plt.plot(df_filtrado.index, df_filtrado['ffit'], color=reta_tendencia_color[i],  linestyle='--', linewidth=4, label=f'{reta_tendencia_label[i]}')
                legenda_reta_tendencia.append(legenda)
                i+=1

    if(len(ticks)>2):
        ax.set_yticks(np.arange(ticks[0], ticks[1]+ticks[2], ticks[2]))

    if(interval_yticker):
        ax.yaxis.set_major_locator(ticker.MultipleLocator(interval_yticker))   

    if(days_interval<2):
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_tick_params(which='major', rotation=90, pad=60)

        ax.xaxis.set_minor_locator(minor_loc)
        ax.xaxis.set_minor_formatter(minor_formatter)
        ax.xaxis.set_tick_params(which='minor', rotation=90)
    else:
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_tick_params(which='major', rotation=90, pad=0)

    fig.set_figheight(figure_heigh)
    fig.set_figwidth(figure_width)
    plt.grid(True)

    if(len(ylabel)>0):
        plt.ylabel(ylabel, fontsize=label_font_size, fontweight = label_font_weight)
        
    if(len(xlabel)>0):
        plt.xlabel(xlabel, fontsize=label_font_size, fontweight = label_font_weight)

    #Dia Destaque
    legenda_data_line = []
    dia_destaque_data = params.get("dia_destaque_data",[])
    if(len(dia_destaque_data) > 0):
        color_dia = params.get("dia_destaque_color", 'r')
        label_dia = params.get('dia_destaque_label', "Dia Destaque")
        
        for dia in dia_destaque_data:                
            plt.axvline(x=dia, color = color_dia, lw=2)
            line_limpeza = Line2D([0], [0], color=color_dia, lw=2)
            line_limpeza.set_label(label_dia)
        legenda_data_line.append(line_limpeza)

    # Data Blocos (True, False, Bloco destaque)
    legenda_bloco = []   
    tipos_blocos = ["treino_true", "treino_false", "bloco_destaque"]
    dic_bloco = {"treino_true":["green", "Treino (limpo)"], "treino_false": ["red", "Treino (sujo)"], "bloco_destaque": ["yellow", "Bloco Destaque"]}
    
    for bloco in tipos_blocos:
        bloco_data = params.get(f"{bloco}_data")
        if(bloco_data and len(bloco_data)>0):
            color_bloco = params.get(f'{bloco}_color', dic_bloco[bloco][0])
            label_bloco = params.get(f'{bloco}_label', f'Período de {dic_bloco[bloco][1]}')
            alpha_bloco = params.get(f'{bloco}_alpha', 0.2)

            for data in bloco_data: 
                ini = pd.Timestamp(data[0])
                fim = pd.Timestamp(data[1])
                if(ini>fim):
                    raise(ValueError(f'O parametro {bloco} está com a Data Início maior que Data Fim.'))
                plt.axvspan(ini,fim, color=color_bloco, alpha=alpha_bloco)
                legenda = mpatches.Patch(color=color_bloco, label=label_bloco,alpha=alpha_bloco)
            legenda_bloco.append(legenda)
    
    
    legenda_bloco_h = [] 
    tipos_blocos = ["treino_true_h", "treino_false_h", "bloco_destaque_h"]
    dic_bloco = {"treino_true_h":["green", "Treino (limpo)"], "treino_false_h": ["red", "Treino (sujo)"], "bloco_destaque_h": ["yellow", "Bloco Destaque"]}
    
    for bloco in tipos_blocos:
        bloco_data = params.get(f"{bloco}_data")
        if(bloco_data and len(bloco_data)>0):
            color_bloco = params.get(f'{bloco}_color', dic_bloco[bloco][0])
            label_bloco = params.get(f'{bloco}_label', f'Período de {dic_bloco[bloco][1]}')
            alpha_bloco = params.get(f'{bloco}_alpha', 0.2)
            
            for data in bloco_data: 
                plt.axhspan(data['ymin'],data['ymax'], xmin=data['xmin'], xmax=data['xmax'], color=color_bloco, alpha=alpha_bloco)
                legenda = mpatches.Patch(color=color_bloco, label=label_bloco, alpha=alpha_bloco)
            legenda_bloco_h.append(legenda)

    if(show_legenda and ( len(legenda_tag)>0 or len(legenda_data_line)>0 or len(legenda_bloco)>0 or len(legenda_bloco_h)>0)):
        plt.legend(handles=legenda_tag + legenda_data_line + legenda_bloco + legenda_reta_tendencia + legenda_bloco_h, fontsize=legenda_font_size, loc='best')
    
    if(len(save_figure_name)>0):
        plt.savefig(save_figure_name, dpi=150, bbox_inches='tight')

    plt.show() 

    # Removendo estilos        
    plt.clf()
    plt.close()


def correlacao_var(df, size=(8, 8), title='', annot_size=10, save_figure_name = ''):
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True


    # Set up the matplotlib
    #  figure

    fig, ax = plt.subplots(figsize=size)
    fig.suptitle(title)
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot = True, annot_kws={"size":annot_size}, ax=ax)
    
    if(len(save_figure_name)>0):
        plt.savefig(save_figure_name, dpi=150, bbox_inches='tight')

    plt.show()

def plot_dist(df,list_tags, suptitle=''):

    plt.figure(figsize = (19,len(list_tags)*4))
    plt.suptitle(suptitle)
    i = 1
    for tag in list_tags:

        plt.subplot(len(list_tags),2,i)
        df[[tag]].boxplot()
        plt.title('Boxplot:'+str(tag))
        i+=1
        plt.subplot(len(list_tags),2,i)
        sns.distplot(df[[tag]].dropna())
        plt.title('Distribution:'+str(tag))
        i+=1
    plt.show()


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def plot_acf(serie, nlags, subplot=False, titulo=''):
    lag_acf = acf(serie, nlags=nlags)

    if subplot == False:
        plt.figure(figsize=(20,8))
        plt.autoscale(enable=True)

    plt.stem(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(serie)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(serie)),linestyle='--',color='gray')
    plt.title(f"Autocorrelation Function {titulo}")

def plot_pacf(serie, nlags, subplot=False, titulo=''):
    lag_pacf = pacf(serie, nlags=nlags, method='ols')

    if subplot == False:
        plt.figure(figsize=(20,8))
        plt.autoscale(enable=True)

    plt.stem(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(serie)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(serie)),linestyle='--',color='gray')
    plt.title(f"Partial Autocorrelation Function {titulo}")

def plot_acf_pacf(serie, nlags, titulo):
    plt.figure(figsize=(20,8))
    plt.autoscale(enable=True)
    
    plt.subplot(221)
    plot_acf(serie, nlags, True, titulo)

    plt.subplot(222)
    plot_pacf(serie, nlags, True, titulo)

def plot_box(df, cols, **params):

    #Recuperando parametros de **params
    size = params.get("size",(10, 4))
    color = params.get( "color","Paired")
    titulo = params.get( "titulo",'')
    ticker_num = params.get( "ticker_num",25)
    orientacao = params.get("orientacao","")
    save_figure_name = params.get("save_figure_name","")
 
    plt.figure(figsize=size)  
    plt.autoscale(enable=True)
    plt.title(titulo)

    ax = sns.boxplot(data=df.filter(items=cols), palette=color, orient=orientacao)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(ticker_num))
    plt.grid(True)
    
    if(len(save_figure_name)>0):
        plt.savefig(save_figure_name, dpi=150, bbox_inches='tight')

    plt.show()

