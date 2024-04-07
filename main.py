
global ruta_PDF
global emergente
emergente = None


##########################################################
###                 SECCION DE LIBRERIAS               ###
##########################################################

# Aqui se realiza la importacion de las librerias

#   Back

import os                                   # libreria para el manejo del sistema
import shutil                               # libreria para el control de archivos
import cv2                                  # libreria para el manejo de las imágenes
from pdf2image import convert_from_path     # libreria de conversion de archivos
import matplotlib.pyplot as plt             # libreria para visualizar imagen
import numpy as np                          # Libreria para manipular vectores de la señal
import time                                 # Libreria para control de tiempo
import sys                                  # Libreria para puntos de ruptura
from scipy.signal import savgol_filter      # Libreria para suavizar la señal muestreada

#   Front
from tkinter import *
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pandas as pd


###########################################################
###         PREPARACION DE LAS RUTAS DE DATOS           ###
###########################################################


# Aqui se ajustan las variables de rutas para el proyecto
# además de crear y validar los directorios necesarios

def Preparacion(): 
    """
    Funcion que se encarga de ajustar el proyecto,
    verificando las rutas relativas y eliminando
    los resultados de un examen previo.
    Al ejecutar se genera la creacion de la ruta relativa ""
    """
    if os.path.exists("./Code/Data/Output"):
        shutil.rmtree("./Code/Data/Output")
        os.mkdir("./Code/Data/Output")
    else:
        os.mkdir("./Code/Data/Output")


###########################################################
####         CUERPO DE FUNCIONES DEL CÓDIGO             ###
###########################################################

# Función de conversión de PDF a JPEG
def PDF_to_jpeg(ruta_PDF):
    """
    Funcion encargada de realizar la conversion de cada hoja del archivo PDF
    en una serie de imagenes que posteriormente son almacenadas en la ruta
    "./Code/Data/Output"
    Args:
        ruta_PDF (path): Ruta del archivo PDF a procesar (Electrocardiograma de Kardia)
    """
    global pages
    poppler_path = os.path.abspath("./Code/lib/poppler-22.04.0/Library/bin")            # Configuracion de rutas de los archivos
    pdf_path = os.path.abspath(ruta_PDF)                                            # Configuracion de rutas de los archivos
    saving_folder = os.path.abspath("./Code/Data/Output")                                # Configuracion de rutas de los archivos
    pages = convert_from_path(pdf_path = pdf_path, poppler_path = poppler_path )    # Obtención de la cantidad de páginas del documento
    i = 1
    for page in pages:
        img_name = f"img-{i}.jpeg"
        page.save(os.path.join(saving_folder,img_name),"JPEG")                      # Conversion de las páginas a PDF
        i += 1    
# Función de recorte de las imágenes
def crop_image():
    """
    La funcion se encarga de recortar cada una de las imagenes 
    producto de la conversion del PDF, estas imagenes son
    recortadas mediante valores fijos correspondientes con 
    los vertices del recuadro.
    """
    global emergente
    for i in range (2,6,1):
        image="img-"+ str(i) + ".jpeg"
        rimage= os.path.abspath("./Code/Data/Output/"+image)                                 # Construción de la ruta de imagen que se desea leer
        try: 
            data = cv2.imread(rimage)                                                   # Lectura de la imagen original
            data = data [204:2091,63:1638]                                              # Recorte de la imágen
            cv2.imwrite('./Code/Data/Output/img-'+ str(i)+'-crop.jpeg',data)                 # Almacenaje de las immágenes recortadas
        except (TypeError):
            pass

# Función para conectar las imágenes recortadas
def Concat_image():
    """
    Mediante esta funcion se realiza la concatenacion
    horizontal de las imagenes, donde el producto final
    es una única imagen que contiene la plenitud del 
    electrocardiograma.
    """
    im2 = cv2.imread("./Code/Data/Output/img-2-crop.jpeg")    
    im3 = cv2.imread("./Code/Data/Output/img-3-crop.jpeg")
    im4 = cv2.imread("./Code/Data/Output/img-4-crop.jpeg")
    im5 = cv2.imread("./Code/Data/Output/img-5-crop.jpeg")         
    imga = cv2.hconcat([im2, im3])
    imgb = cv2.hconcat([im4, im5])
    img = cv2.hconcat([imga, imgb])
    img = img[:,97:6002]
    cv2.imwrite('./Code/Data/Output/img-ECG.jpeg',img)

# Función para limpiar la imágen
def Limpiar_imagen():
    """
    Mediante este funcion se realiza un barrido de la imagen
    a la par con una mascara. De la mano con este proceso se
    realiza un filtrado y binarizado de la imagen.
    Estos procesos se hacen para eliminar textos presentes en
    la imagen y posibles ruidos.
    """
    img = cv2.imread("./Code/Data/Output/img-ECG.jpeg")
   #########################################################################
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,img = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
   #########################################################################
    mask = cv2.imread("./Code/lib/mask.png",0)
    for f in range(img.shape[0]):
        for c in range(img.shape[1]):
            if (int(mask[f,c]) == 0): 
                img[f,c] = 255
                 
    ########################################################################  
    cv2.imwrite('./Code/Data/Output/img-ECG.jpeg',img)
    
#Detecto los ejes y secciono la imágen
def Lines_Hough(img): 
    """
    Mediante esta funcion se estable la posible existencia de
    los ejes de la imagen, esta lectura no es del todo precisa,
    por esta razon se hace un proceso de aproximacion al valor 
    principal del eje. en base a estos valores se hace un recorte 
    para la separacion de cada derivacion cardiaca con un ancho
    determinado de 300 px.
    Estas señales son almacenadas en la ruta "./Code/Data/Output/"

    Args:
        img (array): Imagen leida y en forma de numpy array de 2D
    """
    global Ejesy
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=500, maxLineGap=50)
    Ejesx=[]
    Ejesy=[]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if (x1 != x2):
            Ejesx.append(y1)
            cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 1, cv2.LINE_AA)
        else: 
            Ejesy.append(x1)
            cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 1, cv2.LINE_AA)
    #Determino el valor de los ejes de cada derivación
    temp=sorted(Ejesx)
    Ejesx=[]
    aux=[]
    x=0
    i=0
    while (i+x)<len(temp)-1:
        while (int(temp[x+i])-int(temp[i]))<=10 and (i+x)<len(temp)-1:
            aux.append(int(temp[x+i]))
            x=x+1
        i=x+i
        x=0
        Ejesx.append(round((sum(aux)/len(aux))))
        aux=[]
    temp=sorted(Ejesy)
    Ejesy=[]
    aux=[]
    x=0
    i=0
    while (i+x)<len(temp)-1:
        while (int(temp[x+i])-int(temp[i]))<=10 and (i+x)<len(temp)-1:
            aux.append(int(temp[x+i]))
            x=x+1
        i=x+i
        x=0
        Ejesy.append(round((sum(aux)/len(aux))))
        aux=[]
    img =  cv2.imread("./Code/Data/Output/img-ECG.jpeg")
    cv2.imwrite('./Code/Data/Output/I.jpeg',img[Ejesx[0]-150:Ejesx[0]+150,:])
    cv2.imwrite('./Code/Data/Output/II.jpeg',img[Ejesx[1]-150:Ejesx[1]+150,:])
    cv2.imwrite('./Code/Data/Output/III.jpeg',img[Ejesx[2]-150:Ejesx[2]+150,:])
    cv2.imwrite('./Code/Data/Output/aVR.jpeg',img[Ejesx[3]-150:Ejesx[3]+150,:])
    cv2.imwrite('./Code/Data/Output/aVL.jpeg',img[Ejesx[4]-150:Ejesx[4]+150,:])
    cv2.imwrite('./Code/Data/Output/aVF.jpeg',img[Ejesx[5]-150:Ejesx[5]+150,:])


###########################################################
####          CONSIDERACIONES DE RESOLUCIÓN            ####  
##################                   ######################
# TIEMPO                                                  #
#                                                         #
# ------------  1 Cuadro -> 5 mm -> 0.2 Seg --------------#
#                                                         #
# VOLTAJE                                                 #
# ------------  1 Cuadro -> 5 mm -> 0.5 mV ---------------#
#                                                         #
# DIMENSIONES                                             #
#                                                         #
# ------------  1 Cuadro -> 39px X 39px  ---------------- #
# ---------  1 px Vertical -> 12.82051282 uV ------------ #    
# --------  1 px Horizontal -> 5.128205128 mS  ---------- #      
###########################################################
    

# Vectorización de la Señal
def Vectorizacion_Señal():    
    global aVF, aVL, aVR, I, II, III, Vtiempo
    global fallaCorte
    fallaCorte = False
    Signals = ['aVF','aVL','aVR','I','II','III']
    for signalstr in Signals:
        strtemp = './Code/Data/Output/' + signalstr + '.jpeg'
        img = cv2.imread(strtemp)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _,img = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
        aux = np.asarray(img)
        aux1 = np.sum(aux, axis=0)
        posceros = np.where(aux1==76245)
        aux2 = aux[:,posceros[0][0]]
        Ejex = np.where(aux2==0)
        Ejesx = Ejex[0][0]
        aux2 = aux[0,:]
        posceros = np.where(aux2==0)
        Ejesy = posceros[0][:]
        if len(Ejesy) == 30 :
            for i in range(len(aux1)): 
                if aux1[i] < 76245:
                    img[Ejesx, i] = 255
            for i in range(img.shape[0]):
                for ejey in Ejesy:
                    temp = (((img[i,(ejey+1)]==255) or (img[i,(ejey-1)]==255)) and i!=Ejesx)
                    if temp:
                        img[i,ejey]=255           
            signal=[]
            for i in range(img.shape[1]):
                temp=aux[0:img.shape[0], i]
                ceros =  np.where(temp==0)
                ceros = np.asanyarray(ceros[0])
                if len(ceros) == 0:
                    val = -10000
                    signal.append(val)
                else:
                    temp = np.where(abs(ceros - Ejesx)==np.amax(abs(ceros - Ejesx)))
                    if len(temp[0][:])==1:
                        signal.append(ceros[temp][0])
                    else: 
                        temp = ceros[temp]
                        try:
                            postemp=np.where(abs(temp-signal[-1])==np.amax(abs(temp-signal[-1])))
                            signal.append(temp[postemp][0])
                        except IndexError:
                            fallaCorte = True
            signal = np.array(signal)
            huecos = np.where(signal==-10000)
            for hueco in huecos:
                signal[hueco] = ((signal[hueco + 1]) + (signal[hueco - 1]))/2
            signal =(signal - Ejesx)*-0.000012820512819999999
            signal = savgol_filter(signal,7,2)
            if signalstr == 'aVF':
                aVF = signal
            elif signalstr == 'aVR':
                aVR = signal
            elif signalstr == 'aVL':
                aVL = signal
            elif signalstr == 'I':
                I = signal
            elif signalstr == 'II':
                II = signal
            else: 
                III = signal
            Vtiempo = (np.arange(0,int(signal.shape[0]),1))*0.005128205128000001         
        else:
            fallaCorte = True
        
#Caracterización de la señal
def ObtencionPico():
    global aVF, aVL, aVR, I, II, III, Vtiempo
    global SegmentoSeñal, SegmentoTiempo
    global Maximos, Periodo
    global FrecuenciaCardiaca
    
    if (len(np.where(I>0.0015)[0])>0): 
        print('Descarte la señal')
        print('La señal presenta mucho ruido y picos demasiado altos')
        sys.exit()
    else:
        Maximos = np.where(I>0.0005)[0]
        if len(Maximos) == 0:
            print('Desarte la señal, la onda R se encuentra bajo el estandar de 0.5 mV')
            sys.exit()
        else:
            maximos=np.asarray(Maximos[:])
            if len(maximos) < 150:
                print('Error en la detección')
                print('La onda R no se detecta como marcador')
                sys.exit()
            else:
                temp = []
                aux = [] 
                Maximos = []
                for i in range(len(maximos)): 
                    if i >=1 : 
                        ja = maximos[i] - maximos[i-1]
                        if (maximos[i] - maximos[i-1]) <= 20 : 
                            temp.append(maximos[i])
                        else: 
                            aux = np.asarray(I[temp][:])


                            indice = np.where(aux == indice)[:][0][0]
                            Maximos.append(temp[indice])
                            aux = []
                            temp = [] 
                    else: 
                        temp.append(maximos[i])   
                
                temp = []
                for i in range(len(Maximos)):
                    if i >= 1:
                        temp.append(Maximos[i] - Maximos[i-1])
                Periodo = round(np.mean(temp))
                FrecuenciaCardiaca = len(Maximos)*2    
                Picos = I[Maximos]-0.0005
                indice = [np.where(Picos==np.min(Picos))][:][0][0][0]
                indice = Maximos[indice]
                Pico = I[indice]
                Time = Vtiempo[indice]       
                
                SegmentoSeñal = I[indice-round(Periodo/2):indice+round(Periodo/2)]
                SegmentoTiempo = Vtiempo[indice-round(Periodo/2):indice+round(Periodo/2)]
   
#Obtener características del pico
def Características():
    """
    La función aplica un filtro Savitzky-Golay para suavizar el segmento de señal.
    Calcula la primera y segunda derivadas de la señal suavizada.
    La función identifica picos y valles en la primera derivada para determinar los
    puntos temporales de varias características de la señal.
    Trata los casos en los que se encuentran varios puntos temporales
    para una característica y comprueba si hay divergencias.
    La función calcula las amplitudes de las características encontrando los valores
    correspondientes de la señal en los puntos temporales identificados.
    """
    global SegmentoSeñal, SegmentoTiempo, Periodo
    global VdatoTiempo, VdatoAmplitud, VdatoString
    global divergencias
    
    divergencias = []
    SegmentoSeñal = savgol_filter(SegmentoSeñal,10,5)  # Filtro la señal 
    
    d1x = np.diff(SegmentoSeñal)        # Extraigo las derivadas
    d2x = np.diff(d1x) 
    
    
    ####################################################################################
    ####            CALCULO LAS ONDAS POR OBSERVACIONES EN LAS DERIVADAS            ####
    ####################################################################################
    
            
    picos_indices = np.where((np.diff(np.sign(d1x)) < 0) & (d1x[:-1] > 0))[0] + 1           
    CaD = SegmentoTiempo[picos_indices]                     
    picos_indices = np.where((np.diff(np.sign(d1x)) > 0) & (d1x[:-1] < 0))[0] + 1
    DaC = SegmentoTiempo[picos_indices]  
    picos_indices = np.where(np.diff(np.sign(d1x)))[0] + 1
    Picos = SegmentoTiempo[picos_indices]
        
    TiempoR = CaD[np.where(CaD == SegmentoTiempo[np.where(SegmentoSeñal == np.max(SegmentoSeñal))])][:][0] 
        
    TiempoQa = DaC[np.where(DaC == SegmentoTiempo[np.where(SegmentoSeñal == np.min(SegmentoSeñal[0:np.where(SegmentoTiempo == TiempoR)[:][0][0]]))])][:][0]
    TiempoQb = Picos[np.where(Picos == TiempoR)[:][0][0]-1]
    if TiempoQa == TiempoQb: 
        TiempoQ = TiempoQa
    else: 
        divergencias.append('Q')
        if ((TiempoR - TiempoQa > 0.13) and (TiempoR - TiempoQb < 0.13)) :
            TiempoQ = TiempoQb
        elif ((TiempoR - TiempoQa < 0.13) and (TiempoR - TiempoQb > 0.13)):   
            TiempoQ = TiempoQa
        elif SegmentoSeñal[np.where(SegmentoTiempo ==TiempoQa)[:][0][0]] >= SegmentoSeñal[np.where(SegmentoTiempo ==TiempoQb)[:][0][0]]:
            TiempoQ = TiempoQb
        else:
            TiempoQ = TiempoQa
            
    TiempoSa = DaC[np.where(DaC == SegmentoTiempo[np.where(SegmentoSeñal == np.min(SegmentoSeñal[np.where(SegmentoTiempo == TiempoR)[:][0][0]:]))])][:][0]
    TiempoSb = Picos[np.where(Picos == TiempoR)[:][0][0]+1]
    if TiempoSa == TiempoSb: 
        TiempoS = TiempoSa
    else: 
        divergencias.append('S')
        if ((TiempoSa - TiempoR > 0.13) and (TiempoSb - TiempoR < 0.13)) :
            TiempoS = TiempoSb
        elif ((TiempoSa - TiempoR < 0.13) and (TiempoSb - TiempoR > 0.13)) :
            TiempoS = TiempoSa
        elif SegmentoSeñal[np.where(SegmentoTiempo ==TiempoSa)[:][0][0]] >= SegmentoSeñal[np.where(SegmentoTiempo ==TiempoSb)[:][0][0]]:
            TiempoS = TiempoSb
        else:
            TiempoS = TiempoSa            
            
    TiempoPa = CaD[np.where(CaD == SegmentoTiempo[np.where(SegmentoSeñal == np.max(SegmentoSeñal[:np.where(SegmentoTiempo == TiempoQ)[:][0][0]]))])][:][0]
    TiempoPb = Picos[np.where(Picos == TiempoQ)[:][0][0]-1]
    if TiempoPa == TiempoPb: 
        TiempoP = TiempoPa
    else: 
        divergencias.append('P')
        if SegmentoSeñal[np.where(SegmentoTiempo ==TiempoPa)[:][0][0]] >= SegmentoSeñal[np.where(SegmentoTiempo ==TiempoPb)[:][0][0]]:
            TiempoP = TiempoPa
        else:
            TiempoP = TiempoPb
        
    TiempoTa = CaD[np.where(CaD == SegmentoTiempo[np.where(SegmentoSeñal == np.max(SegmentoSeñal[np.where(SegmentoTiempo == TiempoS)[:][0][0]:]))])][:][0]   
    TiempoTb = Picos[np.where(Picos == TiempoS)[:][0][0]+1]
    if TiempoTa == TiempoTb: 
        TiempoT = TiempoTa
    else: 
        divergencias.append('T')
        if SegmentoSeñal[np.where(SegmentoTiempo ==TiempoTa)[:][0][0]] >= SegmentoSeñal[np.where(SegmentoTiempo ==TiempoTb)[:][0][0]]:
            TiempoT = TiempoTa
        else:
            TiempoT = TiempoTb 
    
    xD = d2x[np.where(SegmentoTiempo == Picos[np.where(Picos == TiempoP)[:][0][0]-1])[:][0][0] : np.where(SegmentoTiempo == TiempoP)[:][0][0]]
    aux = SegmentoTiempo[np.where(SegmentoTiempo == Picos[np.where(Picos == TiempoP)[:][0][0]-1])[:][0][0] : np.where(SegmentoTiempo == TiempoP)[:][0][0]]
    TiempoPI =  aux[np.where(xD == np.max(xD))[:][0][0]]
    aux = SegmentoTiempo[np.where(SegmentoTiempo == TiempoP)[:][0][0] : np.where(SegmentoTiempo == TiempoQ)[:][0][0]]
    TiempoPF = aux[int(len(aux)/2)]
 
    aux = SegmentoTiempo[np.where(SegmentoTiempo == Picos[np.where(Picos == TiempoT)[:][0][0]-1])[:][0][0] : np.where(SegmentoTiempo == TiempoT)[:][0][0]]
    TiempoTI = aux[int(len(aux)/2)]
    aux = SegmentoTiempo[np.where(SegmentoTiempo == TiempoT)[:][0][0] : np.where(SegmentoTiempo == Picos[np.where(Picos == TiempoT)[:][0][0]+1])[:][0][0] ] 
    TiempoTF = aux[int(len(aux)/2)]
       
    R = SegmentoSeñal[np.where(SegmentoTiempo == TiempoR)][:][0]
    S = SegmentoSeñal[np.where(SegmentoTiempo == TiempoS)][:][0]
    T = SegmentoSeñal[np.where(SegmentoTiempo == TiempoT)][:][0]      
    Q = SegmentoSeñal[np.where(SegmentoTiempo == TiempoQ)][:][0]
    P = SegmentoSeñal[np.where(SegmentoTiempo == TiempoP)][:][0]
    PI = SegmentoSeñal[np.where(SegmentoTiempo == TiempoPI)][:][0]
    PF = SegmentoSeñal[np.where(SegmentoTiempo == TiempoPF)][:][0]
    TI = SegmentoSeñal[np.where(SegmentoTiempo == TiempoTI)][:][0]
    TF = SegmentoSeñal[np.where(SegmentoTiempo == TiempoTF)][:][0]
    
    VdatoTiempo = [TiempoPI, TiempoP, TiempoPF, TiempoQ, TiempoR, TiempoS, TiempoTI, TiempoT, TiempoTF]
    VdatoAmplitud = [PI, P, PF, Q, R, S, TI, T, TF]
    VdatoString = ['PI', 'P', 'PR', 'Q', 'R', 'S', 'TI', 'T', 'TF']

# Obtencion de los segmentos y datos específicos
def ObtencionParametros():
    """
    La función ObtencionParametros calcula los valores de
    diferentes segmentos de una señal cardiaca basándose
    en los valores de entrada proporcionados.

    Calcule la duración del segmento RR multiplicando el período medio por un valor constante.
    Calcule la duración del segmento PR restando el intervalo de tiempo de los segmentos primero y cuarto.
    Calcular la duración del segmento QRS restando el intervalo de tiempo de los segmentos tercero y quinto.
    Calcular la duración del segmento QT dividiendo la diferencia entre los intervalos de tiempo octavo y
    cuarto por la raíz cuadrada de la duración del segmento RR.
    Calcular la duración del segmento ST restando el intervalo de tiempo de los segmentos sexto y quinto.
    """
    global VdatoAmplitud, VdatoTiempo, VdatoString, Periodo
    global VdataSegmento, VdatoSegmento
    
    SegmentoRR = Periodo*0.005128205128000001 
    SegmentoPR = VdatoTiempo[3] - VdatoTiempo[0]    
    SegmentoQRS = VdatoTiempo[5] - VdatoTiempo[3]
    SegmentoQT = ((VdatoTiempo[8] - VdatoTiempo[3])/np.sqrt(SegmentoRR))
    SegmentoST = VdatoTiempo[6] - VdatoTiempo[5]
    
    VdatoSegmento = ['RR', 'PR', 'QRS', 'QT', 'ST']
    VdataSegmento = [SegmentoRR, SegmentoPR, SegmentoQRS, SegmentoQT, SegmentoST]


##########################################################
###        DESARROLLO DE LA INTERFAZ DE USUARIO        ###
##########################################################
# Parametros de personalizacion
fondo = "#FFF"
fuente = "PT Sans"
ancho = 900
alto = 700
color_letra_titulo = "#8F141B"
color_letra_texto = "#000"

# Declaracion de la ventana
ventana = Tk()

#Configuracion para pantalla centrada y dimensiones de la ventana
ancho_ventana = ventana.winfo_screenwidth() // 2 - ancho // 2
alto_ventana = ventana.winfo_screenheight() // 2 - alto // 2
posicion = str(ancho) + "x" + str(alto) + "+" + str(ancho_ventana) + "+" + str(alto_ventana)
ventana.geometry(posicion)

#Personalizacion de la ventana
ventana.resizable(0,0)                                                                          # No puede maximizarse
icono = PhotoImage(file = "./Code/lib/usco.png")                                                           # Icono de la ventana
ventana.iconphoto(True, icono)
ventana.title("Interfaz grafica de digitalización y detección de enfermedades cardíacas")       # Titulo de la ventana
ventana.config(bg = fondo)

frameMayor = Frame(ventana, bg = fondo)
frameMayor.pack(fill = 'both', expand = True)

imagenUsco = ImageTk.PhotoImage(Image.open("./Code/lib/universidad-surcolombiana.png"))
archivo = None

def inicio():
    Preparacion()
    for widgets in frameMayor.winfo_children():
        widgets.destroy()
    global inicioFrame
    inicioFrame = Frame(frameMayor, bg = "#F2F2F2")
    inicioFrame.pack(fill = 'both', expand = 1)

    frameHeader = Frame(inicioFrame, bg = "#F2F2F2")
    frameMid = Frame(inicioFrame, bg = "#F2F2F2")
    frameFooter = Frame(inicioFrame, bg = "#F2F2F2")

    # Declaracion de etiquetas
    label1 = Label(frameHeader, text = "Interfaz de digitalización y detección de enfermedades cardíacas", fg = color_letra_titulo, bg = "#F2F2F2", font = (fuente, 20, "bold"))
    label2 = Label(frameMid, text = "Byron Hernando Galindo Suárez - 20171155352", fg = "black", bg = "#F2F2F2", font = (fuente, 17, "bold"))
    label3 = Label(frameMid, text = "Juan Esteban Narváez Carvajal - 20171159625", fg = "black", bg = "#F2F2F2", font = (fuente, 17, "bold"))
    label4 = Label(frameMid, image = imagenUsco, bg ="#F2F2F2")

    # Declaracion de botones
    btnContinuar = Button(frameFooter, text = "Continuar", font = (fuente, 20, "bold"), command = datos, width = 20, height = 20)
    
    # Posicionamiento de frames
    frameHeader.pack(side = "top", fill = "both")
    frameMid.pack(fill = "both", expand = True)
    frameFooter.pack(side = "bottom", fill = "both")

    # Posicionamiento de etiquetas
    label1.pack(pady = 100)
    label2.pack()
    label3.pack()
    label4.pack(pady = 75)

    # Posicionamiento de botones
    btnContinuar.pack(pady = 50)

def datos():
    for widgets in frameMayor.winfo_children():
        widgets.destroy()
    global datosFrame
    datosFrame = Frame(frameMayor, bg = "#F2F2F2")
    datosFrame.pack(fill = "both", expand = 1)

    

    frameHeader = Frame(datosFrame, bg = "#F2F2F2")
    frameMid = Frame(datosFrame, bg = "#F2F2F2")
    frameFooter = Frame(datosFrame, bg = "#F2F2F2")

    # Posicionamiento de frames
    frameHeader.pack(side = "top", fill = "both")
    frameMid.pack(fill = "both", expand = True)
    frameFooter.pack(side = "bottom", fill = "both")

    lbl7 = Label(frameMid, text = "Archivo: ", fg = "black", bg = "#F2F2F2", font = (fuente, 10, "bold"))

    def abrirArchivo():
        btnProcesar['state'] = DISABLED
        lbl7['text'] = "Archivo: "
        global archivo, pages
        try:
            archivo = filedialog.askopenfile(title = "Abrir", initialdir = os.path.abspath(os.getcwd()), filetypes = (("Formato PDF", "*.pdf"),))
            if archivo.name != None :
                PDF_to_jpeg(archivo.name)
                lbl7['text'] = lbl7['text'] + str(archivo.name)
                ventana.imgtk = ImageTk.PhotoImage((Image.open("./Code/Data/Output/img-1.jpeg")).resize((250,300)))
                labelvista.configure(image=ventana.imgtk)
                labelvista.pack()
                if len(pages)==5:
                    btnProcesar['state'] = NORMAL
                else:
                    emergente = tk.Toplevel(ventana)
                    emergente.title('File Error!!!')
                    emergente.geometry(str('290x180+' + str(ventana.winfo_screenwidth()//2 - 290 // 2) + "+" + str(ventana.winfo_screenheight() // 2 - 180 // 2)))
                    emergente.configure(background = 'white')
                    emergente.attributes("-toolwindow", True)
                    emergente.resizable(False,False)
                    emergente.protocol("WM_DELETE_WINDOW", lambda: None)
                    etiqueta = tk.Label(emergente, text = "Error!!!")
                    etiqueta.pack( anchor = CENTER)
                    etiqueta.config(font = (fuente, 10, "bold"), fg = 'red', bg ='white')
                    labelAlerta = tk.Label(emergente, bg = fondo, image = None)
                    ventana.alerta = ImageTk.PhotoImage((Image.open("./Code/lib/alerta.png")))
                    labelAlerta.configure(image=ventana.alerta)
                    labelAlerta.pack()
                    etiqueta2 = tk.Label(emergente, text = "El archivo seleccionado no cumple con las \ncaracterísticas de un ECG de Kardia")
                    etiqueta2.pack(anchor = CENTER)
                    etiqueta2.configure( font = (fuente, 10, "bold"), bg ='white')
                    btn_aceptar = tk.Button(emergente, text = "Aceptar",  command = emergente.destroy,  font = (fuente, 10, "bold"))
                    btn_aceptar.pack(pady = 10)
                    btnProcesar['state'] = DISABLED 
        except AttributeError:
            pass
    
    lbl5 = Label(frameHeader, text = "Por favor, cargue el archivo en formato PDF\n del electrocardiograma entregado por Kardia 6L", fg = "black", bg = "#F2F2F2", font = (fuente, 12))

    btnCargar = Button(frameHeader, text = "Cargar archivo", font = (fuente, 15, "bold"), command = abrirArchivo)
    lbl6 = Label(frameMid, text = "Previsualizacion del pdf cargado", fg = "black", bg = "#F2F2F2", font = (fuente, 15, "bold"))
    frameVisualizacion = Frame(frameMid, bg = "white", width = 250, height = 300, highlightbackground = "black", highlightthickness = 2)
    labelvista = Label(frameVisualizacion, bg = "#F2F2F2", image = None)
    btnVolver = Button(frameFooter, text = "Volver", font = (fuente, 20, "bold"), command = inicio)
    btnProcesar = Button(frameFooter, text = "Procesar", font = (fuente, 20, "bold"), command = resultados, state = DISABLED)

    lbl5.pack(side = "left", padx = 75, pady = 75)
    btnCargar.pack(side = "right", padx = 75, pady = 75)
    lbl6.pack()
    frameVisualizacion.pack(pady = 10)
    lbl7.pack()
    btnVolver.pack(side = "left", padx = 75, pady = 25)
    btnProcesar.pack(side = "right", padx = 75, pady = 25)
    
def resultados():
    global fallaCorte, ancho_ventana, alto_ventana, divergencias
    global resultadosFrame, SignalPlot, Clicks, Capsulasegmento, Capsulapico
    global I, II, III, aVR, aVL, aVF, Vtiempo, SegmentoTiempo, SegmentoSeñal
    global Inicio_pico_Seg, Fin_pico_Seg
    global Maximos, Periodo
    global FrecuenciaCardiaca
    
    for widgets in frameMayor.winfo_children():
        widgets.destroy()
    crop_image()
    Concat_image() 
    Limpiar_imagen()
    Lines_Hough(cv2.imread('./Code/Data/Output/img-ECG.jpeg'))
    Vectorizacion_Señal()
    if fallaCorte == True:
        emergente = tk.Toplevel(ventana)
        emergente.title('Process Error!!!')
        emergente.geometry(str('390x250+' + str(ventana.winfo_screenwidth()//2 - 390 // 2) + "+" + str(ventana.winfo_screenheight() // 2 - 250 // 2)))
        emergente.configure(background = 'white')
        emergente.attributes("-toolwindow", True)
        emergente.resizable(False,False)
        emergente.protocol("WM_DELETE_WINDOW", lambda: None)
        etiqueta = tk.Label(emergente, text = "Error")
        etiqueta.pack( anchor = CENTER)
        etiqueta.config(font = (fuente, 10, "bold"), fg = 'red', bg ='white')
        labelAlerta = tk.Label(emergente, bg = fondo, image = None)
        ventana.alerta = ImageTk.PhotoImage((Image.open("./Code/lib/alerta2.png")).resize((80,80), resample=Image.LANCZOS))
        labelAlerta.configure(image=ventana.alerta)
        labelAlerta.pack()
        etiqueta2 = tk.Label(emergente, text = "La señal detectada fue cortada durante la digitalización, \nla señal puede tener mucho ruido o \n es una señal con muy poca amplitud. \nEl procesamiento no es válido \nReintente con un nuevo archivo!")
        etiqueta2.pack(anchor = CENTER)
        etiqueta2.configure( font = (fuente, 10, "bold"), bg ='white')
        def alfa():
            emergente.destroy()
            datos() 
        btn_aceptar = tk.Button(emergente, text = "Aceptar",  command = alfa,  font = (fuente, 10, "bold"))
        btn_aceptar.pack(pady = 10)
    else:
        if (len(np.where(I>0.0015)[0])>0): 
            emergente = tk.Toplevel(ventana)
            emergente.title('Process Error!!!')
            emergente.geometry(str('310x210+' + str(ventana.winfo_screenwidth()//2 - 310 // 2) + "+" + str(ventana.winfo_screenheight() // 2 - 210 // 2)))
            emergente.configure(background = 'white')
            emergente.attributes("-toolwindow", True)
            emergente.resizable(False,False)
            emergente.protocol("WM_DELETE_WINDOW", lambda: None)
            etiqueta = tk.Label(emergente, text = "Error")
            etiqueta.pack( anchor = CENTER)
            etiqueta.config(font = (fuente, 10, "bold"), fg = 'red', bg ='white')
            labelAlerta = tk.Label(emergente, bg = fondo, image = None)
            ventana.alerta = ImageTk.PhotoImage((Image.open("./Code/lib/alerta2.png")).resize((80,80), resample=Image.LANCZOS))
            labelAlerta.configure(image=ventana.alerta)
            labelAlerta.pack()
            etiqueta2 = tk.Label(emergente, text = "La señal detectada presenta demasido ruido \nEl procesamiento no es válido \nReintente con un nuevo archivo!")
            etiqueta2.pack(anchor = CENTER)
            etiqueta2.configure( font = (fuente, 10, "bold"), bg ='white')
            def alfa():
                emergente.destroy()
                datos() 
            btn_aceptar = tk.Button(emergente, text = "Aceptar",  command = alfa,  font = (fuente, 10, "bold"))
            btn_aceptar.pack(pady = 10)
            
        else:
            Maximos = np.where(I>0.0005)[0]
            if len(Maximos) == 0:
                emergente = tk.Toplevel(ventana)
                emergente.title('Process Error!!!')
                emergente.geometry(str('250x210+' + str(ventana.winfo_screenwidth()//2 - 250 // 2) + "+" + str(ventana.winfo_screenheight() // 2 - 210 // 2)))
                emergente.configure(background = 'white')
                emergente.attributes("-toolwindow", True)
                emergente.resizable(False,False)
                emergente.protocol("WM_DELETE_WINDOW", lambda: None)
                etiqueta = tk.Label(emergente, text = "Error")
                etiqueta.pack( anchor = CENTER)
                etiqueta.config(font = (fuente, 10, "bold"), fg = 'red', bg ='white')
                labelAlerta = tk.Label(emergente, bg = fondo, image = None)
                ventana.alerta = ImageTk.PhotoImage((Image.open("./Code/lib/alerta2.png")).resize((80,80), resample=Image.LANCZOS))
                labelAlerta.configure(image=ventana.alerta)
                labelAlerta.pack()
                etiqueta2 = tk.Label(emergente, text = "En la señal detectada presenta poca amplitud \nEl procesamiento no es válido \nReintente con un nuevo archivo!")
                etiqueta2.pack(anchor = CENTER)
                etiqueta2.configure( font = (fuente, 10, "bold"), bg ='white')
                def alfa():
                    emergente.destroy()
                    datos() 
                btn_aceptar = tk.Button(emergente, text = "Aceptar",  command = alfa,  font = (fuente, 10, "bold"))
                btn_aceptar.pack(pady = 10)
            else:
                maximos=np.asarray(Maximos[:])
                if len(maximos) < 100:
                    emergente = tk.Toplevel(ventana)
                    emergente.title('Process Error!!!')
                    emergente.geometry(str('250x210+' + str(ventana.winfo_screenwidth()//2 - 250 // 2) + "+" + str(ventana.winfo_screenheight() // 2 - 210 // 2)))
                    emergente.configure(background = 'white')
                    emergente.attributes("-toolwindow", True)
                    emergente.resizable(False,False)
                    emergente.protocol("WM_DELETE_WINDOW", lambda: None)
                    etiqueta = tk.Label(emergente, text = "Error")
                    etiqueta.pack( anchor = CENTER)
                    etiqueta.config(font = (fuente, 10, "bold"), fg = 'red', bg ='white')
                    labelAlerta = tk.Label(emergente, bg = fondo, image = None)
                    ventana.alerta = ImageTk.PhotoImage((Image.open("./Code/lib/alerta2.png")).resize((80,80), resample=Image.LANCZOS))
                    labelAlerta.configure(image=ventana.alerta)
                    labelAlerta.pack()
                    etiqueta2 = tk.Label(emergente, text = "Error en la detección de la onda R \nEl procesamiento no es válido \nReintente con un nuevo archivo!")
                    etiqueta2.pack(anchor = CENTER)
                    etiqueta2.configure( font = (fuente, 10, "bold"), bg ='white')
                    def alfa():
                        emergente.destroy()
                        datos() 
                    btn_aceptar = tk.Button(emergente, text = "Aceptar",  command = alfa,  font = (fuente, 10, "bold"))
                    btn_aceptar.pack(pady = 10)
                else:
                    temp = []
                    aux = [] 
                    Maximos = []
                    for i in range(len(maximos)): 
                        if i >=1 : 
                            ja = maximos[i] - maximos[i-1]
                            if (maximos[i] - maximos[i-1]) <= 20 : 
                                temp.append(maximos[i])
                            else: 
                                aux = np.asarray(I[temp][:])
                                indice = np.max(aux)
                                indice = np.where(aux == indice)[:][0][0]
                                Maximos.append(temp[indice])
                                aux = []
                                temp = [] 
                        else: 
                            temp.append(maximos[i])   
                    
                    temp = []
                    for i in range(len(Maximos)):
                        if i >= 1:
                            temp.append(Maximos[i] - Maximos[i-1])
                    Periodo = round(np.mean(temp))
                    FrecuenciaCardiaca = len(Maximos)*2    
                    Picos = I[Maximos]-0.0005
                    indice = [np.where(Picos==np.min(Picos))][:][0][0][0]
                    indice = Maximos[indice]
                    Pico = I[indice]
                    Time = Vtiempo[indice]       
                    
                    SegmentoSeñal = I[indice-round(Periodo/2):indice+round(Periodo/2)]
                    SegmentoTiempo = Vtiempo[indice-round(Periodo/2):indice+round(Periodo/2)]
                    
                    Características()
                    ObtencionParametros()

                
                    SignalPlot = I
                    Clicks = 0
                    
                    # Vista por defecto  es con la señal I
                    
                    Inicio_pico_Seg = Vtiempo[np.where (Vtiempo == SegmentoTiempo[0])][:][0]
                    Fin_pico_Seg = Vtiempo[np.where (Vtiempo == SegmentoTiempo[-1])][:][0]
                    
                    Datospicoplottiempo = Vtiempo[np.where(Vtiempo == Inicio_pico_Seg)[:][0][0] : np.where(Vtiempo == Fin_pico_Seg)[:][0][0]]
                    Datospicoplotseñal = SignalPlot[np.where(Vtiempo == Inicio_pico_Seg)[:][0][0] : np.where(Vtiempo == Fin_pico_Seg)[:][0][0]]*1000
                    
                    Datossegmentoplottiempo = Vtiempo[(Clicks*590):((Clicks+1)*590)]
                    Datossegmentoplotseñal = SignalPlot[(Clicks*590):((Clicks+1)*590)]*1000
                
                    resultadosFrame = Frame(frameMayor, bg = "#F2F2F2")
                    resultadosFrame.pack(fill = "both", expand = 1)
                    frameHeader = Frame(resultadosFrame, height = 50, bg = "#F2F2F2")
                    frameParametros = Frame(resultadosFrame, height = 100, bg = "#F2F2F2")
                    frameFooter = Frame(resultadosFrame, height = 50, bg = "#F2F2F2")
                    ecgFrame = Frame(frameParametros, height = 50, bg = "#F2F2F2")
                    
                    def res_3s():
                        global Clicks, SignalPlot, Capsulasegmento, Vtiempo
                        if Clicks > 0: 
                            Capsulasegmento.get_tk_widget().destroy()
                            Clicks = Clicks - 1
                            Datossegmentoplottiempo = Vtiempo[(Clicks*590):((Clicks+1)*590)]
                            Datossegmentoplotseñal = SignalPlot[(Clicks*590):((Clicks+1)*590)]*1000  
                            figure = plt.Figure(dpi=55)
                            Segmento = figure.add_subplot(1, 1, 1)
                            Segmento.set_ylabel('Amplitud (mV)')
                            Segmento.plot(Datossegmentoplottiempo,Datossegmentoplotseñal)
                            Segmento.set_xlabel('Tiempo (s)')
                            Segmento.set_title(' Señal I')
                            Capsulasegmento = FigureCanvasTkAgg(figure, master = ecgFrame)
                            Capsulasegmento.get_tk_widget().pack(pady = 5)     
                            # aqui debe de actualizar el plot  
                            
                        # Debe de desplazar la señal hacia atras 6seg, ojo para clicks > 0
                        
                    def sum_3s():
                        global Clicks, SignalPlot, Capsulasegmento, Vtiempo
                        if Clicks < 9:
                            Capsulasegmento.get_tk_widget().destroy()
                            Clicks = Clicks + 1
                            Datossegmentoplottiempo = Vtiempo[(Clicks*590):((Clicks+1)*590)]
                            Datossegmentoplotseñal = SignalPlot[(Clicks*590):((Clicks+1)*590)]*1000  
                            figure = plt.Figure(dpi=55)
                            Segmento = figure.add_subplot(1, 1, 1)
                            Segmento.set_ylabel('Amplitud (mV)')
                            Segmento.plot(Datossegmentoplottiempo,Datossegmentoplotseñal)
                            Segmento.set_xlabel('Tiempo (s)')
                            Segmento.set_title(' Señal I')
                            Capsulasegmento = FigureCanvasTkAgg(figure, master = ecgFrame)
                            Capsulasegmento.get_tk_widget().pack(pady = 5)     
                            # aqui debe de actualizar el plot  

                        # debe de desplazar la señal hacia adelante 6seg, ojo para clicks < 5
                    
                    def Cambiar_grafica(event):
                        global SignalPlot, Capsulasegmento, Capsulapico, Clicks
                        global Inicio_pico_Seg, Fin_pico_Seg
                        global Vtiempo, VdatoTiempo, VdataSegmento, VdatoSegmento
                        global VdatoTiempo, VdatoAmplitud, VdatoString, FrecuenciaCardiaca
                        
                        Clicks = 0
                        
                        SignalSelect = desplegable.get()
                        SignalPlot = globals()[SignalSelect]
                        
                        Datospicoplottiempo = Vtiempo[np.where(Vtiempo == Inicio_pico_Seg)[:][0][0] : np.where(Vtiempo == Fin_pico_Seg)[:][0][0]]
                        Datospicoplotseñal = SignalPlot[np.where(Vtiempo == Inicio_pico_Seg)[:][0][0] : np.where(Vtiempo == Fin_pico_Seg)[:][0][0]]*1000
                        
                        Datossegmentoplottiempo = Vtiempo[(Clicks*590):((Clicks+1)*590)]
                        Datossegmentoplotseñal = SignalPlot[(Clicks*590):((Clicks+1)*590)]*1000
                        
                        Capsulasegmento.get_tk_widget().destroy()
                        figure = plt.Figure(dpi=55)
                        Segmento = figure.add_subplot(1, 1, 1)
                        Segmento.set_ylabel('Amplitud (mV)')
                        Segmento.plot(Datossegmentoplottiempo,Datossegmentoplotseñal)
                        Segmento.set_xlabel('Tiempo (s)')
                        Segmento.set_title(' Señal '+ SignalSelect)
                        Capsulasegmento = FigureCanvasTkAgg(figure, master = ecgFrame)
                        Capsulasegmento.get_tk_widget().pack(pady = 5)     
                        Capsulapico.get_tk_widget().destroy()
                        P = (SignalPlot[np.where(Vtiempo == VdatoTiempo[1])][:][0])*1000
                        Q = (SignalPlot[np.where(Vtiempo == VdatoTiempo[3])][:][0])*1000
                        R = (SignalPlot[np.where(Vtiempo == VdatoTiempo[4])][:][0])*1000      
                        S = (SignalPlot[np.where(Vtiempo == VdatoTiempo[5])][:][0])*1000
                        T = (SignalPlot[np.where(Vtiempo == VdatoTiempo[7])][:][0])*1000    
                        figure = plt.Figure(dpi=55)
                        Pico = figure.add_subplot(1, 1, 1)
                        Pico.set_ylabel('Amplitud (mV)')
                        Pico.plot(Datospicoplottiempo,Datospicoplotseñal)
                        Pico.annotate("P", xy = (VdatoTiempo[1],P), xytext = (VdatoTiempo[1],P+0.05), arrowprops=dict(facecolor='black', arrowstyle='->'))
                        Pico.annotate("Q", xy = (VdatoTiempo[3],Q), xytext = (VdatoTiempo[3],Q-0.05), arrowprops=dict(facecolor='black', arrowstyle='->'))
                        Pico.annotate("R", xy = (VdatoTiempo[4],R), xytext = (VdatoTiempo[4],R+0.05), arrowprops=dict(facecolor='black', arrowstyle='->'))
                        Pico.annotate("S", xy = (VdatoTiempo[5],S), xytext = (VdatoTiempo[5],S-0.05), arrowprops=dict(facecolor='black', arrowstyle='->'))
                        Pico.annotate("T", xy = (VdatoTiempo[7],T), xytext = (VdatoTiempo[7],T+0.05), arrowprops=dict(facecolor='black', arrowstyle='->'))
                        Pico.set_xlabel('Tiempo (s)')
                        Pico.set_title('Pico Seleleccionado en '+ SignalSelect)
                        Capsulapico = FigureCanvasTkAgg(figure, master = picoFrame)
                        Capsulapico.get_tk_widget().pack(pady = 5)
                        
                        #actualizar el graficos el segmento y el pico

                    btnAtras = Button(ecgFrame, text = "- 3s", font = (fuente, 15, "bold"), command = res_3s)
                    btnAdelante = Button(ecgFrame, text = "+ 3s", font = (fuente, 15, "bold"), command = sum_3s)

                    btnAtras.pack(side = 'left', padx = (150, 0), pady = 50)
                    btnAdelante.pack(side = 'right', padx = (0, 150), pady = 50)

                    figure = plt.Figure(dpi=55)
                    Segmento = figure.add_subplot(1, 1, 1)
                    Segmento.set_ylabel('Amplitud (mV)')
                    Segmento.plot(Datossegmentoplottiempo,Datossegmentoplotseñal)
                    Segmento.set_xlabel('Tiempo (s)')
                    Segmento.set_title(' Señal I')
                    Capsulasegmento = FigureCanvasTkAgg(figure, master = ecgFrame)
                    Capsulasegmento.get_tk_widget().pack(pady = 5)
                    picoFrame = Frame(frameParametros, height = 50, bg = "#F2F2F2")
                    frameComplejos = Frame(picoFrame, highlightbackground="black", highlightthickness=2)
                    frameComplejos.pack(side = "left", padx = (75,0))
                    labelCabecera = Label(frameComplejos,fg = color_letra_texto, text = 'Características', bg = "#F2F2F2", font = (fuente, 13, "bold"))
                    labelCabecera.pack(pady=10)
                    tabla = ttk.Treeview(frameComplejos)
                    tabla['columns'] = ('Parámetro', 'Valor')
                    tabla.column('#0', width=0, stretch=tk.NO)  # Columna de índice oculta
                    tabla.column('Parámetro', width=115)
                    tabla.column('Valor', width=100)
                    tabla.heading('#0', text='', anchor=tk.W)
                    tabla.heading('Parámetro', text='Parámetro', anchor=tk.W)
                    tabla.heading('Valor', text='Valor', anchor=tk.W)
                    tabla.insert('', 'end', text='1', values=('Frecuencia Cardiaca ',str(FrecuenciaCardiaca)+' LPM' ))
                    tabla.insert('', 'end', text='2', values=('Momento ' + VdatoString[1],str(round(VdatoTiempo[1],2))+' Seg' ))
                    tabla.insert('', 'end', text='3', values=('Momento ' + VdatoString[3],str(round(VdatoTiempo[3],2))+' Seg' ))
                    tabla.insert('', 'end', text='4', values=('Momento ' + VdatoString[4],str(round(VdatoTiempo[4],2))+' Seg' ))
                    tabla.insert('', 'end', text='5', values=('Momento ' + VdatoString[5],str(round(VdatoTiempo[5],2))+' Seg' ))
                    tabla.insert('', 'end', text='6', values=('Momento ' + VdatoString[7],str(round(VdatoTiempo[7],2))+' Seg' ))
                    tabla.insert('', 'end', text='7', values=('Duración ' + VdatoString[1],str(round(VdatoTiempo[2]-VdatoTiempo[0],2))+' Seg' ))
                    tabla.insert('', 'end', text='8', values=('Duración ' + VdatoString[7],str(round(VdatoTiempo[8]-VdatoTiempo[6],2))+' Seg' ))
                    tabla.insert('', 'end', text='9', values=('Amplitud ' + VdatoString[4],str(round(VdatoAmplitud[4]*1000,2))+' mV' ))
                    tabla.insert('', 'end', text='10', values=('Dur. Comp. ' + VdatoSegmento[0],str(round(VdataSegmento[0],2))+' Seg' ))
                    tabla.insert('', 'end', text='11', values=('Dur. Comp. ' + VdatoSegmento[1],str(round(VdataSegmento[1],2))+' Seg' ))
                    tabla.insert('', 'end', text='12', values=('Dur. Comp. ' + VdatoSegmento[2],str(round(VdataSegmento[2],2))+' Seg' ))
                    tabla.insert('', 'end', text='13', values=('Dur. Comp. ' + VdatoSegmento[3],str(round(VdataSegmento[3],2))+' Seg' ))
                    tabla.insert('', 'end', text='14', values=('Dur. Comp. ' + VdatoSegmento[4],str(round(VdataSegmento[4],2))+' Seg' ))
                    tabla.pack()

                    desplegable = ttk.Combobox(picoFrame, values = ["I", "II", "III", "aVR", "aVL", "aVF"], font = (fuente, 13), width = 5)
                    desplegable.set("I")
                    desplegable.bind("<<ComboboxSelected>>", Cambiar_grafica)
                    desplegable.pack(side = "right", padx = (0,75))
                    
                    P = (SignalPlot[np.where(Vtiempo == VdatoTiempo[1])][:][0])*1000
                    Q = (SignalPlot[np.where(Vtiempo == VdatoTiempo[3])][:][0])*1000
                    R = (SignalPlot[np.where(Vtiempo == VdatoTiempo[4])][:][0])*1000      
                    S = (SignalPlot[np.where(Vtiempo == VdatoTiempo[5])][:][0])*1000
                    T = (SignalPlot[np.where(Vtiempo == VdatoTiempo[7])][:][0])*1000    
                    
                    figure = plt.Figure(dpi=55)
                    Pico = figure.add_subplot(1, 1, 1)
                    Pico.set_ylabel('Amplitud (mV)')
                    Pico.plot(Datospicoplottiempo,Datospicoplotseñal)
                    Pico.annotate("P", xy = (VdatoTiempo[1],P), xytext = (VdatoTiempo[1]+0.02,P), arrowprops=dict(facecolor='red', arrowstyle='->'))
                    Pico.annotate("Q", xy = (VdatoTiempo[3],Q), xytext = (VdatoTiempo[3]+0.02,Q), arrowprops=dict(facecolor='red', arrowstyle='->'))
                    Pico.annotate("R", xy = (VdatoTiempo[4],R), xytext = (VdatoTiempo[4]+0.02,R), arrowprops=dict(facecolor='red', arrowstyle='->'))
                    Pico.annotate("S", xy = (VdatoTiempo[5],S), xytext = (VdatoTiempo[5]+0.02,S), arrowprops=dict(facecolor='red', arrowstyle='->'))
                    Pico.annotate("T", xy = (VdatoTiempo[7],T), xytext = (VdatoTiempo[7]+0.02,T), arrowprops=dict(facecolor='red', arrowstyle='->'))
                    Pico.set_xlabel('Tiempo (s)')
                    Pico.set_title('Pico Seleleccionado en I')
                    Capsulapico = FigureCanvasTkAgg(figure, master = picoFrame)
                    Capsulapico.get_tk_widget().pack(pady = 5)

                    lbl8 = Label(frameHeader, text = "PARAMETROS", fg = "black", font = (fuente, 17, "bold"), bg = "#F2F2F2")
                    
                    btnVolver = Button(frameFooter, text = "Volver", font = (fuente, 12, "bold"), command = datos) 
                    btnDiagnosticar = Button(frameFooter, text = "Diagnosticar", font = (fuente, 12, "bold"), command = diagnostico)

                    frameHeader.pack(side = 'top', fill = 'x')
                    frameParametros.pack(fill = 'both', expand = '1')
                    frameFooter.pack(side = 'bottom', fill = 'x')
                    ecgFrame.pack(side = 'top', fill = 'both', expand = True)
                    picoFrame.pack(side = 'bottom', fill = 'both', expand = True)

                    lbl8.pack(side = 'top', fill = 'both', expand = 1, pady = 5)

                    btnVolver.pack(side = 'left', padx = (100, 0), pady = 25)
                    btnDiagnosticar.pack(side = 'right', padx = (0, 100), pady = 25)
                    if len(divergencias)>=1:
                        if len(divergencias)==1:
                            emergente = tk.Toplevel(ventana)
                            emergente.title('Advertencia')
                            emergente.geometry(str('240x180+' + str(ventana.winfo_screenwidth()//2 - 240 // 2) + "+" + str(ventana.winfo_screenheight() // 2 - 180 // 2)))
                            emergente.configure(background = 'white')
                            emergente.attributes("-toolwindow", True)
                            emergente.resizable(False,False)
                            emergente.protocol("WM_DELETE_WINDOW", lambda: None)
                            etiqueta = tk.Label(emergente, text = "Advertencia!!!")
                            etiqueta.pack( anchor = CENTER)
                            etiqueta.config(font = (fuente, 10, "bold"), fg = 'red', bg ='white')
                            labelAlerta = tk.Label(emergente, bg = fondo, image = None)
                            ventana.alerta = ImageTk.PhotoImage((Image.open("./Code/lib/alerta3.png")))
                            labelAlerta.configure(image=ventana.alerta)
                            labelAlerta.pack()
                            etiqueta2 = tk.Label(emergente, text =("Divergencia en la estimación de \n" + "la onda " + str(divergencias[0])))
                            etiqueta2.pack(anchor = CENTER)
                            etiqueta2.configure( font = (fuente, 10, "bold"), bg ='white')
                            btn_aceptar = tk.Button(emergente, text = "Aceptar",  command = emergente.destroy,  font = (fuente, 10, "bold"))
                            btn_aceptar.pack(pady = 10)
                        else: 
                            temp = ""
                            for i in range(len(divergencias)):
                                if i == len(divergencias)-1: 
                                    temp = temp + str(divergencias[i]) + "."
                                else: 
                                    temp = temp + str(divergencias[i]) + "," 
                            emergente = tk.Toplevel(ventana)
                            emergente.title('Advertencia')
                            emergente.geometry(str('240x180+' + str(ventana.winfo_screenwidth()//2 - 240 // 2) + "+" + str(ventana.winfo_screenheight() // 2 - 180 // 2)))
                            emergente.configure(background = 'white')
                            emergente.attributes("-toolwindow", True)
                            emergente.resizable(False,False)
                            emergente.protocol("WM_DELETE_WINDOW", lambda: None)
                            etiqueta = tk.Label(emergente, text = "Advertencia!!!")
                            etiqueta.pack( anchor = CENTER)
                            etiqueta.config(font = (fuente, 10, "bold"), fg = 'red', bg ='white')
                            labelAlerta = tk.Label(emergente, bg = fondo, image = None)
                            ventana.alerta = ImageTk.PhotoImage((Image.open("./Code/lib/alerta3.png")))
                            labelAlerta.configure(image=ventana.alerta)
                            labelAlerta.pack()
                            etiqueta2 = tk.Label(emergente, text =("Divergencia en la estimación de \n" + "las ondas " + temp))
                            etiqueta2.pack(anchor = CENTER)
                            etiqueta2.configure( font = (fuente, 10, "bold"), bg ='white')
                            btn_aceptar = tk.Button(emergente, text = "Aceptar",  command = emergente.destroy,  font = (fuente, 10, "bold"))
                            btn_aceptar.pack(pady = 10)

def diagnostico():
    for widgets in frameMayor.winfo_children():
        widgets.destroy()
    
    global diagnosticoFrame
    global VdataSegmento, VdatoTiempo, VdatoAmplitud, FrecuenciaCardiaca
    global Diagnostico
    diagnosticoFrame = Frame(frameMayor, bg = "#F2F2F2")
    diagnosticoFrame.pack(fill = "both", expand = 1)

    frameHeader = Frame(diagnosticoFrame, height = 50, bg = "#F2F2F2")
    lbl25 = Label(frameHeader, text = "DIAGNOSTICO", fg = "black", font = (fuente, 20, "bold"), bg = "#F2F2F2")
    lbl25.pack(side = 'top', fill = 'both', expand = 1, pady = 30)
    frameHeader.pack(side = 'top', fill = 'x')

    frameEnfermedades = Frame(diagnosticoFrame, height = 100, bg = "#F2F2F2")
    frameEnfermedades.pack(fill = 'both', expand = '1')
    listaFrame = Frame(frameEnfermedades, bg = "#F2F2F2", highlightbackground = 'black', highlightthickness = 2)

    lbl9 = Label(listaFrame, text = "Posibles enfermedades: ", fg = "black", font = (fuente, 13, "bold"), bg = "#F2F2F2")

    lbl9.pack(side = 'top', fill = 'both', pady = 15, padx = 50, anchor = "w")

    listaFrame.pack(side = 'left', fill = 'both', padx = 5)       

    frameSoporte = Frame(frameEnfermedades, bg = "#F2F2F2")
    lbl26 = Label(frameSoporte, text = "Soporte de diagnóstico", fg = "black", font = (fuente, 15, "bold"), bg = "#F2F2F2")
    lbl26.pack(side = 'top', fill = 'both', pady = 12, anchor = 'w')
    lbl27 = Label(frameSoporte, text = "", fg = "black", font = (fuente, 12, "bold"), bg = "#F2F2F2")
    lbl27.pack(fill = 'both', pady = 5, anchor = 'w')
    frameSoporte.pack(side = 'right', fill = 'both', expand = '1')
    
    ventana.a = tk.BooleanVar()
    ventana.b = tk.BooleanVar()
    ventana.c = tk.BooleanVar()    
    ventana.d = tk.BooleanVar()
    ventana.m = tk.BooleanVar() 
    ventana.f = tk.BooleanVar() 
    ventana.e = tk.BooleanVar() 
    ventana.g = tk.BooleanVar() 
    ventana.h = tk.BooleanVar() 
    ventana.i = tk.BooleanVar() 
    ventana.k = tk.BooleanVar() 
    ventana.l = tk.BooleanVar() 
    ventana.j = tk.BooleanVar() 

    if (VdatoTiempo[8]-VdatoTiempo[6] < 0.1 and VdatoAmplitud[1] < 0.00005 and VdataSegmento[2] > 0.12):
            ventana.a.set(True)
            lbl27['text'] = lbl27['text'] + "\n  Hiperpotasemia: \n"
            if VdatoTiempo[8]-VdatoTiempo[6] < 0.1:
                lbl27['text'] = lbl27['text'] + "- La duración de la onda T es inferior a 0.1 Seg. \n"
            if  VdatoAmplitud[1] < 0.00005:
                lbl27['text'] = lbl27['text'] + "- La amplitud de la onda P es inferior a 0.05 mV. \n"
            if VdataSegmento[2] > 0.12:
                lbl27['text'] = lbl27['text'] + "- La duración del complejo QRS es superior a 0.12 Seg. \n"
    else:
            ventana.a.set(False)

    if (VdatoAmplitud[1] > 0.00025):
        ventana.b.set(True)
        lbl27['text'] = lbl27['text'] + "\n  Hipertrofia auricular derecha: \n"
        lbl27['text'] = lbl27['text'] + "- La amplitud de la onda P es inferior a 0.25 mV. \n"       
    else:
        ventana.b.set(False)

    
    
    if (VdatoTiempo[2]-VdatoTiempo[0] > 0.1):
        ventana.c.set(True)
        lbl27['text'] = lbl27['text'] + "\n  Dilatacion auricular: \n"
        lbl27['text'] = lbl27['text'] + "- La duración de la onda P es superior a 0.1 Seg. \n"        
    else:
        ventana.c.set(False)    
        

    if (VdataSegmento[1] > 0.2):
        ventana.d.set(True)
        lbl27['text'] = lbl27['text'] + "\n  Bloqueo auroventricular: \n"
        lbl27['text'] = lbl27['text'] + "- La duración del complejo PR es superior a 0.2 Seg. \n"  
    else:
        ventana.d.set(False)
        
    if (VdataSegmento[2] > 0.12):
        ventana.m.set(True)
        lbl27['text'] = lbl27['text'] + "\n  Bloqueo de rama: \n"
        lbl27['text'] = lbl27['text'] + "- La duración del complejo QRS es superior a 0.12 Seg. \n"  
    else:
        ventana.m.set(False)
    
    if (VdataSegmento[3] > 0.42 and VdatoAmplitud[4] < 0.002):
        ventana.f.set(True)
        lbl27['text'] = lbl27['text'] + "\n  Miocarditis: \n"
        if VdataSegmento[3] > 0.42:
            lbl27['text'] = lbl27['text'] + "- La duración del complejo QT es superior a 0.42 Seg. \n" 
        if VdatoAmplitud[4] < 0.002:
            lbl27['text'] = lbl27['text'] + "- La amplitud de la onda R es inferior a 0.2 mV. \n"       
    else:
        ventana.f.set(False)
    
    if (VdataSegmento[3] < 0.32):
        ventana.e.set(True)
        lbl27['text'] = lbl27['text'] + "\n  Hipercalcemia: \n"
        lbl27['text'] = lbl27['text'] + "- La duración del complejo QT es inferior a 0.32 Seg. \n" 
    else:
        ventana.e.set(False)
        
    if (VdataSegmento[3] > 0.42):
        ventana.g.set(True)
        lbl27['text'] = lbl27['text'] + "\n  Sindrome del QT Prolongado: \n"
        lbl27['text'] = lbl27['text'] + "- La duración del complejo QT es superior a 0.42 Seg. \n" 
    else:
        ventana.g.set(False)
        
    if (VdataSegmento[4] > 0.15 and VdatoAmplitud[7] > (VdatoAmplitud[4]/0.003)):
        ventana.h.set(True)
        lbl27['text'] = lbl27['text'] + "\n  Pericarditis: \n"
        if VdataSegmento[4] > 0.15:
            lbl27['text'] = lbl27['text'] + "- La duración del complejo ST es superior a 0.15 Seg. \n"
        if  VdatoAmplitud[7] > (VdatoAmplitud[4]/0.003):
            lbl27['text'] = lbl27['text'] + "- la amplitud de la onda T es superior a la Amplitud de R / 3mV. \n"
    else:
        ventana.h.set(False)
        
    if (VdatoTiempo[8]-VdatoTiempo[6] > 0.12):
        ventana.j.set(True)
        lbl27['text'] = lbl27['text'] + "\n  Hipertrófia Auricular Izquierda: \n"
        lbl27['text'] = lbl27['text'] + "- La duración de la onda T es superior a 0.12 Seg. \n"
    else:
        ventana.j.set(False) 

    
    if (VdatoTiempo[8]-VdatoTiempo[6] > 0.12 and VdatoAmplitud[7] > (VdatoAmplitud[4]/0.003)):
        ventana.i.set(True)
        lbl27['text'] = lbl27['text'] + "\n  Isquemia: \n"
        if VdatoTiempo[8]-VdatoTiempo[6] > 0.12:
            lbl27['text'] = lbl27['text'] + "- La duración de la onda T es superior a 0.12 Seg. \n"
        if VdatoAmplitud[7] > (VdatoAmplitud[4]/0.003):
            lbl27['text'] = lbl27['text'] + "- la amplitud de la onda T es superior a la Amplitud de R / 3mV. \n"
    else:
        ventana.i.set(False)
        
    if (FrecuenciaCardiaca > 100):
        ventana.k.set(True)
        lbl27['text'] = lbl27['text'] + "\n  Taquicardia Sinusual: \n"
        lbl27['text'] = lbl27['text'] + "- La frecuencia cardiaca es superior a 100 LPM. \n"
    else:
        ventana.k.set(False)
    
    if (FrecuenciaCardiaca < 60):
        ventana.l.set(True)
        lbl27['text'] = lbl27['text'] + "\n  Bradicardia Sinusual: \n"
        lbl27['text'] = lbl27['text'] + "- La frecuencia cardiaca es inferior a 60 LPM. \n"
    else:
        ventana.l.set(False)
    
    lblDisclaimer = Label(frameSoporte, text = "Los resultados de este análisis se basan en la derivación I y no son concluyentes, \n por lo que se recomienda consultar con un médico.", bg = "#F2F2F2", fg = color_letra_texto, font = (fuente, 10, "bold"))
    lblDisclaimer.pack(side = 'bottom', pady = 10)

    lbl10 = Checkbutton(listaFrame, state = "disabled", variable = ventana.a , text = "Hiperpotasemia", fg = color_letra_texto, font = (fuente, 10))
    lbl11 = Checkbutton(listaFrame, state = "disabled", variable = ventana.b , text = "Hipertrofia auricular derecha", fg = color_letra_texto, font = (fuente, 10))
    lbl12 = Checkbutton(listaFrame, state = "disabled", variable = ventana.c , text = "Dilatacion auricular", fg = color_letra_texto, font = (fuente, 10))
    lbl15 = Checkbutton(listaFrame, state = "disabled", variable = ventana.d , text = "Bloqueo Auroventricular", fg = color_letra_texto, font = (fuente, 10))
    lbl16 = Checkbutton(listaFrame, state = "disabled", variable = ventana.e , text = "Hipercalcemia", fg = color_letra_texto, font = (fuente, 10))
    lbl17 = Checkbutton(listaFrame, state = "disabled", variable = ventana.f , text = "Miocarditis", fg = color_letra_texto, font = (fuente, 10))
    lbl18 = Checkbutton(listaFrame, state = "disabled", variable = ventana.g , text = "Sindrome del QT Prolongado", fg = color_letra_texto, font = (fuente, 10))
    lbl19 = Checkbutton(listaFrame, state = "disabled", variable = ventana.h , text = "Pericarditis", fg = color_letra_texto, font = (fuente, 10))
    lbl20 = Checkbutton(listaFrame, state = "disabled", variable = ventana.i , text = "Isquemia", fg = color_letra_texto, font = (fuente, 10))
    lbl21 = Checkbutton(listaFrame, state = "disabled", variable = ventana.j , text = "Hipertrofia auricular izquierda", fg = color_letra_texto, font = (fuente, 10))
    lbl22 = Checkbutton(listaFrame, state = "disabled", variable = ventana.k , text = "Taquicardia", fg = color_letra_texto, font = (fuente, 10))
    lbl23 = Checkbutton(listaFrame, state = "disabled", variable = ventana.l , text = "Bradicardia", fg = color_letra_texto, font = (fuente, 10))
    lbl24 = Checkbutton(listaFrame, state = "disabled", variable = ventana.m , text = "Hipertrofia Ventricular o Bloqueo de Rama", fg = color_letra_texto, font = (fuente, 10))
    lbl10.pack(anchor = "w")
    lbl11.pack(anchor = "w")
    lbl12.pack(anchor = "w")
    lbl15.pack(anchor = "w")
    lbl16.pack(anchor = "w")
    lbl17.pack(anchor = "w")
    lbl18.pack(anchor = "w")
    lbl19.pack(anchor = "w")
    lbl20.pack(anchor = "w")
    lbl21.pack(anchor = "w")
    lbl22.pack(anchor = "w")
    lbl23.pack(anchor = "w")
    lbl24.pack(anchor = "w")

    frameFooter = Frame(diagnosticoFrame, height = 50, bg = "#F2F2F2")
    btnVolver = Button(frameFooter, text = "Volver", font = (fuente, 15, "bold"), command = resultados)
    btnVolver.pack(side = 'left', padx = (100, 0), pady = 25)
    btnSalir = Button(frameFooter, text = "Salir", font = (fuente, 15, "bold"), command = ventana.destroy)
    btnSalir.pack(side = 'right', padx = (0, 100), pady = 25)
    btnHome = Button(frameFooter, text = "Inicio", font = (fuente, 15, "bold"), command = inicio)
    btnHome.pack(pady = 25)
    frameFooter.pack(side = 'bottom', fill = 'x')

inicio()
ventana.mainloop()    

###########################################################
###              SECCION DE PRUEBAS SIN UI              ###
###########################################################


# inicio = time.time()
# Preparacion()
# PDF_to_jpeg(ruta_PDF)
# crop_image()
# Concat_image()
# Limpiar_imagen()
# Lines_Hough(cv2.imread('./Code/Data/Output/img-ECG.jpeg'))
# Vectorizacion_Señal()
# ObtencionPico()
# Características()
# ObtencionParametros()
# Diagnosticar()
# fin = time.time()
# print(str(fin-inicio))

