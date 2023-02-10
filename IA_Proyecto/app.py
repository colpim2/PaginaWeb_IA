import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Landing.html')

@app.route('/predictApriori',methods=['POST'])
def predictApriori():
    #Convierte a flotante los valores obtenidos
    min_soporte = float(list(request.form.values())[0])
    min_confianza = float(list(request.form.values())[1])
    lift = float(list(request.form.values())[2])

    # Importing the libraries
    import pandas as pd                 # Para la manipulación y análisis de los datos
    import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
    from apyori import apriori

    DatosTransacciones = pd.read_csv('Datos/TV_Shows.csv', header=None)

    #Se incluyen todas las transacciones en una sola lista
    Transacciones = DatosTransacciones.values.reshape(-1).tolist() #-1 significa 'dimensión desconocida'

    #Se crea una matriz (dataframe) usando la lista y se incluye una columna 'Frecuencia'
    Lista = pd.DataFrame(Transacciones)
    Lista['Frecuencia'] = 1

    #Se agrupa los elementos
    Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
    Lista['Porcentaje'] = (Lista['Frecuencia'] / Lista['Frecuencia'].sum()) #Porcentaje
    Lista = Lista.rename(columns={0 : 'Item'})

    # Se genera un gráfico de barras
    plt.figure(figsize=(16,16), dpi=300)
    plt.ylabel('Item')
    plt.xlabel('Frecuencia')
    plt.barh(Lista['Item'], width=Lista['Frecuencia'], color='blue')
    plt.savefig('static/public/assets/img/apriori.png', format='png', bbox_inches='tight')

    TransaccionesLista = DatosTransacciones.stack().groupby(level=0).apply(list).tolist()
    Reglas = apriori(TransaccionesLista,min_support=min_soporte,min_confidence=min_confianza,min_lift=lift)
    
    Resultados = list(Reglas)
    i = 0
    ReglaFormato = []

    for item in Resultados:
        auxcadena = ''
        #El primer índice de la lista
        Emparejar = item[0]
        items = [x for x in Emparejar]
        auxcadena ="Regla: " + str(item[0]) + "\n"
        i+=1
        ReglaFormato.insert(i,auxcadena)

        #El segundo índice de la lista
        auxcadena = "Soporte: " + str(item[1]) + "\n"
        i+=1
        ReglaFormato.insert(i,auxcadena)

        #El tercer índice de la lista
        auxcadena = "Confianza: " + str(item[2][0][2]) + "\n"
        i+=1
        ReglaFormato.insert(i,auxcadena)
        auxcadena = "Lift: " + str(item[2][0][3]) + "\n"
        i+=1
        ReglaFormato.insert(i,auxcadena)
        auxcadena ="====================================="  + "\n"
        ReglaFormato.insert(i,auxcadena)
        i+=1

    return render_template('Landing.html', ReglasApriori = ReglaFormato)

@app.route('/predictDistancias',methods=['POST'])
def predictDistancias():
    #Convierte a int los valores obtenidos
    obj1 = int(list(request.form.values())[0])
    obj2 = int(list(request.form.values())[1])

    import pandas as pd                         # Para la manipulación y análisis de datos
    import numpy as np                          # Para crear vectores y matrices n dimensionales
    import matplotlib.pyplot as plt             # Para generar gráficas a partir de los datos
    from scipy.spatial.distance import cdist    # Para el cálculo de distancias
    from scipy.spatial import distance

    Datos = pd.read_csv("Datos/WDBCOriginal.csv") #Considerando que existe un encabezado
    Datos = Datos.drop(columns=['IDNumber','Diagnosis'])

    import seaborn as sea 
    plt.figure(figsize=(5,4))
    MapaCalor=np.triu(Datos.corr())
    sea.heatmap(Datos.corr(),cmap='RdBu_r', annot=True, mask=MapaCalor)
    plt.savefig('static/public/assets/img/mapaCalorDistancias.png', format='png', bbox_inches='tight')

    from sklearn.preprocessing import StandardScaler, MinMaxScaler  
    estandarizar = StandardScaler()                               
    MEstandarizada = estandarizar.fit_transform(Datos)  

    #Calculo de todas las distancias
    DstEuclidiana = cdist(MEstandarizada, MEstandarizada, metric='euclidean')
    DstChebyshev = cdist(MEstandarizada, MEstandarizada, metric='chebyshev')
    DstManhattan = cdist(MEstandarizada, MEstandarizada, metric='cityblock')
    DstMinkowski = cdist(MEstandarizada, MEstandarizada, metric='minkowski', p=1.5)
    MEuclidiana = pd.DataFrame(DstEuclidiana)
    MChebyshev = pd.DataFrame(DstChebyshev)
    MManhattan = pd.DataFrame(DstManhattan)
    MMinkowski = pd.DataFrame(DstMinkowski)

    #Comparación entre objetos
    Objeto1 = MEstandarizada[obj1]
    Objeto2 = MEstandarizada[obj2]
    dstEuclidiana = distance.euclidean(Objeto1,Objeto2)
    dstChebyshev = distance.chebyshev(Objeto1,Objeto2)
    dstManhattan = distance.cityblock(Objeto1,Objeto2)
    dstMinkowski = distance.minkowski(Objeto1,Objeto2, p=1.5)

    #Lista de salida
    DistanciaLista = []
    DistanciaLista.insert(1,"Distancia entre objeto "+str(obj1)+ " y objeto "+ str(obj2))
    DistanciaLista.insert(1,'Euclidiana: '+str(dstEuclidiana)+'\n')
    DistanciaLista.insert(2,'Chebyshev: '+str(dstChebyshev)+'\n')
    DistanciaLista.insert(3,'Manhattan: '+str(dstManhattan)+'\n')
    DistanciaLista.insert(4,'Minkowski: '+str(dstMinkowski)+'\n')

    return render_template('Landing.html', Distancias = DistanciaLista)

@app.route('/predictJerarquico',methods=['POST'])
def predictJerarquico():
    #Convierte a int los valores obtenidos
    metodo = str(list(request.form.values())[0])

    import pandas as pd               # Para la manipulación y análisis de datos
    import numpy as np                # Para crear vectores y matrices n dimensionales
    import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
    import seaborn as sns             # Para la visualización de datos basado en matplotlib

    BCancer = pd.read_csv('Datos/WDBCOriginal.csv')
    CorrBCancer = BCancer.corr(method='pearson')

    plt.figure(figsize=(14,7))
    MatrizInf = np.triu(CorrBCancer)
    sns.heatmap(CorrBCancer, cmap='RdBu_r', annot=True, mask=MatrizInf)
    plt.savefig('static/public/assets/img/mapaCalorClustering.png', format='png', bbox_inches='tight')

    MatrizVariables = np.array(BCancer[['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension']])

    from sklearn.preprocessing import StandardScaler, MinMaxScaler  
    estandarizar = StandardScaler()                                # Se instancia el objeto StandardScaler o MinMaxScaler 
    MEstandarizada = estandarizar.fit_transform(MatrizVariables) 

    #Se importan las bibliotecas de clustering jerárquico para crear el árbol
    import scipy.cluster.hierarchy as shc
    from sklearn.cluster import AgglomerativeClustering
    plt.figure(figsize=(10, 7))
    plt.title("Pacientes con cáncer de mama")
    plt.xlabel('Observaciones')
    plt.ylabel('Distancia')
    Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric=metodo))
    plt.savefig('static/public/assets/img/ArbolJerarquicoClustering.png', format='png', bbox_inches='tight')

    #Se crean las etiquetas de los elementos en los clusters
    MJerarquico = AgglomerativeClustering(n_clusters=4, linkage='complete', affinity=metodo)
    MJerarquico.fit_predict(MEstandarizada)

    #Centroides
    BCancer['clusterH'] = MJerarquico.labels_
    CentroidesH = BCancer.groupby(['clusterH'])['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension'].mean()

    plt.figure(figsize=(10, 7))
    plt.scatter(MEstandarizada[:,0], MEstandarizada[:,1], c=MJerarquico.labels_)
    plt.grid()
    plt.savefig('static/public/assets/img/DispesionJerarquicoClustering.png', format='png', bbox_inches='tight')

    return render_template('Landing.html')

@app.route('/predictParticional',methods=['POST'])
def predictParticional():
    #Convierte a int los valores obtenidos
    metodo = str(list(request.form.values())[0])

    import pandas as pd               # Para la manipulación y análisis de datos
    import numpy as np                # Para crear vectores y matrices n dimensionales
    import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
    import seaborn as sns             # Para la visualización de datos basado en matplotlib

    BCancer = pd.read_csv('Datos/WDBCOriginal.csv')
    CorrBCancer = BCancer.corr(method='pearson')

    MatrizVariables = np.array(BCancer[['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension']])

    from sklearn.preprocessing import StandardScaler, MinMaxScaler  
    estandarizar = StandardScaler()                                # Se instancia el objeto StandardScaler o MinMaxScaler 
    MEstandarizada = estandarizar.fit_transform(MatrizVariables) 

    #Se importan las bibliotecas
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min

    #Definición de k clusters para K-means
    #Se utiliza random_state para inicializar el generador interno de números aleatorios
    SSE = [] #sumatoria de los clustres
    for i in range(2, 12):
        km = KMeans(n_clusters=i, random_state=0)
        km.fit(MEstandarizada)
        SSE.append(km.inertia_)

    #Se grafica SSE en función de k
    plt.figure(figsize=(10, 7))
    plt.plot(range(2, 12), SSE, marker='o')
    plt.xlabel('Cantidad de clusters *k*')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    plt.savefig('static/public/assets/img/ElbowMethodClustering.png', format='png', bbox_inches='tight')

    from kneed import KneeLocator
    kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")

    plt.style.use('ggplot')
    kl.plot_knee()
    plt.savefig('static/public/assets/img/KneePlot.png', format='png', bbox_inches='tight')

    #Se crean las etiquetas de los elementos en los clusters
    MParticional = KMeans(n_clusters=5, random_state=0).fit(MEstandarizada)
    MParticional.predict(MEstandarizada)

    BCancer['clusterP'] = MParticional.labels_

    #Centroides
    CentroidesP = BCancer.groupby(['clusterP'])['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension'].mean()

    # Gráfica de los elementos y los centros de los clusters
    plt.figure(figsize=(10, 7))
    plt.rcdefaults()
    plt.scatter(MEstandarizada[:,0], MEstandarizada[:,1], c=MParticional.labels_)
    plt.grid()
    plt.savefig('static/public/assets/img/Dispersion3DParticional.png', format='png', bbox_inches='tight')

    return render_template('Landing.html',metodoout=metodo)

@app.route('/predictLogistica',methods=['POST'])
def predictLogistica():
    #Convierte a int y float los valores obtenidos
    embarazos = int(list(request.form.values())[0])
    glucosa = int(list(request.form.values())[1])
    presionArterial = int(list(request.form.values())[2])
    espesorPliegue = int(list(request.form.values())[3])
    insulina = int(list(request.form.values())[4])
    indiceMasaCorporal = float(list(request.form.values())[5])
    funcionPedigriDiabetes = float(list(request.form.values())[6])
    edad = int(list(request.form.values())[7])

    import pandas as pd               
    import numpy as np                
    import matplotlib.pyplot as plt   
    import seaborn as sns             # Para la visualización de datos basado en matplotlib

    Diabetes = pd.read_csv('Datos/Diabetes.csv')

    #Agrupación
    plt.figure(figsize=(10, 7))
    plt.scatter(Diabetes['BloodPressure'], Diabetes['Glucose'], c = Diabetes.Outcome)
    plt.grid()
    plt.xlabel('BloodPressure')
    plt.ylabel('Glucose')
    plt.title('Agrupamiento por Diagnostico')
    plt.savefig('static/public/assets/img/AgrupamientoLogistico.png', format='png', bbox_inches='tight')

    #Mapa de Calor
    plt.figure(figsize=(14,7))
    MatrizInf = np.triu(Diabetes.corr())
    sns.heatmap(Diabetes.corr(), cmap='RdBu_r', annot=True, mask=MatrizInf)
    plt.savefig('static/public/assets/img/mapaCalorLogistico.png', format='png', bbox_inches='tight')

    #Variables predictoras
    X = np.array(Diabetes[['Pregnancies', 
                        'Glucose', 
                        'BloodPressure', 
                        'SkinThickness', 
                        'Insulin', 
                        'BMI',
                        'DiabetesPedigreeFunction',
                        'Age']])
    #Variable clase
    Y = np.array(Diabetes[['Outcome']])

    #Importar bibliotecas para la creación de modelo
    from sklearn import model_selection
    from sklearn import linear_model
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    
    #Entrenamiento del Modelo
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)

    ClasificacionRL = linear_model.LogisticRegression()
    ClasificacionRL.fit(X_train, Y_train)

    #Clasificación final 
    Y_ClasificacionRL = ClasificacionRL.predict(X_validation)

    #Matriz de clasificación
    ModeloClasificacion = ClasificacionRL.predict(X_validation)
    Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                    ModeloClasificacion, 
                                    rownames=['Reales'], 
                                    colnames=['Clasificación']) 

    #Curva ROC
    from sklearn.metrics import RocCurveDisplay
    CurvaROC = RocCurveDisplay.from_estimator(ClasificacionRL, X_validation, Y_validation, name="Diabetes")
    plt.savefig('static/public/assets/img/curvaROCLogistico.png', format='png', bbox_inches='tight')

    #Paciente
    PacienteID = pd.DataFrame({'Pregnancies': [embarazos],
                        'Glucose': [glucosa],
                        'BloodPressure': [presionArterial],
                        'SkinThickness': [espesorPliegue],
                        'Insulin': [insulina],
                        'BMI': [indiceMasaCorporal],
                        'DiabetesPedigreeFunction': [funcionPedigriDiabetes],
                        'Age': [edad]})

    #Reporte
    Reporte = []
    if ClasificacionRL.predict(PacienteID) == 1:
        Reporte.insert(1,"Diabetetico")
    else:
        Reporte.insert(1,"No Diabetetico")
    Reporte.insert(2,"Exactitud: "+ str(accuracy_score(Y_validation, Y_ClasificacionRL)*100)+'%')

    return render_template('Landing.html', ReporteLogistico = Reporte)

@app.route('/predictADyBA',methods=['POST'])
def predictADyBA():
    #Obtener nombre de ticket acción y valores de pronostico
    accion = str(list(request.form.values())[0])
    valorAbierto = float(list(request.form.values())[1])
    valorAlto = float(list(request.form.values())[2])
    valorBajo = float(list(request.form.values())[3])

    import pandas as pd               
    import numpy as np                
    import matplotlib.pyplot as plt   
    import seaborn as sns             
    import yfinance as yf

    #Importar información
    DataAccion = yf.Ticker(accion)
    AccionHist = DataAccion.history(start = '2019-1-1', end = '2022-12-31', interval='1d')

    plt.figure(figsize=(20, 5))
    plt.plot(AccionHist['Open'], color='purple', marker='+', label='Open')
    plt.plot(AccionHist['High'], color='blue', marker='+', label='High')
    plt.plot(AccionHist['Low'], color='orange', marker='+', label='Low')
    plt.plot(AccionHist['Close'], color='green', marker='+', label='Close')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de las acciones')
    plt.title(accion)
    plt.grid(True)
    plt.legend()
    plt.savefig('static/public/assets/img/accionesAD.png', format='png', bbox_inches='tight')

    MDatos = AccionHist.drop(columns = ['Volume', 'Dividends', 'Stock Splits'])
    # En caso de tener valores nulos
    MDatos = MDatos.dropna()

    # ========================== Árbol de Decisión =======================
    from sklearn import model_selection
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    #Variables predictoras
    X = np.array(MDatos[['Open',
                        'High',
                        'Low']])
    #Variables a pronosticar
    Y = np.array(MDatos[['Close']])

    #División de datos
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 0, 
                                                                    shuffle = True)

    #Entrenamiento e modelo
    PronosticoAD = DecisionTreeRegressor(max_depth=9, min_samples_split=8, min_samples_leaf=4, random_state=0)
    PronosticoAD.fit(X_train, Y_train)

    #Se genera el pronóstico
    Y_Pronostico = PronosticoAD.predict(X_test)
    Valores = pd.DataFrame(Y_test, Y_Pronostico)

    #Modelo
    plt.figure(figsize=(20, 5))
    plt.plot(Y_test, color='red', marker='+', label='Real')
    plt.plot(Y_Pronostico, color='green', marker='+', label='Estimado')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de las acciones')
    plt.title('Pronóstico de las acciones')
    plt.grid(True)
    plt.legend()
    plt.savefig('static/public/assets/img/modeloAD.png', format='png', bbox_inches='tight')

    #Generación del árbol
    from sklearn.tree import plot_tree
    plt.figure(figsize=(16,16))  
    plot_tree(PronosticoAD, feature_names = ['Open', 'High', 'Low'])
    plt.savefig('static/public/assets/img/AD.png', format='png', bbox_inches='tight')

    #Objeto acción AD
    PrecioAccionAD = pd.DataFrame({'Open': [valorAbierto],
                            'High': [valorAlto], 
                            'Low': [valorBajo]})

    #Reporte Árbol de Decisión
    Reporte = []
    Reporte.insert(1,'Pronostico valor: '+str(PronosticoAD.predict(PrecioAccionAD)))   
    Reporte.insert(2,'Criterio: ' +str(PronosticoAD.criterion))
    Reporte.insert(3,'Importancia variables: '+str(PronosticoAD.feature_importances_))
    Reporte.insert(4,"MAE: "+str(mean_absolute_error(Y_test, Y_Pronostico)))
    Reporte.insert(5,"MSE: "+str(mean_squared_error(Y_test, Y_Pronostico)))
    Reporte.insert(6,"RMSE: "+str(mean_squared_error(Y_test, Y_Pronostico, squared=False)))
    Reporte.insert(7,'Score: '+str(r2_score(Y_test, Y_Pronostico)))

    # ========================== Bosque Aletorio =======================
    from sklearn import model_selection
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    #Variables predictorias
    X2 = np.array(MDatos[['Open',
                        'High',
                        'Low']])
    #Variables a pronosticar
    Y2 = np.array(MDatos[['Close']])

    #División de datos
    X2_train, X2_test, Y2_train, Y2_test = model_selection.train_test_split(X2, Y2, 
                                                                    test_size = 0.2, 
                                                                    random_state = 0, 
                                                                    shuffle = True)

    #Entrenar modelo
    PronosticoBA = RandomForestRegressor(n_estimators=105, max_depth=8, min_samples_split=8, min_samples_leaf=4, random_state=0)
    PronosticoBA.fit(X2_train, Y2_train)
    Y2_Pronostico = PronosticoBA.predict(X2_test)
    Valores2 = pd.DataFrame(Y2_test, Y2_Pronostico)

    #Modelo
    plt.figure(figsize=(20, 5))
    plt.plot(Y2_test, color='red', marker='+', label='Real')
    plt.plot(Y2_Pronostico, color='green', marker='+', label='Estimado')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de las acciones')
    plt.title('Pronóstico de las acciones de Amazon')
    plt.grid(True)
    plt.legend()
    plt.savefig('static/public/assets/img/modeloBA.png', format='png', bbox_inches='tight')

    #Generaición de bosque
    Estimador = PronosticoBA.estimators_[50]
    from sklearn.tree import plot_tree
    plt.figure(figsize=(16,16))  
    plot_tree(Estimador, 
            feature_names = ['Open', 'High', 'Low'])
    plt.savefig('static/public/assets/img/BA.png', format='png', bbox_inches='tight')

    #Objeto pronostico
    PrecioAccionBA = pd.DataFrame({'Open': [valorAbierto],
                            'High': [valorAlto], 
                            'Low': [valorBajo]})

    #Reporte 
    Reporte2 = []
    Reporte2.insert(1,'Pronostico valor (Bosque Aleatorio): '+str(PronosticoBA.predict(PrecioAccionBA)))
    Reporte2.insert(2,'Criterio:' + str(PronosticoBA.criterion))
    Reporte2.insert(3,'Importancia variables: ' + str(PronosticoBA.feature_importances_))
    Reporte2.insert(4,"MAE: " +str(mean_absolute_error(Y2_test, Y2_Pronostico)))
    Reporte2.insert(5,"MSE: " +str(mean_squared_error(Y2_test, Y2_Pronostico)))
    Reporte2.insert(6,"RMSE: " +str(mean_squared_error(Y2_test, Y2_Pronostico, squared=False)))
    Reporte2.insert(7,'Score: ' +str(r2_score(Y2_test, Y2_Pronostico)))

    return render_template('Landing.html', ReporteAD = Reporte, ReporteBA = Reporte2)


if __name__ == "__main__":
    app.run(debug=True)