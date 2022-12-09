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


if __name__ == "__main__":
    app.run(debug=True)