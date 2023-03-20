from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd
import random

#creamos el dataframe
datos = pd.DataFrame()
#numero de puntos
n=1000
#llenamos el dataframe con 1000 puntos aleatorios dentro de un triangulo equilatero de lado 0.6, etiquetados como dentro coordenadas de los vertices del triangulo (0.2,0.2)(0.8,0.2)(0.5,0.72) altura = [sqrt(3)/2]*l ~ 0.52
while n>0:
    (x , y) = ( random.uniform(0.2,0.8) , random.uniform(0.2,0.72) )
    if y < (1.7333*x-0.147) and y < (-1.7333*x+1.586):
        datos = pd.concat([datos,pd.Series({'x': x,  'y': y, 'etiqueta': 'dentro','color':'cornflowerblue'}).to_frame().T],ignore_index=True)
        n-=1

#llenamos el dataframe con datos etiquetados como 'fuera' en la parte interna de un cuadro de 1x1 pero fuera de un triangulo
while n<1000:
    (x , y) = ( random.uniform(0.0,1.0) , random.uniform(0.0,1.0) )
    if y < 0.2 or y > (1.7333*x-0.147) or y > (-1.7333*x+1.586):
        datos=pd.concat([datos,pd.Series({'x': x,  'y': y, 'etiqueta': 'fuera','color':'mediumorchid'}).to_frame().T],ignore_index=True)
        n+=1

#mostramos la grafica de puntos donde los azules(cornflowerblue) son etiquetados dentro y los morados(mediumorchid) fuera
ax =datos.plot.scatter(x='x',y='y',c='color')
ax.set_title('Triangulo')
plt.show()

#creamos el bosque de arboles aleatorios con 100 arboles, basado en entropia, numero de caracteristicas la raiz cuadrada de caracteristicas, seleccion aleatoria de muestras con 2/3 de los datos y el oob(out of bag) para verificar que funcione el bosque
bosque = RandomForestClassifier(n_estimators=100, criterion="entropy",max_features="sqrt",bootstrap=True,max_samples=2/3,oob_score=True)
#creamos el bosque con los datos que creamos
bosque.fit(datos[datos.columns[:-2]].values,datos['etiqueta'].values)
#revisamos una prediccion partiular
print('prediccion del punto (0.11,0.11): ',bosque.predict([[0.11,0.11]]))
#evaluamos con los datos con los que creamos el bosque
print('evaluacion de los datos: ',bosque.score(datos[datos.columns[:-2]].values,datos['etiqueta'].values))
#evaluamos con los datos oob
print('evaluacion con datos para verificar: ',bosque.oob_score_)
#imprimimos dos arboles del bosque
for arbol in bosque.estimators_[:2]:
    tree.plot_tree(arbol, feature_names=datos.columns[:-1])
    plt.show()

