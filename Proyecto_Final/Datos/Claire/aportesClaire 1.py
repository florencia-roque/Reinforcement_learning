import pandas as pd
import numpy as np


# Levanto datos historicos de los aportes del proceso estocastico "Historico_Markov_2025_116c"
rutaArchivo = r'D:\Reinforcement_learning\Proyecto_Final\Datos\MOP\datosProcHistorico.xlt'
df = pd.read_csv(rutaArchivo, sep='\s+', header=7, encoding='cp1252')
df['APORTE-SALTO'] = df['APORTE-SALTO']/2

df['APORTE-CLAIRE'] = df[['APORTE-BONETE', 'APORTE-PALMAR', 'APORTE-SALTO']].sum(axis=1)


df = df.rename(columns={'Estacion': 'Semana'})

cols_aporte = ['APORTE-BONETE', 'APORTE-PALMAR', 'APORTE-SALTO']

# a) Si usan punto decimal
df[cols_aporte] = df[cols_aporte].apply(pd.to_numeric, errors='coerce')

nuevo_df = df[['Cronica', 'Semana', 'APORTE-CLAIRE']]
pivot_df = nuevo_df.pivot(index='Semana', columns='Cronica', values='APORTE-CLAIRE')

pivot_df.to_csv('aporte_claire.csv', index=False)


n_clases = 5  # o 4, 10, etc.

# Se crea el csv "Clasificado" que tiene para cada semana y para cada año, la clase a la cual pertenece el aporte de Claire 
# dividiendo en quintiles los valores  cada semana. La clasificacion en clases se hace con qcut.
df_clasificado = pd.DataFrame(index=pivot_df.index, columns=pivot_df.columns)

for semana_idx, fila in pivot_df.iterrows():
    valores = pd.to_numeric(fila, errors='coerce')
    try:
        clases = pd.qcut(valores, q=n_clases, labels=[i for i in range(n_clases)])
    except ValueError:
        clases = pd.Series([None]*len(valores), index=valores.index)   
    df_clasificado.loc[semana_idx] = clases
# Resultado:
df_clasificado.to_csv('clasificado.csv', index=False)
print(df_clasificado.head())


df_numeric = df_clasificado.replace({i for i in range(n_clases)}) 
df_numeric = df_numeric.dropna()

# Al final del dataframe agrego la primer fila para que la ultima fila la compare con la primera
# En esta ultima fila agregada, al primer valor lo pongo al final para que el valor de la semana 52 se compare con la semana 1 del año siguiente
primer_fila =  df_numeric.iloc[0].tolist()
valor = primer_fila.pop(1)
primer_fila.append(valor)
nueva_fila = pd.DataFrame([primer_fila], columns=df_numeric.columns)
df_rotado = pd.concat([df_numeric,nueva_fila], ignore_index=True)



# Número de clases
n_clases = df_clasificado.max().max() 
matrices_por_semana = {}

# Total de semanas
semanas = df_clasificado.index.tolist()

# Defino este metodo para calcular las matrices de transicion usando una semana y la siguiente
def calcular_matriz_transicion(origen, destino, clases=5):
    matriz = np.zeros((clases, clases), dtype=int)

    for o, d in zip(origen, destino):
        if pd.notna(o) and pd.notna(d):
            matriz[int(o), int(d)] += 1

    # Convertir a porcentajes fila a fila (porcentaje de transiciones desde cada estado)
    matriz_porcentual = matriz / matriz.sum(axis=1, keepdims=True) * 100

    return matriz, matriz_porcentual

# Guardo en filas cada una de las matrices aplanadas a medida que voy calculando las matrices de transicion
filas = []

for i in range(len(semanas)):
    semana_actual = semanas[i]
    semana_siguiente = semanas[(i + 1) % len(semanas)]  # para cerrar ciclo 52 → 1

    origen = df_clasificado.loc[semana_actual].values
    destino = df_clasificado.loc[semana_siguiente].values

    print(origen)
    print(destino)
    # Matriz de conteo para esta transición
    matriz = calcular_matriz_transicion(origen=origen, destino=destino)
    print(matriz)


    fila = [i] + matriz[1].flatten().tolist()  # aplana la matriz y agrega el índice
    filas.append(fila)


n_clases = 5
columnas = ['Semana'] + [f'{i}-{j}' for i in range(n_clases) for j in range(n_clases)]
df_resultado = pd.DataFrame(filas, columns=columnas)
df_resultado.to_csv('matrices_sem.csv', index=False, header=True)




