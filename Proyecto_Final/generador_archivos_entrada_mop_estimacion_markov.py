# este script sirve para generar datos.txt que se usa como entrada para estimar 
# un proceso estocastico de Markov
# en nuestro caso pone los datos de aportes de Claire

import pandas as pd

df_aporte_claire = pd.read_csv(f"Datos\\Claire\\aporte_claire.csv", sep=",", header=0, encoding='cp1252')

df_datos_entrada_originales_mop = pd.read_csv(
    f"Datos\\MARKOV\\Entradas\\datos_originales.txt",
    sep="\t",
    skiprows=7,     # saltea cuatro filas de metadatos
    header=0        # la fila que queda ahora es el header
)

df_datos_entrada_mop = df_datos_entrada_originales_mop.copy()

df_datos_entrada_mop = df_datos_entrada_mop[["CRONICA","PASO","APORTE-BONETE"]]

df_datos_entrada_mop = df_datos_entrada_mop.rename(columns={"APORTE-BONETE": "APORTE-CLAIRE"})

for i in range(len(df_aporte_claire.columns)):
    inicio_cronica = i*52
    fin_cronica = inicio_cronica + 52 
    df_datos_entrada_mop.iloc[inicio_cronica:fin_cronica,2] = df_aporte_claire.iloc[:,i]

df_datos_entrada_mop.to_csv(f"Datos\\MARKOV\\Entradas\\datos.txt", sep="\t", index=False)
