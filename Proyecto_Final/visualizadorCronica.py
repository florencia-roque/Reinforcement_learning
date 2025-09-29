import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import unicodedata
import os

# === Función para normalizar nombres de columnas ===
def normalizar_columna(col):
    col = col.lower().replace(" ", "_")
    col = ''.join(c for c in unicodedata.normalize('NFD', col) if unicodedata.category(c) != 'Mn')
    return col

# === Abrir file chooser ===
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Selecciona el archivo CSV",
    filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
)
if not file_path:
    print("No se seleccionó ningún archivo.")
    exit()

# === Leer CSV y normalizar columnas ===
if file_path.endswith('csv'):
    df = pd.read_csv(file_path, sep=",", engine="python")
    df.columns = [normalizar_columna(c) for c in df.columns]
    print("Columnas normalizadas:", df.columns.tolist())

elif file_path.endswith('xlsx'):
    df = pd.read_excel(file_path,header=0)
    df.columns = [normalizar_columna(c) for c in df.columns]
    print("Columnas normalizadas:", df.columns.tolist())

# === Definir columnas ===
col_turbinada = "energia_hidro"
col_renovable = "energia_renovable"
col_termico_bajo = "energia_termico_bajo"
col_termico_alto = "energia_termico_alto"
col_demanda = "demanda"
col_aportes = "aportes"
col_volumen = "volumen"

# Eje X
x = df.index

# === Configuración global de estilo ===
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 20,
})

# === Crear figura alta resolución ===
fig, ax = plt.subplots(figsize=(14, 6), dpi=400)

# Áreas apiladas: mantenemos colores anteriores
ax.stackplot(
    x,
    df[col_turbinada],     # Hydro (azul por defecto)
    df[col_termico_bajo],  # Thermal low-cost (naranja)
    df[col_termico_alto],  # Thermal high-cost (verde)
    labels=["Hydro", "Thermal low-cost", "Thermal high-cost"],
    colors=["#40a0e5", "#fffb87", "#e35e5e"]  # azul, naranja, verde
)

# Demanda (negro)
ax.plot(x, df[col_demanda], label="Demand", color="black", linewidth=2)

# Eje secundario
ax2 = ax.twinx()

# Inflows: marrón punteado
ax2.plot(x, df[col_aportes], label="Inflows", color="#8B4513", linestyle="--", linewidth=2.2)

# Reservoir volume: **naranja continua**
ax2.plot(x, df[col_volumen], label="Reservoir volume", color="#ff7f0e", linestyle="-", linewidth=2.4)

# Títulos y etiquetas (en inglés)
ax.set_title("Energies: Generation and Demand / Volume and Inflows", pad=8)
ax.set_xlabel("Week", labelpad=8)
ax.set_ylabel("Energy [MWh]", labelpad=8)

# ↓ Etiqueta del eje derecho más chica y con espacio extra para que no se corte
ax2.set_ylabel("Volume [hm³]/Inflows [hm³/week]", fontsize=19, labelpad=8)

# Leyenda combinada, un poco más abajo para no chocar con 'Week'
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4)

# Layout: más margen derecho y abajo para etiquetas y leyenda
plt.tight_layout(rect=[0, 0, 0.98, 1])  # deja 2% libre a la derecha
fig.subplots_adjust(right=0.89, bottom=0.28)

# Guardado
os.makedirs("figures/paper/prueba", exist_ok=True)
plt.savefig("figures/paper/prueba/dispatch_evaluation_det.png", dpi=400, bbox_inches="tight")
plt.savefig("figures/paper/prueba/dispatch_evaluation_det.pdf", bbox_inches="tight")  # vectorial para el paper
plt.show()