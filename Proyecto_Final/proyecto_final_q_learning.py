# type: ignore

import os
import time
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
matplotlib.use("TkAgg")

# --- Plotter simple de recompensas por episodio ---
class LiveRewardPlotter:
    def __init__(self, window=100, refresh_every=10, title="Recompensa por episodio"):
        self.window = window
        self.refresh_every = refresh_every
        self.rewards_ep = []
        self.moving_avg = []

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Episodio")
        self.ax.set_ylabel("Recompensa")
        self.ax.set_title(title)
        self.ax.grid(True)

        (self.line,) = self.ax.plot([], [], lw=1, label="Reward")
        (self.line_avg,) = self.ax.plot([], [], lw=2, label=f"Media móvil ({window})")
        self.ax.legend()
        self.fig.show()

    def update(self, r):
        self.rewards_ep.append(float(r))
        # media móvil
        w = min(self.window, len(self.rewards_ep))
        self.moving_avg.append(np.mean(self.rewards_ep[-w:]))

        # refrescar cada N episodios
        if len(self.rewards_ep) % self.refresh_every == 0:
            x = np.arange(1, len(self.rewards_ep) + 1)
            self.line.set_data(x, self.rewards_ep)
            self.line_avg.set_data(x, self.moving_avg)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.show(block=False)

# Leer archivo 
def leer_archivo(rutaArchivo, sep=None, header=0, sheet_name=0):
    if rutaArchivo.endswith('.xlsx') or rutaArchivo.endswith('.xls'):
        return pd.read_excel(rutaArchivo, header=header,sheet_name=sheet_name)
    else:
        return pd.read_csv(rutaArchivo, sep=sep, header=header, encoding='cp1252')

class HydroThermalEnv(gym.Env):
    T0 = 0
    T_MAX = 103
    N_HIDRO = 5

    P_CLAIRE_MAX = 1541 # MW
    P_SOLAR_MAX = 254 # MW
    P_EOLICO_MAX = 1584.7 # MW
    P_BIOMASA_MAX = 487.3 # MW

    P_TERMICO_BAJO_MAX = 500 # MW
    P_TERMICO_ALTO_MAX = 5000 # MW

    Q_CLAIRE_MAX = 11280 * 3600 / 1e6 # hm3/h

    V_CLAIRE_MIN = 0 # hm3
    V_CLAIRE_MAX = 12500*20 # hm3
    V0 = V_CLAIRE_MAX / 20 # hm3

    K_CLAIRE = P_CLAIRE_MAX / Q_CLAIRE_MAX # MWh/hm3

    V_CLAIRE_TUR_MAX = P_CLAIRE_MAX * 168 / K_CLAIRE # hm3

    VALOR_EXPORTACION = 0 # USD/MWh 
    COSTO_TERMICO_BAJO = 100 # USD/MWh
    COSTO_TERMICO_ALTO = 300 # USD/MWh

    # cambiar a 0 si queremos usar aportes estocásticos
    DETERMINISTICO = 0

    def __init__(self):
        self.N_BINS_VOL = 20
        self.VOL_EDGES = np.linspace(self.V_CLAIRE_MIN, self.V_CLAIRE_MAX, self.N_BINS_VOL + 1)
        self.N_STATES = self.N_BINS_VOL*self.N_HIDRO*(self.T_MAX+1)
        self.N_ACTIONS = 40
        self.Q = np.zeros((self.N_STATES, self.N_ACTIONS))

        self.alpha = 0.001   # learning rate
        self.gamma = 0.99  # discount
        self.min_epsilon = 0.01 # exploración
        self.max_epsilon = 1.0
        self.decay_rate = 0

        # Espacio de observación
        self.observation_space = spaces.Discrete(self.N_STATES)
        
        # Fracción a turbinar del volumen del embalse
        # El agente puede turbinar entre los niveles 0,1,2,3,4 y 5
        self.action_space = spaces.Discrete(self.N_ACTIONS)

        # cargar matriz de aportes discretizada (con estado hidrológico 0,1,2,3,4)
        self.data_matriz_aportes_discreta = leer_archivo(f"Datos\\Claire\\clasificado.csv", sep=",", header=0)
        
        self.aportes_deterministicos = leer_archivo(f"Datos\\MOP\\aportesDeterministicos.csv", sep=",", header=0)

        # cargar matriz de aportes continuos (unidad de los aportes de Claire: m3/s )
        self.data_matriz_aportes_claire = leer_archivo(f"Datos\\Claire\\aporte_claire.csv", sep=",", header=0)
        # convertir a unidad hm3/h
        self.data_matriz_aportes_claire = self.data_matriz_aportes_claire * 3600 / 1e6
        
        # Cargar datos de energías renovables y demanda
        self.data_biomasa = leer_archivo(f"Datos\\MOP\\Deterministicos.xlsx", header=0, sheet_name=0)
        self.data_biomasa = self.data_biomasa.iloc[:,1:]
        self.data_eolico = leer_archivo(f"Datos\\MOP\\Deterministicos.xlsx", header=0, sheet_name=1)
        self.data_eolico = self.data_eolico.iloc[:,1:]
        self.data_solar = leer_archivo(f"Datos\\MOP\\Deterministicos.xlsx", header=0, sheet_name=2)
        self.data_solar = self.data_solar.iloc[:,1:]
        self.data_demanda = leer_archivo(f"Datos\\MOP\\Deterministicos.xlsx", header=0, sheet_name=3)
        self.data_demanda = self.data_demanda.iloc[:,1:]

        # Agregar columna con promedio de crónicas
        self.data_biomasa["PROMEDIO"] = self.data_biomasa.mean(axis=1)
        self.data_eolico["PROMEDIO"] = self.data_eolico.mean(axis=1)
        self.data_solar["PROMEDIO"] = self.data_solar.mean(axis=1)
        self.data_demanda["PROMEDIO"] = self.data_demanda.mean(axis=1)

        # Cargar datos de matrices hidrológicas con las probabilidades de transición entre estados
        self.data_matrices_hidrologicas = leer_archivo(f"Datos\\Claire\\matrices_sem.csv", sep=",", header=0)
        self.data_matrices_hidrologicas = self.data_matrices_hidrologicas.iloc[:, 1:] # Quito la columna de semanas
        self.matrices_hidrologicas = {}
        for i in range(self.data_matrices_hidrologicas.shape[0]):
            array_1d = self.data_matrices_hidrologicas.iloc[i, :].values
            self.matrices_hidrologicas[i] = array_1d.reshape(5, 5) 

        # Inicializar variables internas
        self.reset()

    def reset(self, seed=None, options=None):
        # IMPORTANTE: inicializa el RNG del entorno
        super().reset(seed=seed)

        self.volumen = self.V0
        self.volumen_discreto = discretizar_volumen(self,self.V0)
        self.tiempo = 0
        self.hidrologia = self._inicial_hidrologia()

        info = {
            "volumen_inicial": self.volumen,
            "volumen_discreto_inicial": self.volumen_discreto,
            "hidrologia_inicial": self.hidrologia,
            "tiempo_inicial": self.tiempo
        }
        return self._get_obs(), info
    
    def _inicial_hidrologia(self):
        # retorna el estado inicial del estado hidrológico 0,1,2,3,4
        return np.int64(2)

    def _siguiente_hidrologia(self):
        # retorna el estado hidrológico siguiente 0,1,2,3,4
        clases = np.arange(self.matrices_hidrologicas[self.tiempo % 52].shape[0])
        # USAR el RNG del env (no el global):
        hidrologia_siguiente = self.np_random.choice(
            clases, 
            p=self.matrices_hidrologicas[self.tiempo % 52][self.hidrologia,:]
        )
        return hidrologia_siguiente

    def _aporte(self):
        # guardo fila de estados para la semana actual
        estados_t = self.data_matriz_aportes_discreta.loc[self.tiempo % 52] 

        # guardo las columnas que tienen el eshy actual
        coincidencias = (estados_t == self.hidrologia)
        cronicas_coincidentes = coincidencias[coincidencias].index

        # con las cronicas coincidentes tengo que obtener los aportes para la semana y eshy actual
        aportes = self.data_matriz_aportes_claire.loc[self.tiempo % 52, cronicas_coincidentes] # hm3/h

        # calculo la media de los aportes para la semana y eshy actual
        aportes_promedio = np.mean(aportes) # hm3/h

        rango_valido_inf = aportes_promedio-aportes_promedio*0.05
        rango_valido_sup = aportes_promedio+aportes_promedio*0.05

        # me quedo con los aportes que estén en el promedio +/- 10% 
        aportes_validos = aportes[(aportes>=rango_valido_inf) & (aportes<=rango_valido_sup)] # hm3/h

        # si aportes_validos es vacio tomo como aporte valido el promedio de aportes
        if aportes_validos.empty:
            aporte_final = aportes_promedio
        else:
        # sorteo uniformemente uno de los validos
            aporte_final = self.np_random.choice(aportes_validos)

        # aporte_final = aportes_promedio
        
        valor = self.aportes_deterministicos.iloc[self.tiempo , 0] # hm3/h
       
        if pd.isna(valor):
            valor = 0.0
            print("OJO OJO OJO no encontro valor de aporte determnistico")
            print("paso: ", self.tiempo)
 
        if(self.DETERMINISTICO == 1):    
            return valor
        else:
            return aporte_final * 168

    def _demanda(self):
        # Obtener demanda de energía para el tiempo actual según la cronica sorteada
        energias_demandas = self.data_demanda["PROMEDIO"]
        if self.tiempo < len(energias_demandas):
            return energias_demandas.iloc[self.tiempo]*1.2
        else:
            raise ValueError("Tiempo fuera de rango para datos de demanda")
    
    def _gen_eolico(self):
        # Obtener generación eólica para el tiempo actual según la cronica sorteada
        energias_eolico = self.data_eolico["PROMEDIO"]
        if self.tiempo < len(energias_eolico):
            return energias_eolico.iloc[self.tiempo] 
        else:
            raise ValueError("Tiempo fuera de rango para datos eólicos")

    def _gen_solar(self):
        # Obtener generación solar para el tiempo actual según la cronica sorteada
        energias_solar = self.data_solar["PROMEDIO"]
        if self.tiempo < len(energias_solar):
            return energias_solar.iloc[self.tiempo]
        else:
            raise ValueError("Tiempo fuera de rango para datos solares")

    def _gen_bio(self):
        # Obtener generación de biomasa para el tiempo actual según la cronica sorteada
        energias_biomasa = self.data_biomasa["PROMEDIO"]
        if self.tiempo < len(energias_biomasa):
            return energias_biomasa.iloc[self.tiempo]
        else:
            raise ValueError("Tiempo fuera de rango para datos biomasa")

    def _gen_renovable(self):
        # Generación total de energías renovables no convencionales
        return self._gen_eolico() + self._gen_solar() + self._gen_bio()

    def _gen_termico_bajo(self, demanda_residual):
        if demanda_residual <= self.P_TERMICO_BAJO_MAX * 168:
            return demanda_residual
        else:
            return self.P_TERMICO_BAJO_MAX*168

    def _gen_termico_alto(self, demanda_residual):
        if demanda_residual <= self.P_TERMICO_ALTO_MAX * 168:
            return demanda_residual
        else:
            raise ValueError("Demanda residual excede la capacidad del térmico alto")

    def _despachar(self, v_turb):
        demanda_residual = self._demanda() - self._gen_renovable() - (self.K_CLAIRE * v_turb) # MWh
        energia_termico_bajo = 0 # MWh
        energia_termico_alto = 0 # MWh
        energia_exportada = 0 # MWh

        if demanda_residual > 0:
            # Primero uso termico barato
            energia_termico_bajo = self._gen_termico_bajo(demanda_residual)
            demanda_residual -= energia_termico_bajo

            # Si aún hay demanda residual, uso termico alto
            if demanda_residual > 0:
                energia_termico_alto = self._gen_termico_alto(demanda_residual)
                demanda_residual -= energia_termico_alto

        # Si me queda generación, la exporto
        if demanda_residual < 0:
            energia_exportada = -demanda_residual

        # Retornar ingresos por exportación y costos de generación térmica
        ingreso_exportacion = energia_exportada * self.VALOR_EXPORTACION # USD
        costo_termico = energia_termico_bajo * self.COSTO_TERMICO_BAJO + energia_termico_alto * self.COSTO_TERMICO_ALTO # USD
        return ingreso_exportacion, energia_exportada, costo_termico, energia_termico_bajo, energia_termico_alto
    
    def step(self, action):
        # === Normalizar acción a entero escalar 0..n-1 ===
        if isinstance(action, (np.ndarray, list, tuple)):
            # [0.], [2.0] → 0, 2
            action = int(np.asarray(action).squeeze().item())
        else:
            action = int(action)

        assert self.action_space.contains(action), f"Acción inválida: {action}"

        # Volumen a turbinar

        frac = action /  (self.N_ACTIONS-1)   # mapea a 0.0, 0.25, 0.5, 0.75, 1.0

        accion_turbinado = frac*self.V_CLAIRE_TUR_MAX
        v_turb = min(self.volumen,accion_turbinado) #hm3
        # v_turb = frac * v_max #hm3

        # despacho: e_eolo + e_sol + e_bio + e_termico + e_hidro = dem + exp
        ingreso_exportacion, energia_exportada, costo_termico, energia_termico_bajo, energia_termico_alto = self._despachar(v_turb)

        # recompensa: −costo_termico + ingreso_exportacion
        reward = (-costo_termico + ingreso_exportacion) / 1e6 # MUSD

        info = {
            "volumen_discreto": self.volumen_discreto,
            "hidrologia": self.hidrologia,
            "tiempo": self.tiempo,
            "volumen": self.volumen,
            "turbinado": v_turb,
            "energia_turbinada": v_turb * self.K_CLAIRE,
            "energia_eolica": self._gen_eolico(),
            "energia_solar": self._gen_solar(),
            "energia_biomasa": self._gen_bio(),
            "energia_renovable": self._gen_renovable(),
            "energia_termico_bajo": energia_termico_bajo,
            "energia_termico_alto": energia_termico_alto,
            "energia_exportada": energia_exportada,
            "costo_termico": costo_termico,
            "ingreso_exportacion": ingreso_exportacion,
            "demanda": self._demanda(),
            "demanda_residual": self._demanda() - self._gen_renovable()
        }

        aporte_paso = self._aporte() # hm3 de la semana (volumen)

        v_intermedio = self.volumen - v_turb + aporte_paso
        self.volumen = min(v_intermedio, self.V_CLAIRE_MAX) # hm3
        self.vertimiento = max(v_intermedio - self.V_CLAIRE_MAX, 0) 

        # Actualizar estado hidrologia
        self.hidrologia = self._siguiente_hidrologia()
        # Actualizar estado volumen discreto
        self.volumen_discreto = discretizar_volumen(self,self.volumen)
        # Actualizar estado tiempo
        self.tiempo += 1
        
        info["aportes"] = aporte_paso
        info["vertimiento"] = self.vertimiento

        done = (self.tiempo >= self.T_MAX)
        return self._get_obs(), reward, done, False, info
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"Semana {self.tiempo}:")
            print(f"  Volumen embalse: {self.volumen:.2f}/{self.V_CLAIRE_MAX}")
            print(f"  Estado hidrológico: {self.hidrologia}")
            print(f"  Porcentaje llenado: {(self.volumen/self.V_CLAIRE_MAX)*100:.1f}%")
            print("-" * 30)
        elif mode == 'rgb_array':
            # Retornar una imagen como array numpy para grabación
            pass
        elif mode == 'ansi':
            # Retornar string para mostrar en terminal
            return f"T:{self.tiempo} V:{self.volumen:.1f} H:{self.hidrologia}"
        
    def _get_obs(self):
        # Mapeo de variables internas a observación del agente
        idx = codificar_estados(self.volumen_discreto,self.N_BINS_VOL,self.hidrologia,self.N_HIDRO,self.tiempo)
        
        # Validar contra observation_space
        assert self.observation_space.contains(idx), f"Observación inválida: {idx}. Debe estar en {self.observation_space}"
        return idx

def discretizar_volumen(env, v: float) -> int:
    # Asigna v a un bin en [0..5] usando bordes reales VOL_EDGES
    b = np.digitize([v], env.VOL_EDGES, right=False)[0] - 1
    return int(np.clip(b, 0, env.N_BINS_VOL - 1))
    
def codificar_estados(volumen_discreto,N_BINS_VOL,hidrologia,N_HIDRO,tiempo):
    # representar con un numero entre 0 y idx la tupla (v, h, t)
    idx = volumen_discreto + N_BINS_VOL * (hidrologia + N_HIDRO * tiempo)
    return idx
    
def decodificar_estados(env):
    # es la inversa de codificar_estados_aplanando, devuelve (v, h, t) a partir de idx
    volumen = env.idx % env.N_BINS_VOL
    hidrologia = (env.idx // env.N_BINS_VOL) % env.N_HIDRO
    tiempo    = env.idx // (env.N_BINS_VOL * env.N_HIDRO)
    return (int(volumen), int(hidrologia), int(tiempo))

def politica_optima(Q):
    policy = Q.argmax(axis=1)
    return policy
    
def politica_cubo(env,policy):
    inner_env = env.unwrapped
    policy_cube = policy.reshape(inner_env.T_MAX+1, inner_env.N_HIDRO, inner_env.N_BINS_VOL)  # [t, h, v]
    return policy_cube

def make_env():
    env = HydroThermalEnv()
    env = TimeLimit(env, max_episode_steps=HydroThermalEnv.T_MAX+1)
    return env

def entrenar(env):
    print("Comienzo de entrenamiento...")
    t0 = time.perf_counter()

    # calcular total_timesteps: por ejemplo 5000 episodios * 104 pasos
    total_episodes = 50000

    plotter = LiveRewardPlotter(window=100, refresh_every=20,title="Q-learning: recompensa por episodio")

    for episode in range(total_episodes):

        if episode % 1000 == 0:
            print("Episodio: ", episode)
            
        inner_env = env.unwrapped
        inner_env.reset()
        inner_env.idx = codificar_estados(inner_env.volumen_discreto,inner_env.N_BINS_VOL,inner_env.hidrologia,inner_env.N_HIDRO,inner_env.tiempo)

        done = False
        reward_episodio = 0.0

        # if episode < 10:
        #     epsilon = 0.1  # explorar completamente
        # else:
        #     epsilon = inner_env.min_epsilon + (inner_env.max_epsilon - inner_env.min_epsilon) * np.exp(-inner_env.decay_rate * (episode - 10))

        epsilon = 0.01
        while not done:
            # 1 con probabilidad epsilon
            explorar = np.random.binomial(1,epsilon)

            if explorar == 1:
                a = np.random.randint(inner_env.N_ACTIONS)
            else:
                a = np.argmax(inner_env.Q[inner_env.idx])

            # tomar acción en el ambiente
            _, reward, terminated, truncated, _ = inner_env.step(a)
            done = terminated or truncated

            next_idx = codificar_estados(inner_env.volumen_discreto,inner_env.N_BINS_VOL,inner_env.hidrologia,inner_env.N_HIDRO,inner_env.tiempo)

            # actualización Q-learning
            inner_env.Q[inner_env.idx, a] += inner_env.alpha * (reward + inner_env.gamma * np.max(inner_env.Q[next_idx]) - inner_env.Q[inner_env.idx, a])
            reward_episodio += reward
            inner_env.idx = next_idx
        
        plotter.update(reward_episodio)

    dt = time.perf_counter() - t0
    dt /= 60  # convertir a minutos
    print(f"Entrenamiento completado en {dt:.2f} minutos")    

    plotter.close()
    return inner_env.Q

def evaluar_modelo(eval_env, Q, num_pasos=103, n_eval_episodes=1):
    resultados_todos_episodios = []
    recompensa_total = []

    politica = politica_optima(Q)

    inner_env = eval_env.unwrapped

    print(f"Evaluando durante {n_eval_episodes} episodios...")
    for i in range(n_eval_episodes):
        _, info = inner_env.reset()
        recompensa_episodio = 0
            
        for _ in range(num_pasos + 1):

            action = politica[codificar_estados(inner_env.volumen_discreto,inner_env.N_BINS_VOL,inner_env.hidrologia,inner_env.N_HIDRO,inner_env.tiempo)]
            _, reward, done, _, info = inner_env.step(action)
            resultado_paso = info.copy()
            resultado_paso["action"] = action
            resultado_paso["reward"] = reward
            recompensa_episodio += reward
            resultados_todos_episodios.append(resultado_paso)

            if done:
                break
            
        recompensa_total.append(recompensa_episodio)
        
    # Calcular y mostrar recompensa promedio por episodio
    recompensa_promedio = np.mean(recompensa_total)
    recompensa_std = np.std(recompensa_total)
    print(f"Recompensa promedio por episodio: {recompensa_promedio:.2f} +/- {recompensa_std:.2f}")

    # Convertir todo a un único DataFrame
    df_all = pd.DataFrame(resultados_todos_episodios)

    # Calcular el promedio por paso de tiempo
    df_avg = df_all.groupby("tiempo").mean().reset_index()
                
    return df_avg, df_all

def guardar_trayectorias(fecha_hora, df_trayectorias, output_dir="figures"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig_fecha_hora = os.path.join(output_dir, f"resultados_{fecha_hora}")
    if not os.path.exists(fig_fecha_hora):
        os.makedirs(fig_fecha_hora)

    df_trayectorias_copy = df_trayectorias.copy()
    tiempos = df_trayectorias_copy.pop("tiempo")

    for col in df_trayectorias_copy.columns:
        fig, ax = plt.subplots()
        ax.plot(tiempos, df_trayectorias_copy[col], marker='o')
        ax.set_ylabel(col)
        ax.set_xlabel("Semanas")
        ax.grid(True)
        nombre_figura = f"{col}.png"
        fig.savefig(os.path.join(fig_fecha_hora, nombre_figura))
        plt.close(fig)

if __name__ == "__main__":
    start_time = time.time()
    fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    carpeta = os.path.join("salidas", f"resultados_{fecha_hora}")
    # Crear la carpeta (si no existe)
    os.makedirs(carpeta, exist_ok=True)

    resultados_promedio = os.path.join(carpeta,"promedios")
    os.makedirs(resultados_promedio, exist_ok=True)

    EVAL_CSV_PATH = os.path.join(resultados_promedio,"trayectorias.csv")
    EVAL_CSV_ENERGIAS_PATH = os.path.join(resultados_promedio,"energias.csv")
    EVAL_CSV_ESTADOS_PATH = os.path.join(resultados_promedio,"estados.csv")
    EVAL_CSV_RESULTADOS_AGENTE_PATH = os.path.join(resultados_promedio,"resultados_agente.csv")
    EVAL_CSV_COSTOS_PATH = os.path.join(resultados_promedio,"costos.csv")

    eval_env = make_env()
    inner_env = eval_env.unwrapped

    # Verificar si existe la tabla Q
    if os.path.exists("Q_table.npy"):
        try:
            print(f"Cargando tabla Q desde Q_table.npy...")
            Q = np.load("Q_table.npy")
            print("Tabla Q cargada exitosamente.")
        except Exception as e:
            print(f"Error al cargar la tabla Q: {e}")
            print("Entrenando Q-learning de nuevo...")
            Q = entrenar(eval_env)
    else:
        print("Archivo de tabla Q no encontrado, entrenando uno nuevo...")        
        Q = entrenar(eval_env)
    
    # Evaluar el modelo
    print("Iniciando evaluación del modelo...")
    eval_env.reset(seed=123)
    df_eval, df_all = evaluar_modelo(eval_env,Q)
    df_eval["reward_usd"] = df_eval["reward"] * 1e6
    guardar_trayectorias(fecha_hora,df_eval)

    # Graficar mapa de calor de la tabla Q obtenida después del entrenamiento
    plt.figure(figsize=(8, 10))
    plt.imshow(Q, aspect='auto')
    plt.colorbar(label="Q(s,a)")
    plt.xlabel("Acciones (0..4)")
    plt.ylabel("Estados (idx 0..3119)")
    plt.title("Q-table completa (estados x acciones)")
    plt.tight_layout()
    plt.show()

    # Obtener politica optima aplanada
    politica = politica_optima(Q) # array de shape (3120,)

    num_pasos = 103  

    # Lista para guardar los DataFrames
    dfs_escenarios = [df_all.iloc[i*num_pasos:(i+1)*num_pasos].reset_index(drop=True) for i in range(100)]

    # --- Guardar tabla Q ---
    ruta_q = os.path.join(carpeta, "Q_table.npy")
    np.save(ruta_q, Q)

    for i in range(len(dfs_escenarios)):
        df_escenario = dfs_escenarios[i]
        # Crear nombre con fecha y hora actual
        ruta_csv = os.path.join(carpeta, f"escenario_{i}.csv")
        df_escenario.to_csv(ruta_csv, index=False)
    
    # Guardar y visualizar los resultados de la evaluación 
    df_eval.to_csv(EVAL_CSV_PATH, index=False)
    print(f"Resultados de la evaluación guardados en {EVAL_CSV_PATH}")

    # Guardar energias en un mismo csv
    df_energias = df_eval.loc[:, ["energia_turbinada", "energia_eolica", "energia_solar", "energia_biomasa", "energia_renovable", "energia_termico_bajo", "energia_termico_alto", "demanda", "demanda_residual"]]
    df_energias.to_csv(EVAL_CSV_ENERGIAS_PATH, index=False)
    print(f"Resultados de energia guardados en {EVAL_CSV_ENERGIAS_PATH}")

    # Guardar variables de estado en un mismo csv
    df_estados = df_eval.loc[:, ["volumen_discreto","volumen", "hidrologia", "tiempo", "aportes", "vertimiento", "turbinado"]]
    df_estados.to_csv(EVAL_CSV_ESTADOS_PATH, index=False)
    print(f"Resultados de variables de estado guardados en {EVAL_CSV_ESTADOS_PATH}")

    # Guardar energias en un mismo csv
    df_resultados_agente = df_eval.loc[:, ["action", "reward"]]
    df_resultados_agente.to_csv(EVAL_CSV_RESULTADOS_AGENTE_PATH, index=False)
    print(f"Resultados del agente guardados en {EVAL_CSV_RESULTADOS_AGENTE_PATH}")

    # Guardar energias en un mismo csv
    df_costos = df_eval.loc[:, ["costo_termico", "ingreso_exportacion"]]
    df_costos.to_csv(EVAL_CSV_COSTOS_PATH, index=False)
    print(f"Resultados de costos guardados en {EVAL_CSV_COSTOS_PATH}")

    total_reward = df_eval["reward"].sum()
    print(f"Recompensa total en evaluación: {total_reward:.2f}")

    end_time = time.time()
    execution_time_seconds = end_time - start_time
    execution_time_minutes = execution_time_seconds / 60
    print(f"Tiempo de ejecución de main: {execution_time_minutes:.2f} minutos")
