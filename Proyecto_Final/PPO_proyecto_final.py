# type: ignore
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

import os
import time
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
# Seleccionar backend ANTES de importar pyplot
import matplotlib
from datetime import datetime
try:
    matplotlib.use("TkAgg")
except Exception as e:
    print(f"No se pudo activar TkAgg: {e}")
import matplotlib.pyplot as plt
plt.ion()
print(f"Matplotlib backend: {matplotlib.get_backend()}")

class LivePlotCallback(BaseCallback):
    def __init__(self, verbose=0, plot_every=20):
        super().__init__(verbose)
        self.plot_every = plot_every

        self.episode_rewards = []
        self.moving_avg_rewards = []

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlabel("Episodio")
        self.ax.set_ylabel("Recompensa por episodio")
        self.ax.grid(True)

        self.line, = self.ax.plot([], [], lw=1, label="Reward")
        self.line_avg, = self.ax.plot([], [], lw=2, label="Moving Avg (100)")

        self.ax.legend()
        # Mostrar ventana sin bloquear y asegurar primer draw
        plt.show(block=False)
        self.fig.canvas.draw()
        plt.pause(0.001)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                r = info["episode"]["r"]
                self.episode_rewards.append(r)

                window = 100
                if len(self.episode_rewards) >= window:
                    avg = np.mean(self.episode_rewards[-window:])
                else:
                    avg = np.mean(self.episode_rewards)
                self.moving_avg_rewards.append(avg)

                # Actualiza datos
                x = list(range(len(self.episode_rewards)))
                y = self.episode_rewards
                self.line.set_data(x, y)
                self.line_avg.set_data(x, self.moving_avg_rewards)

                # Ajusta ejes
                self.ax.relim()
                self.ax.autoscale_view()

                # Dibuja y procesa eventos GUI cada 'plot_every' episodios
                if len(self.episode_rewards) % self.plot_every == 0:
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                    plt.pause(0.001)

        return True
    
    def _on_training_end(self) -> None:
        # Desactiva el modo interactivo y muestra block hasta que se cierre la ventana
        plt.ioff()
        plt.show()

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
    V_CLAIRE_MAX = 60000 # hm3
    V0 = V_CLAIRE_MAX * 0.75 # hm3
    
    K_CLAIRE = P_CLAIRE_MAX / Q_CLAIRE_MAX # MWh/hm3

    V_CLAIRE_TUR_MAX = P_CLAIRE_MAX * 168 / K_CLAIRE # hm3

    VALOR_EXPORTACION = 0.0 # USD/MWh 
    COSTO_TERMICO_BAJO = 100.0 # USD/MWh 
    COSTO_TERMICO_ALTO = 300.0 # USD/MWh
    COSTO_VERTIMIENTO = 30.0 # USD/MWh

    # cambiar a 0 si queremos usar aportes estocásticos
    DETERMINISTICO = 1

    def __init__(self):
        # Espacio de observación
        self.observation_space = spaces.Dict({
            "volumen": spaces.Box(self.V_CLAIRE_MIN, self.V_CLAIRE_MAX, shape=(), dtype=np.float32),
            "hidrologia": spaces.Discrete(self.N_HIDRO, start=0),
            "tiempo": spaces.Discrete(self.T_MAX + 1, start=0)
        })
        
        # Espacio de acción continuo: fracción de turbinado en [0, 1]
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)

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
        self.tiempo = 0
        self.hidrologia = self._inicial_hidrologia()
        self.hidrologia_anterior = self.hidrologia

        info = {
            "volumen_inicial": self.volumen,
            "hidrologia_inicial": self.hidrologia,
            "tiempo_inicial": self.tiempo
        }
        return self._get_obs(), info
    
    def _inicial_hidrologia(self):
        # retorna el estado inicial del estado hidrológico 0,1,2,3,4
        return np.int64(2)

    def _siguiente_hidrologia(self):
        # retorna el estado hidrológico siguiente 0,1,2,3,4
        self.hidrologia_anterior = self.hidrologia
        # array con las clases 0,1,2,3,4
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
        
        valor = self.aportes_deterministicos.iloc[self.tiempo , 0] # hm3/semana
       
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
            return energias_demandas.iloc[self.tiempo] * 1.2
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
            return self.P_TERMICO_BAJO_MAX * 168

    def _gen_termico_alto(self, demanda_residual):
        if demanda_residual <= self.P_TERMICO_ALTO_MAX * 168:
            return demanda_residual
        else:
            raise ValueError("Demanda residual excede la capacidad del térmico alto")

    def _despachar(self, qt):
        # Demanda residual
        demanda_residual = self._demanda() - self._gen_renovable()# MWh

        # Energia hidro
        energia_hidro = self.K_CLAIRE * qt # MWh
        demanda_residual -= energia_hidro
        
        # Termico bajo
        energia_termico_bajo = self._gen_termico_bajo(max(demanda_residual, 0.0)) # MWh
        demanda_residual -= energia_termico_bajo
        
        # Termico alto
        energia_termico_alto = 0.0
        if demanda_residual > 0.0:
            energia_termico_alto = self._gen_termico_alto(demanda_residual) # MWh
            demanda_residual -= energia_termico_alto

        # Energia exportada
        energia_exportada = max(-demanda_residual, 0.0) # MWh

        # Ingreso por exportación
        ingreso_exportacion = energia_exportada * self.VALOR_EXPORTACION # USD
        # Costo de energía termica
        costo_termico = (energia_termico_bajo * self.COSTO_TERMICO_BAJO + 
                         energia_termico_alto * self.COSTO_TERMICO_ALTO) # USD
        
        return ingreso_exportacion, costo_termico, energia_exportada, energia_termico_bajo, energia_termico_alto, energia_hidro

    def step(self, action):
        # Validar que la acción esté en el espacio válido
        action = np.array(action, dtype=np.float32).reshape(1,)
        assert self.action_space.contains(action), f"Acción inválida: {action}. Debe estar en {self.action_space}"
        frac = float(action[0])

        # Demanda residual considerando solo renovables (sin hidro)
        demanda_residual_pre = max(self._demanda() - self._gen_renovable(), 0.0)  # MWh

        # Máximo físico semanal (por potencia y por volumen) en hm3
        qt_max_fisico = min(frac * self.V_CLAIRE_TUR_MAX, self.volumen)

        # Limita la hidro a no exceder la demanda residual (evita exportar con hidro)
        energia_hidro_max_frac = self.K_CLAIRE * qt_max_fisico  # MWh
        energia_hidro_obj = min(energia_hidro_max_frac, demanda_residual_pre)  # MWh
        qt = energia_hidro_obj / self.K_CLAIRE  # hm3

        # despacho: e_eolo + e_sol + e_bio + e_termico + e_hidro = dem + exp
        ingreso_exportacion, costo_termico, energia_exportada, energia_termico_bajo, energia_termico_alto, energia_hidro = self._despachar(qt)

        info = {
            "volumen": self.volumen,
            "hidrologia": self.hidrologia,
            "tiempo": self.tiempo,
            "volumen_turbinado": qt,
            "energia_hidro": energia_hidro,
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
            "demanda_residual": self._demanda() - self._gen_renovable(),
            "fraccion_turbinado": frac,

            # --- métricas de diagnóstico ---
            "qt_max_fisico": qt_max_fisico,
            "energia_hidro_max_frac": energia_hidro_max_frac,
            "energia_hidro_obj": energia_hidro_obj,
        }
        
        # Actualizar variables internas
        self.hidrologia = self._siguiente_hidrologia()
        aporte_paso = self._aporte() # hm3 de la semana (volumen)
        v_intermedio = self.volumen - qt + aporte_paso
        self.vertimiento = max(v_intermedio - self.V_CLAIRE_MAX, 0) 
        self.volumen = min(v_intermedio, self.V_CLAIRE_MAX) # hm3
        self.tiempo += 1
        
        info["aportes"] = aporte_paso
        info["vertimiento"] = self.vertimiento

        # Costo por vertimiento
        energia_vertida = self.vertimiento * self.K_CLAIRE
        costo_vertimiento = energia_vertida * self.COSTO_VERTIMIENTO # USD

        # Recompensa
        reward_usd = - costo_termico - costo_vertimiento  # USD
        reward = reward_usd / 1e6  # escalar a MUSD

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
        obs = {
            "volumen": np.array(self.volumen, dtype=np.float32),
            "hidrologia": int(self.hidrologia),
            "tiempo": int(self.tiempo)
        }
        
        # Validar contra observation_space (opcional, útil para debug)
        assert self.observation_space.contains(obs), f"Observación inválida: {obs}. Debe estar en {self.observation_space}"
        return obs
    
class OneHotFlattenObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 1, n_hidro para one-hot de hidrología, T_MAX + 1 para one-hot de tiempo
        dim = 1 + HydroThermalEnv.N_HIDRO + HydroThermalEnv.T_MAX + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)

    def observation(self, obs):
        # Normalizar volumen
        v = float(obs["volumen"])
        v_norm = v / HydroThermalEnv.V_CLAIRE_MAX

        # One-Hot encoding Hidrologia
        h = obs["hidrologia"]
        hidro_oh = np.zeros(HydroThermalEnv.N_HIDRO, dtype=np.float32) 
        hidro_oh[h] = 1.0

        # One-Hot encoding Tiempo
        semana = obs["tiempo"]
        time_oh = np.zeros(HydroThermalEnv.T_MAX + 1, dtype=np.float32)
        time_oh[semana] = 1.0

        obs_res = np.concatenate(([v_norm], hidro_oh, time_oh), axis=0)
        return obs_res

def make_env():
    env = HydroThermalEnv()
    env = OneHotFlattenObs(env)
    env = TimeLimit(env, max_episode_steps=HydroThermalEnv.T_MAX+1)
    return env

def entrenar():
    print("Comienzo de entrenamiento...")
    t0 = time.perf_counter()
    # vectorizado de entrenamiento (usar DummyVecEnv en Windows para evitar sobrecarga de procesos)
    n_envs = 8
    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)  # <<< normaliza

    # Definir una arquitectura de red más grande
    policy_kwargs = dict(
        lstm_hidden_size=128,
        n_lstm_layers=1,
        net_arch=dict(pi=[128], vf=[128]),
    )

    model = RecurrentPPO(
        MlpLstmPolicy,
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=52,         # 1/2 episodio por actualización
        gamma=0.999,         # mira mas lejos
        ent_coef=0.005,      # evita colapso temprano a extremos
        learning_rate=3e-4,
        device="auto"       # usa GPU si hay
    )

    # calcular total_timesteps: por ejemplo 2000 episodios * 104 pasos
    total_episodes = 2_000
    total_timesteps = total_episodes * (HydroThermalEnv.T_MAX + 1)

    callback = LivePlotCallback(plot_every=1)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("RecurrentPPO_hydro_thermal_claire_continuous")
    vec_env.save("vecnorm.pkl")  # <<< guarda stats

    dt = time.perf_counter() - t0
    dt /= 60  # convertir a minutos
    print(f"Entrenamiento completado en {dt:.2f} minutos")

def cargar_o_entrenar_modelo(model_path):
    # Verificar si el archivo del modelo existe
    if os.path.exists(f"{model_path}.zip"):
        try:
            print(f"Cargando modelo desde {model_path}...")
            model = RecurrentPPO.load(model_path)
            print("Modelo cargado exitosamente.")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            print("Entrenando un modelo nuevo...")
            entrenar()
            model = RecurrentPPO.load(model_path)
    else:
        print("Archivo del modelo no encontrado, entrenando uno nuevo...")
        entrenar()
        model = RecurrentPPO.load(model_path)

    return model

def evaluar_modelo(model, eval_env, num_pasos=51, n_eval_episodes=100):
    resultados_todos_episodios = []
    recompensa_total = []

    print(f"Evaluando durante {n_eval_episodes} episodios...")
    n_envs = getattr(eval_env, "num_envs", 1)

    for i in range(n_eval_episodes):
        obs = eval_env.reset()
        recompensa_episodio = 0
        state, episode_start = None, np.ones((n_envs,), dtype=bool)
        
        for _ in range(num_pasos + 1):
            action, state = model.predict(
                obs, 
                state=state, 
                episode_start=episode_start, 
                deterministic=True
            )
            obs, rewards, dones, infos = eval_env.step(action)

            # Extrae el primer env (usas n_envs=1)
            a = int(np.asarray(action).reshape(-1)[0])
            r = float(np.asarray(rewards).reshape(-1)[0])
            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            
            resultado_paso = info.copy()
            resultado_paso["action"] = a
            resultado_paso["reward"] = r
            resultados_todos_episodios.append(resultado_paso)
            recompensa_episodio += r

            episode_start = dones
            if np.any(dones):
                break
        
        recompensa_total.append(recompensa_episodio)
    
    # Calcular y mostrar recompensa promedio por episodio
    recompensa_promedio = np.mean(recompensa_total)
    recompensa_std = np.std(recompensa_total)
    print(f"Recompensa promedio por episodio: {recompensa_promedio:.2f} +/- {recompensa_std:.2f}")

    # Convertir todo a un único DataFrame
    df_all = pd.DataFrame(resultados_todos_episodios)

    # Calcular el promedio por paso de tiempo
    df_avg = df_all.groupby("tiempo").mean(numeric_only=True).reset_index()
            
    return df_avg, df_all

def guardar_trayectorias(fecha_hora, df_trayectorias, output_dir="figures"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig_fecha_hora = os.path.join(output_dir, f"resultados_{fecha_hora}")
    if not os.path.exists(fig_fecha_hora):
        os.makedirs(fig_fecha_hora)

    # Función auxiliar para “aplanar” valores a escalares cuando sea posible
    def to_scalar(v):
        if isinstance(v, (np.generic,)):  # np.float64, np.int64, etc.
            return v.item()
        if isinstance(v, np.ndarray):
            if v.ndim == 0:
                return v.item()
            if v.size == 1:
                return v.reshape(()).item()
            return np.nan  # secuencia no-escalar -> no ploteable
        if isinstance(v, (list, tuple)):
            return v[0] if len(v) == 1 else np.nan
        return v

    # Coaccionar celdas potencialmente problemáticas y quedarnos con numéricas
    df_clean = df_trayectorias.copy()
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].map(to_scalar)

    # Solo columnas numéricas
    df_num = df_clean.select_dtypes(include=[np.number]).copy()

    if "tiempo" not in df_num.columns:
        print("Columna 'tiempo' no está disponible como numérica; no se pueden graficar trayectorias.")
        return

    tiempos = df_num.pop("tiempo")

    # Graficar solo columnas numéricas (omitimos las que queden vacías o NaN)
    for col in df_num.columns:
        serie = pd.to_numeric(df_num[col], errors="coerce")
        if serie.isna().all():
            print(f"Saltando columna no numérica o inválida: {col}")
            continue
        try:
            fig, ax = plt.subplots()
            ax.plot(tiempos, serie, marker='o')
            ax.set_ylabel(col)
            ax.set_xlabel("Semanas")
            ax.grid(True)
            nombre_figura = f"{col}.png"
            fig.savefig(os.path.join(fig_fecha_hora, nombre_figura))
            plt.close(fig)
        except Exception as e:
            print(f"No se pudo graficar columna {col}: {e}")

def graficar_resumen_evaluacion(df_eval):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Acciones
    ax1.plot(df_eval["tiempo"], df_eval["action"], marker='o', linestyle='-', color='tab:blue')
    ax1.set_xlabel("Paso (Semana)")
    ax1.set_ylabel("Acción (Fracción a turbinar)")
    ax1.set_title("Acciones durante la Evaluación")
    ax1.grid(True)

    # Recompensas
    ax2.plot(df_eval["tiempo"], df_eval["reward"], marker='o', linestyle='-', color='tab:green')
    ax2.set_xlabel("Paso (Semana)")
    ax2.set_ylabel("Recompensa")
    ax2.set_title("Recompensas durante la Evaluación")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    start_time = time.time()

    fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    carpeta = os.path.join("salidas", f"resultados_{fecha_hora}")
    # Crear la carpeta (si no existe)
    os.makedirs(carpeta, exist_ok=True)

    resultados_promedio = os.path.join(carpeta,"promedios")
    os.makedirs(resultados_promedio, exist_ok=True)

    MODEL_PATH = "RecurrentPPO_hydro_thermal_claire_continuous"
    EVAL_CSV_PATH = os.path.join(resultados_promedio,"trayectorias.csv")
    EVAL_CSV_ENERGIAS_PATH = os.path.join(resultados_promedio,"energias.csv")
    EVAL_CSV_ESTADOS_PATH = os.path.join(resultados_promedio,"estados.csv")
    EVAL_CSV_RESULTADOS_AGENTE_PATH = os.path.join(resultados_promedio,"resultados_agente.csv")
    EVAL_CSV_COSTOS_PATH = os.path.join(resultados_promedio,"costos.csv")

    # Cargar o entrenar el modelo
    model = cargar_o_entrenar_modelo(MODEL_PATH)

    # Evaluar el modelo
    print("Iniciando evaluación del modelo...")
    eval_env = DummyVecEnv([make_env])
    try:
        eval_env = VecNormalize.load("vecnorm.pkl", eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
        print("VecNormalize cargado para evaluación.")
    except Exception as e:
        print(f"No se pudo cargar VecNormalize: {e}")

    eval_env.reset()
    df_eval, df_all = evaluar_modelo(model, eval_env, num_pasos=103, n_eval_episodes=100)
    df_eval["reward_usd"] = df_eval["reward"] * 1e6

    num_pasos = 103  

    # Lista para guardar los DataFrames
    dfs_escenarios = [df_all.iloc[i*num_pasos:(i+1)*num_pasos].reset_index(drop=True) for i in range(100)]

    for i in range(len(dfs_escenarios)):
        df_escenario = dfs_escenarios[i]
        # Crear nombre con fecha y hora actual
        ruta_csv = os.path.join(carpeta, f"escenario_{i}.csv")
        df_escenario.to_csv(ruta_csv, index=False)

    # Guardar y visualizar los resultados de la evaluación 
    df_eval.to_csv(EVAL_CSV_PATH, index=False)
    print(f"Resultados de la evaluación guardados en {EVAL_CSV_PATH}")

    # Guardar energias en un mismo csv
    df_energias = df_eval.loc[:, ["energia_hidro", "energia_eolica", "energia_solar", "energia_biomasa", "energia_renovable", "energia_termico_bajo", "energia_termico_alto", "demanda", "demanda_residual"]]
    df_energias.to_csv(EVAL_CSV_ENERGIAS_PATH, index=False)
    print(f"Resultados de energia guardados en {EVAL_CSV_ENERGIAS_PATH}")

    # Guardar variables de estado en un mismo csv
    df_estados = df_eval.loc[:, ["volumen", "hidrologia", "tiempo", "aportes", "vertimiento", "volumen_turbinado"]]
    df_estados.to_csv(EVAL_CSV_ESTADOS_PATH, index=False)
    print(f"Resultados de variables de estado guardados en {EVAL_CSV_ESTADOS_PATH}")

    # Guardar energias en un mismo csv
    df_resultados_agente = df_eval.loc[:, ["action", "fraccion_turbinado", "reward"]]
    df_resultados_agente.to_csv(EVAL_CSV_RESULTADOS_AGENTE_PATH, index=False)
    print(f"Resultados del agente guardados en {EVAL_CSV_RESULTADOS_AGENTE_PATH}")

    # Guardar energias en un mismo csv
    df_costos = df_eval.loc[:, ["costo_termico", "ingreso_exportacion"]]
    df_costos.to_csv(EVAL_CSV_COSTOS_PATH, index=False)
    print(f"Resultados de costos guardados en {EVAL_CSV_COSTOS_PATH}")

    total_reward = df_eval["reward"].sum()
    print(f"Recompensa total en evaluación: {total_reward:.2f}")

    # Guardar gráficos de cada variable de la trayectoria
    guardar_trayectorias(fecha_hora,df_eval)
    print("Gráficos de trayectoria guardados en la carpeta 'figures'.")

    end_time = time.time()
    execution_time_seconds = end_time - start_time
    execution_time_minutes = execution_time_seconds / 60
    print(f"Tiempo de ejecución de main: {execution_time_minutes:.2f} minutos")

    # Mostrar gráfico resumen de acciones y recompensas
    graficar_resumen_evaluacion(df_eval)
