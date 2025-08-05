from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
import os
from pprint import pprint

# to-do: Rodrigo aconsejo usar actorCritic (con one-hot encoding para las variables discretas) para las acciones continuas, si no converge discretizar el volumen del turbinado (ejemplo 10 niveles) y usar metodos tabulares (QLearning)

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

    P_CLAIRE_MAX = 1541
    P_SOLAR_MAX = 254
    P_EOLICO_MAX = 1584.7
    P_BIOMASA_MAX = 487.3
    # to-do: revisar si estos valores son correctos
    P_TERMICO_BAJO_MAX = 10000
    P_TERMICO_ALTO_MAX = np.inf

    Q_CLAIRE_MAX = 7081

    V_CLAIRE_MIN = 0
    V_CLAIRE_MAX = 11000
    V0 = V_CLAIRE_MAX / 2

    K_CLAIRE = P_CLAIRE_MAX / Q_CLAIRE_MAX

    # to-do: revisar si estos valores son correctos
    VALOR_EXPORTACION = 12.5  
    COSTO_TERMICO_BAJO = 100  
    COSTO_TERMICO_ALTO = 200  

    def __init__(self):
        # Espacio de observación
        self.observation_space = spaces.Dict({
            "volumen": spaces.Box(self.V_CLAIRE_MIN, self.V_CLAIRE_MAX, shape=(), dtype=np.float32),
            "hidrologia": spaces.Discrete(self.N_HIDRO, start=0),
            "tiempo": spaces.Discrete(self.T_MAX + 1, start=0)
        })
        
        # Fracción a turbinar del volumen del embalse
        # El agente puede turbinar entre 0 y 1 (100% del volumen)
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)

        # cargar matriz de aportes discretizada (con estado hidrológico 0,1,2,3,4)
        self.data_matriz_aportes_discreta = leer_archivo(f"Datos\\Claire\\clasificado.csv", sep=",", header=0)
        
        # cargar matriz de aportes continuos
        self.data_matriz_aportes_claire = leer_archivo(f"Datos\\Claire\\aporte_claire.csv", sep=",", header=0)
        
        # Cargar datos de energías renovables y demanda
        self.data_biomasa = leer_archivo(f"Datos\\MOP\\Deterministicos.xlsx", header=0, sheet_name=0)
        self.data_biomasa = self.data_biomasa.iloc[:,1:]
        self.data_eolico = leer_archivo(f"Datos\\MOP\\Deterministicos.xlsx", header=0, sheet_name=1)
        self.data_eolico = self.data_eolico.iloc[:,1:]
        self.data_solar = leer_archivo(f"Datos\\MOP\\Deterministicos.xlsx", header=0, sheet_name=2)
        self.data_solar = self.data_solar.iloc[:,1:]
        self.data_demanda = leer_archivo(f"Datos\\MOP\\Deterministicos.xlsx", header=0, sheet_name=3)
        self.data_demanda = self.data_demanda.iloc[:,1:]

        # Cargar datos de matrices hidrológicas con las probabilidades de transición entre estados
        self.data_matrices_hidrologicas = leer_archivo(f"Datos\\Claire\\matrices_sem.csv", sep=",", header=0)
        self.data_matrices_hidrologicas = self.data_matrices_hidrologicas.iloc[:, 1:] # Quito la columna de semanas
        self.matrices_hidrologicas = {}
        for i in range(self.data_matrices_hidrologicas.shape[0]):
            array_1d = self.data_matrices_hidrologicas.iloc[i, :].values
            self.matrices_hidrologicas[i] = array_1d.reshape(5, 5) # type: ignore

        # Inicializar variables internas
        self.reset(seed=42)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.volumen = self.V0
        self.tiempo = 0
        self.hidrologia = self._inicial_hidrologia()
        
        info = {
            "volumen_inicial": self.volumen,
            "hidrologia_inicial": self.hidrologia,
            "tiempo_inicial": self.tiempo
        }
        return self._get_obs(), info

    def _sortear_cronica_inicial(self):
        return np.random.randint(self.data_matriz_aportes_discreta.shape[1])
    
    def _inicial_hidrologia(self):
        # retorna el estado inicial del estado hidrológico 0,1,2,3,4
        self.cronica = self._sortear_cronica_inicial()
        h0 = self.data_matriz_aportes_discreta.iloc[self.T0, self.cronica]
        return int(h0) # type: ignore

    def _siguiente_hidrologia(self):
        # retorna el estado hidrológico siguiente 0,1,2,3,4
        self.hidrologia_anterior = self.hidrologia
        self.clases_hidrologia = np.arange(self.matrices_hidrologicas[self.tiempo % 52].shape[0]) # array con las clases 0,1,2,3,4
        hidrologia_siguiente = np.random.choice(self.clases_hidrologia, p=self.matrices_hidrologicas[self.tiempo % 52][self.hidrologia,:])
        return hidrologia_siguiente

    def _rotar_fila(self, fila: pd.Series):
        # rota a la izquierda: s = [x0, x1, x2, ...] -> [x1, x2, ..., x0]
        valores = fila.tolist()
        if len(valores) == 0:
            return fila.copy()
        rotada = valores[1:] + [valores[0]]
        return pd.Series(rotada, index=fila.index)

    def _aportes(self):
        # dados dos estados (inicial y final) y dos semanas correspondientes a esos estados, 
        # sorteo una ocurrencia de aportes para el lago claire
        estados_ini = self.data_matriz_aportes_discreta.loc[self.tiempo % 52] 

        if self.tiempo == 51 or self.tiempo == 103:
            estados_fin = self._rotar_fila(self.data_matriz_aportes_discreta.loc[0])  # type: ignore
        else:
            estados_fin = self.data_matriz_aportes_discreta.loc[(self.tiempo + 1) % 52] 

        coincidencias = (estados_ini == self.hidrologia_anterior) & (estados_fin == self.hidrologia)
        columnas_validas = self.data_matriz_aportes_discreta.columns[coincidencias] # type: ignore

        if len(columnas_validas) == 0:
            raise ValueError("No hay coincidencias válidas para los estados hidrológicos actuales")
            
        año_sorteado = np.random.choice(columnas_validas)

        # Obtener valor en claire para fila2 y ese año
        valor_claire = self.data_matriz_aportes_claire.loc[self.tiempo % 52, año_sorteado]
        return valor_claire
    
    def _demanda(self):
        # Obtener demanda de energía para el tiempo actual según la cronica sorteada
        energias_demandas = self.data_demanda.iloc[:,self.cronica]
        if self.tiempo < len(energias_demandas):
            return energias_demandas.iloc[self.tiempo]
        else:
            raise ValueError("Tiempo fuera de rango para datos de demanda")
    
    def _gen_eolico(self):
        # Obtener generación eólica para el tiempo actual según la cronica sorteada
        energias_eolico = self.data_eolico.iloc[:,self.cronica]
        if self.tiempo < len(energias_eolico):
            return energias_eolico.iloc[self.tiempo]
        else:
            raise ValueError("Tiempo fuera de rango para datos eólicos")

    def _gen_solar(self):
        # Obtener generación solar para el tiempo actual según la cronica sorteada
        energias_solar = self.data_solar.iloc[:,self.cronica]
        if self.tiempo < len(energias_solar):
            return energias_solar.iloc[self.tiempo]
        else:
            raise ValueError("Tiempo fuera de rango para datos solares")

    def _gen_bio(self):
        # Obtener generación de biomasa para el tiempo actual según la cronica sorteada
        energias_biomasa = self.data_biomasa.iloc[:,self.cronica]
        if self.tiempo < len(energias_biomasa):
            return energias_biomasa.iloc[self.tiempo]
        else:
            raise ValueError("Tiempo fuera de rango para datos biomasa")

    def _gen_renovable(self):
        # Generación total de energías renovables no convencionales
        return self._gen_eolico() + self._gen_solar() + self._gen_bio()

    def _gen_termico_bajo(self, demanda_residual):
        if demanda_residual <= self.P_TERMICO_BAJO_MAX:
            return demanda_residual
        else:
            return self.P_TERMICO_BAJO_MAX

    def _gen_termico_alto(self, demanda_residual):
        if demanda_residual <= self.P_TERMICO_ALTO_MAX:
            return demanda_residual
        else:
            raise ValueError("Demanda residual excede la capacidad del térmico alto")

    def _despachar(self, qt):
        demanda_residual = self._demanda() - self._gen_renovable() - (self.K_CLAIRE * qt)
        energia_termico_bajo = 0
        energia_termico_alto = 0
        exportacion = 0

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
            exportacion = -demanda_residual

        # Retornar ingresos por exportación y costos de generación térmica
        ingreso_exportacion = exportacion * self.VALOR_EXPORTACION
        costo_termico = energia_termico_bajo * self.COSTO_TERMICO_BAJO + energia_termico_alto * self.COSTO_TERMICO_ALTO
        return ingreso_exportacion, costo_termico, energia_termico_bajo, energia_termico_alto
    
    def step(self, action):
        # Validar que la acción esté en el espacio válido
        action = np.array(action, dtype=np.float32).reshape(1,)
        assert self.action_space.contains(action), f"Acción inválida: {action}. Debe estar en {self.action_space}"

        # Volumen a turbinar
        frac = float(action[0])
        qt = frac * self.volumen

        # despacho: e_eolo + e_sol + e_bio + e_termico + e_hidro = dem + exp
        ingreso_exportacion, costo_termico, energia_termico_bajo, energia_termico_alto = self._despachar(qt)

        # recompensa: −costo_termico + ingreso_exportacion
        reward = -costo_termico + ingreso_exportacion

        info = {
            "volumen": self.volumen,
            "hidrologia": self.hidrologia,
            "tiempo": self.tiempo,
            "turbinado": qt,
            "energia_turbinada": qt * self.K_CLAIRE,
            "energia_eolica": self._gen_eolico(),
            "energia_solar": self._gen_solar(),
            "energia_biomasa": self._gen_bio(),
            "energia_renovable": self._gen_renovable(),
            "energia_termico_bajo": energia_termico_bajo,
            "energia_termico_alto": energia_termico_alto,
            "ingreso_exportacion": ingreso_exportacion,
            "costo_termico": costo_termico,
        }

        # dinámica: v ← v − q − d + a
        self.hidrologia = self._siguiente_hidrologia()
        aportes = self._aportes()
        self.volumen = min(self.volumen - qt + aportes, self.V_CLAIRE_MAX) # type: ignore
        self.tiempo += 1
        
        info["aportes"] = aportes
        info["volumen_siguiente"] = self.volumen
        info["hidrologia_siguiente"] = self.hidrologia
        info["tiempo_siguiente"] = self.tiempo

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

        # 1 para volumen normalizado, n_hidro para one-hot de hidrología, T_MAX + 1 para one-hot de tiempo
        dim = 1 + HydroThermalEnv.N_HIDRO + HydroThermalEnv.T_MAX + 1

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(dim,), dtype=np.float32
        )

    def observation(self, obs):
        # idem: si necesitas V_CLAIRE_MAX usa self.env.unwrapped.V_CLAIRE_MAX
        v_norm = obs["volumen"] / HydroThermalEnv.V_CLAIRE_MAX # type: ignore
        h = obs["hidrologia"]
        hidro_oh = np.zeros(HydroThermalEnv.N_HIDRO, dtype=np.float32) # type: ignore

        hidro_oh[h] = 1.0
        semana = obs["tiempo"] % HydroThermalEnv.T_MAX
        time_oh = np.zeros(HydroThermalEnv.T_MAX + 1, dtype=np.float32)
        time_oh[semana] = 1.0

        return np.concatenate(([v_norm], hidro_oh, time_oh), axis=0)

def make_train_env():
    env = HydroThermalEnv()
    env = OneHotFlattenObs(env)
    env = TimeLimit(env, max_episode_steps=HydroThermalEnv.T_MAX+1)
    return env

def train():
    # vectorizado de entrenamiento (8 envs en procesos separados)
    n_envs = 8
    vec_env = SubprocVecEnv([make_train_env for _ in range(n_envs)])

    model = A2C("MlpPolicy", vec_env, verbose=1, seed=42)

    # calcular total_timesteps: por ejemplo 5000 episodios * 104 pasos
    total_episodes = 5000
    total_timesteps = total_episodes * (HydroThermalEnv.T_MAX + 1)

    model.learn(total_timesteps=total_timesteps)
    model.save("a2c_hydro_thermal_claire")

if __name__ == "__main__":
    model_path = "a2c_hydro_thermal_claire"
    
    # Verificar si el archivo del modelo existe
    if os.path.exists(f"{model_path}.zip"):
        try:
            print(f"Cargando modelo desde {model_path}...")
            model = A2C.load(model_path)
            print("Modelo cargado exitosamente.")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            print("Entrenando un modelo nuevo...")
            train()
            model = A2C.load(model_path)
    else:
        print("Archivo del modelo no encontrado, entrenando uno nuevo...")
        train()
        model = A2C.load(model_path)

    # Evaluar el modelo
    # entorno de evaluación (no paralelo aquí, o se puede hacer otro vectorizado)
    eval_env = make_train_env()
    obs, info = eval_env.reset() # type: ignore
    done = False
    reward_sum = 0.0

    actions_list = []
    rewards_list = []
    steps_list = []

    print("Iniciando evaluación del modelo...")
    step = 0 
    # Evaluar el modelo en un episodio
    while True:
        action, _ = model.predict(obs) # type: ignore
        obs, reward, done, _, info = eval_env.step(action) # type: ignore

        reward_sum += reward # type: ignore

        # Guardar datos para graficar
        steps_list.append(step)
        actions_list.append(action[0] if hasattr(action, "__len__") else action)
        rewards_list.append(reward)

        step += 1

        # evaluar como funciona hasta el primer año
        if info["tiempo"] == 51:
            break

    print(f"Recompensa total en evaluación: {reward_sum:.2f}")

    # Graficar acciones y recompensas
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Acciones
    ax1.plot(steps_list, actions_list, marker='o', color='tab:blue')
    ax1.set_ylabel("Acción")
    ax1.set_title("Acciones vs Pasos")

    # Recompensas
    ax2.plot(steps_list, rewards_list, marker='o', color='tab:green')
    ax2.set_xlabel("Paso")
    ax2.set_ylabel("Recompensa")
    ax2.set_title("Recompensas vs Pasos")

    plt.tight_layout()
    plt.show()
