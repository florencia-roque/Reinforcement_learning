from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# to-do: Rodrigo aconsejo usar actorCritic (con one-hot encoding para las variables discretas) para las acciones continuas, si no converge discretizar el volumen del turbinado (ejemplo 10 niveles) y usar metodos tabulares (QLearning)

# Leer archivo 
def leer_archivo(rutaArchivo, sep=None, header=0, sheet_name=0):
    if rutaArchivo.endswith('.xlsx') or rutaArchivo.endswith('.xls'):
        return pd.read_excel(rutaArchivo, header=header,sheet_name=sheet_name)
    else:
        return pd.read_csv(rutaArchivo, sep=sep, header=header, encoding='cp1252')

class HydroThermalEnv(gym.Env):
    T0 = 0
    T_MAX = 104  # Número máximo de pasos (2 años, 52 semanas * 2)
    N_HIDRO = 5

    P_BON_MAX = 155
    P_BAY_MAX = 108
    P_PAL_MAX = 333
    P_SAL_MAX = 1890
    P_CLAIRE_MAX = P_BON_MAX + P_BAY_MAX + P_PAL_MAX + P_SAL_MAX

    P_SOLAR_MAX = 254
    P_EOLICO_MAX = 1584.7
    P_BIOMASA_MAX = 487.3

    # to-do: revisar si estos valores son correctos
    P_TERMICO_BAJO_MAX = 10000
    P_TERMICO_ALTO_MAX = np.inf

    Q_BON_MAX = 680
    Q_BAY_MAX = 828
    Q_PAL_MAX = 1372
    Q_SAL_MAX = 4200
    Q_CLAIRE_MAX = Q_BON_MAX + Q_BAY_MAX + Q_PAL_MAX + Q_SAL_MAX

    V_BON_MAX = 8200
    V_BAY_MAX = 0
    V_PAL_MAX = 1300
    V_SAL_MAX = 1500
    V_CLAIRE_MAX = V_BON_MAX + V_BAY_MAX + V_PAL_MAX + V_SAL_MAX
    V_CLAIRE_MIN = 0

    K_CLAIRE = P_CLAIRE_MAX / Q_CLAIRE_MAX

    V0 = V_CLAIRE_MAX / 2

    # to-do: revisar si estos valores son correctos
    VALOR_EXPORTACION = 12.5  
    COSTO_TERMICO_BAJO = 100  
    COSTO_TERMICO_ALTO = 200  

    SEMILLA = None # Para reproducibilidad, puedes fijar una semilla si lo deseas

    def __init__(self):
        if self.SEMILLA is not None:
            np.random.seed(self.SEMILLA)

        # Espacio de observación
        self.observation_space = spaces.Dict({
            "volumen": spaces.Box(self.V_CLAIRE_MIN, self.V_CLAIRE_MAX, shape=(), dtype=np.float32),
            "hidrologia": spaces.Discrete(self.N_HIDRO, start=0),
            "tiempo": spaces.Discrete(self.T_MAX, start=0)
        })
        
        # Fracción a turbinar del volumen del embalse
        # El agente puede turbinar entre 0 y 1 (100% del volumen)
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)

        # cargar matriz de aportes discretizada (con estado hidrológico 0,1,2,3,4)
        self.data_matriz_aportes_discreta = leer_archivo(f"Datos\\Claire\\clasificado.csv", sep=",", header=0)
        
        # cargar matriz de aportes discretizada (con estado hidrológico 0,1,2,3,4)
        self.data_matriz_aportes_claire = leer_archivo(f"Datos\\Claire\\aporte_claire.csv", sep=",", header=0)
        
        # Cargar datos de energías
        self.data_biomasa = leer_archivo(f"Datos\\MOP\\Deterministicos.xlsx", header=0, sheet_name=0)
        self.data_biomasa = self.data_biomasa.iloc[:,1:]
        self.data_eolico = leer_archivo(f"Datos\\MOP\\Deterministicos.xlsx", header=0, sheet_name=1)
        self.data_eolico = self.data_eolico.iloc[:,1:]
        self.data_solar = leer_archivo(f"Datos\\MOP\\Deterministicos.xlsx", header=0, sheet_name=2)
        self.data_solar = self.data_solar.iloc[:,1:]
        self.data_demanda = leer_archivo(f"Datos\\MOP\\Deterministicos.xlsx", header=0, sheet_name=3)
        self.data_demanda = self.data_demanda.iloc[:,1:]

        # Cargar datos de matrices hidrológicas
        self.data_matrices_hidrologicas = leer_archivo(f"Datos\\Claire\\matrices_sem.csv", sep=",", header=0)
        self.data_matrices_hidrologicas = self.data_matrices_hidrologicas.iloc[:, 1:] # Quito la columna de semanas
        self.matrices_hidrologicas = {}
        for i in range(self.data_matrices_hidrologicas.shape[0]):
            array_1d = self.data_matrices_hidrologicas.iloc[i, :].values
            self.matrices_hidrologicas[i] = array_1d.reshape(5, 5) # type: ignore

        self.v = self.V0                    # Volumen inicial del embalse
        self.t = 0                          # Tiempo inicial
        self.h = self._inicial_hidrologia() # Estado hidrológico inicial

    def _sortear_cronica_inicial(self):
        return np.random.randint(self.data_matriz_aportes_discreta.shape[1])
    
    def _inicial_hidrologia(self):
        # retorna el estado inicial del estado hidrológico 0,1,2,3,4
        self.c = self._sortear_cronica_inicial()
        print("La cronica sorteada es: ", self.c)
        h0 = self.data_matriz_aportes_discreta.iloc[self.T0,self.c]
        return int(h0) # type: ignore

    def _siguiente_hidrologia(self):
        # retorna el estado hidrológico siguiente 0,1,2,3,4
        self.h_anterior = self.h
        self.clases_hidrologia = np.arange(self.matrices_hidrologicas[self.t].shape[0]) # array con las clases 0,1,2,3,4
        siguiente_eshy = np.random.choice(self.clases_hidrologia, p=self.matrices_hidrologicas[self.t][self.h,:])
        return siguiente_eshy

    def _rotar_fila(self, fila):
        valores = fila.tolist()
        valor = fila.pop(0)
        valores.append(valor)
        nueva_fila = pd.DataFrame([valores], columns=fila.columns)
        return nueva_fila
    
    def _aportes(self):
        # dados dos estados (inicial y final) y dos semanas correspondientes a esos estados, sorteo una ocurrencia de aportes para el lago claire
        estados_ini = self.data_matriz_aportes_discreta.loc[self.t] 

        if(self.t < 51):
            estados_fin = self.data_matriz_aportes_discreta.loc[self.t+1] 
        else:
            estados_fin = self._rotar_fila(self.data_matriz_aportes_discreta.loc[0]) 

        coincidencias = (estados_ini == self.h_anterior) & (estados_fin == self.h)
        columnas_validas = self.data_matriz_aportes_discreta.columns[coincidencias] # type: ignore

        if len(columnas_validas) == 0:
            raise ValueError("No hay coincidencias válidas para los estados hidrológicos actuales")
            
        año_sorteado = np.random.choice(columnas_validas)

        # Obtener valor en claire para fila2 y ese año
        valor_claire = self.data_matriz_aportes_claire.loc[self.t, año_sorteado]
        return valor_claire
    
    def _demanda(self):
        # Leer datos de demanda desde archivo
        energias_demandas = self.data_demanda.iloc[:,self.c]
        if self.t < len(energias_demandas):
            return energias_demandas.iloc[self.t]
        else:
            raise ValueError("Tiempo fuera de rango para datos de demanda")
    
    def _gen_eolico(self):
        # Leer datos eólicos desde archivo
        energias_eolico = self.data_eolico.iloc[:,self.c]
        if self.t < len(energias_eolico):
            return energias_eolico.iloc[self.t]
        else:
            raise ValueError("Tiempo fuera de rango para datos eólicos")

    def _gen_solar(self):
        # Leer datos solares desde archivo
        energias_solar = self.data_solar.iloc[:,self.c]
        if self.t < len(energias_solar):
            return energias_solar.iloc[self.t]
        else:
            raise ValueError("Tiempo fuera de rango para datos solares")

    def _gen_bio(self):
        # Leer datos solares desde archivo
        energias_biomasa = self.data_biomasa.iloc[:,self.c]
        if self.t < len(energias_biomasa):
            return energias_biomasa.iloc[self.t]
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

            if demanda_residual > 0:
                energia_termico_alto = self._gen_termico_alto(demanda_residual)
                demanda_residual -= energia_termico_alto

        if demanda_residual < 0:
            exportacion = -demanda_residual

        # Retornar ingresos por exportación y costos de térmico
        ingreso_exportacion = exportacion * self.VALOR_EXPORTACION
        costo_termico = energia_termico_bajo * self.COSTO_TERMICO_BAJO + energia_termico_alto * self.COSTO_TERMICO_ALTO
        return ingreso_exportacion, costo_termico, energia_termico_bajo, energia_termico_alto
    
    def step(self, action):
        # Validar que la acción esté en el espacio válido
        action = np.array(action, dtype=np.float32).reshape(1,)
        assert self.action_space.contains(action), f"Acción inválida: {action}. Debe estar en {self.action_space}"

        # Volumen a turbinar
        frac = float(action[0])
        qt = frac * self.v

        # despacho: e_eolo + e_sol + e_bio + e_termico + e_hidro = dem + exp
        ingreso_exportacion, costo_termico, energia_termico_bajo, energia_termico_alto = self._despachar(qt)

        # recompensa: −costo_termico + ingreso_exportacion
        reward = -costo_termico + ingreso_exportacion

        info = {
            "volumen": self.v,
            "hidrologia": self.h,
            "tiempo": self.t,
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
        self.h = self._siguiente_hidrologia()
        aportes = self._aportes()
        self.v = min(self.v - qt + aportes, self.V_CLAIRE_MAX) # type: ignore
        self.t += 1
        
        info["aportes"] = aportes
        info["volumen_siguiente"] = self.v
        info["hidrologia_siguiente"] = self.h
        info["tiempo_siguiente"] = self.t

        done = (self.t >= self.T_MAX)
        return self._get_obs(), reward, done, False, info
    
    def reset(self, *, seed=None, options=None):
        self.v = self.V0
        self.t = 0
        self.h = self._inicial_hidrologia()
        info = {
            "volumen_inicial": self.v,
            "hidrologia_inicial": self.h,
            "tiempo_inicial": self.t
        }
        return self._get_obs(), info
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"Semana {self.t}/52:")
            print(f"  Volumen embalse: {self.v:.2f}/{self.V_CLAIRE_MAX}")
            print(f"  Estado hidrológico: {self.h}")
            print(f"  Porcentaje llenado: {(self.v/self.V_CLAIRE_MAX)*100:.1f}%")
            print("-" * 30)
        elif mode == 'rgb_array':
            # Retornar una imagen como array numpy para grabación
            pass
        elif mode == 'ansi':
            # Retornar string para mostrar en terminal
            return f"T:{self.t} V:{self.v:.1f} H:{self.h}"
        
    def _get_obs(self):
        # Mapeo de variables internas a observación del agente

        obs = {
            "volumen": np.array(self.v, dtype=np.float32),
            "hidrologia": int(self.h),
            "tiempo": int(self.t)
        }
        
        # Validar contra observation_space (opcional, útil para debug)
        assert self.observation_space.contains(obs), f"Observación inválida: {obs}. Debe estar en {self.observation_space}"
        return obs
    
class OneHotFlattenObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_weeks = 52

        # "Desenvuelvo" el Monitor (y cualquier otro wrapper) para acceder a N_HIDRO
        n_hidro = self.env.unwrapped.N_HIDRO # type: ignore
        dim = 1 + n_hidro + self.num_weeks

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(dim,), dtype=np.float32
        )

    def observation(self, obs):
        # idem: si necesitas V_CLAIRE_MAX usa self.env.unwrapped.V_CLAIRE_MAX
        v_norm = obs["volumen"] / self.env.unwrapped.V_CLAIRE_MAX # type: ignore
        h = obs["hidrologia"]
        hidro_oh = np.zeros(self.env.unwrapped.N_HIDRO, dtype=np.float32) # type: ignore

        hidro_oh[h] = 1.0
        semana = obs["tiempo"] % self.num_weeks
        time_oh = np.zeros(self.num_weeks, dtype=np.float32)
        time_oh[semana] = 1.0

        return np.concatenate(([v_norm], hidro_oh, time_oh), axis=0)

if __name__ == "__main__":
    # Crear un entorno vectorizado con múltiples instancias
    vec_env = make_vec_env(
        HydroThermalEnv,
        n_envs=8,
        vec_env_cls=SubprocVecEnv,
        seed=42,
        wrapper_class=OneHotFlattenObs
    )

    model = A2C("MlpPolicy", vec_env, verbose=1, seed=42)
    model.learn(total_timesteps=1_000_000)
    # Guardar el modelo entrenado
    model.save("a2c_hydro_thermal_claire")