import numpy as np
import random

class Space:
    def __init__(self, n, env):
        self.n = n
        self.env = env

    def sample(self):
        # Devuelve una acción válida aleatoria 
        return random.choice(self.env.available_actions())

class TaTeTiEnv:
    def __init__(self, win_reward=1, draw_reward=0.1, loss_reward=-10):
        # Inicializa el tablero como una matriz de ceros (3x3) y se definen las recompensas
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None
        self.observation_space = Space(3**9, self)
        self.action_space = Space(9, self)
        self.win_reward = win_reward
        self.draw_reward = draw_reward
        self.loss_reward = loss_reward

    def encode_state(self):
        # Convierte el tablero a un índice de estado entero
        cell_values = self.board.flatten()
        cell_values_mapped = np.where(cell_values == -1, 2, cell_values)
        state_index = 0
        for position, cell_value in enumerate(cell_values_mapped):
            state_index += cell_value * (3 ** position)
        return int(state_index)

    def reset(self, opponent_starts=False): 
        # Reinicia el tablero y el estado del juego 
        self.board = np.zeros((3, 3), dtype=int)  
        self.done = False  
        self.winner = None  
        state_index = self.encode_state()  
          
        if opponent_starts:  
            # El oponente hace un movimiento inicial aleatorio  
            available_actions = self.available_actions()  
            opponent_action = random.choice(available_actions)  
            row, col = divmod(opponent_action, 3)  
            self.board[row, col] = -1  # Opponent is player -1  
            state_index = self.encode_state()  
              
            if self.check_winner(-1):  
                self.done = True  
                self.winner = -1  
  
        return state_index, {}

    def step(self, action):
        # El agente (jugador 1) hace un movimiento
        row, col = divmod(action, 3)
        if self.board[row, col] != 0:
            # Acción inválida, devuelve el mismo estado con una recompensa negativa
            state_index = self.encode_state()
            return state_index, self.loss_reward, True, False, {}

        self.board[row, col] = 1  # El agente siempre es el jugador 1

        if self.check_winner(1):
            # El agente gana
            self.done = True
            self.winner = 1
            state_index = self.encode_state()
            return state_index, self.win_reward, True, False, {}

        if not (self.board == 0).any():
            # Empate
            self.done = True
            state_index = self.encode_state()
            return state_index, self.draw_reward, True, False, {}

        # El entorno (jugador -1) hace un movimiento aleatorio 
        available_actions = self.available_actions()
        if available_actions:
            opponent_action = random.choice(available_actions)
            row, col = divmod(opponent_action, 3)
            self.board[row, col] = -1

            if self.check_winner(-1):
                # Agente pierde
                self.done = True
                self.winner = -1
                state_index = self.encode_state()
                return state_index, self.loss_reward, True, False, {}

            if not (self.board == 0).any():
                # Empate
                self.done = True
                state_index = self.encode_state()
                return state_index, self.draw_reward, True, False, {}

        # El juego continúa
        state_index = self.encode_state()
        reward = 0
        return state_index, reward, False, False, {}

    def check_winner(self, player):
        # Revisa filas, columnas y diagonales para ver si hay un ganador
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if self.board[0, 0] == self.board[1,1] == self.board[2,2] == player or \
           self.board[0, 2] == self.board[1,1] == self.board[2,0] == player:
            return True
        return False

    def available_actions(self):
        # Devuelve una lista de acciones disponibles (posiciones vacías en el tablero)
        return [i for i in range(9) if self.board.flatten()[i] == 0]

    def render(self):
        # Imprime el tablero actual
        print(self.board)

# esta función es para poder jugar contra el agente una vez que fue entrenado
def play_against_agent(Qtable, env):
    state, info = env.reset()
    done = False

    print("Bienvenido al ta te ti! Vos sos el jugador '1' y el agente es el '-1'.")
    # Se pregunta al usuario si quiere comenzar jugando
    user_input = input("Quieres comenzar jugando? (y/n): ").lower()
    if user_input == 'y':
        human_player = 1
        agent_player = -1
        current_player = human_player
    else:
        human_player = 1
        agent_player = -1
        current_player = agent_player

    while not done:
        env.render()
        print("Acciones disponibles:", env.available_actions())
        if current_player == human_player:
            # Turno del humano
            try:
                action = int(input("Tú turno. Selecciona tu acción (0-8): "))
            except ValueError:
                action = -1  # Acción inválida
            while action not in env.available_actions():
                try:
                    action = int(input("Acción inválida. Ingrese una acción válida (0-8): "))
                except ValueError:
                    action = -1  # Acción inválida
            # Humano hace un movimiento
            row, col = divmod(action, 3)
            env.board[row, col] = human_player
            # Se hace check de si hay un ganador o un empate
            if env.check_winner(human_player):
                env.render()
                print("Ganaste!")
                done = True
                break
            elif not env.available_actions():
                env.render()
                print("Empate!")
                done = True
                break
            else:
                current_player = agent_player
                state = env.encode_state()
        else:
            # Turno del agente
            available_actions = env.available_actions()
            agent_q_values = Qtable[state][available_actions]
            max_q = np.max(agent_q_values)
            # Se selecciona una acción aleatoria si hay múltiples acciones con el mismo valor Q
            max_actions = [a for a, q in zip(available_actions, agent_q_values) if q == max_q]
            action = random.choice(max_actions)
            print(f"Turno del agente. Agente juega la acción {action}")
            # Agente hace un movimiento
            row, col = divmod(action, 3)
            env.board[row, col] = agent_player
            # Se chequea si hay un ganador o un empate
            if env.check_winner(agent_player):
                env.render()
                print("Ganó el agente!")
                done = True
                break
            elif not env.available_actions():
                env.render()
                print("Empate!")
                done = True
                break
            else:
                current_player = human_player
                state = env.encode_state()

    print("Game over.")