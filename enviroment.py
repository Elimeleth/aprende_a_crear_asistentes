import numpy as np
import pickle
from time import sleep
from qa_agent import chat
from agent import Assistant

from messages_reccomend import upsert_message

class RAGEnv:
    def __init__(self, qa_chain, state_file='state.pkl'):
        self.qa_chain = qa_chain
        self.state_file = state_file
        self.state, self.done = self.load_state()

    def load_state(self):
        try:
            with open(self.state_file, 'rb') as f:
                state, done = pickle.load(f)
        except FileNotFoundError:
            state, done = None, False
        return state, done

    def save_state(self):
        with open(self.state_file, 'wb') as f:
            pickle.dump((self.state, self.done), f)

    def reset(self, query):
        self.state = query
        self.done = False
        self.save_state()
        return self.state

    def step(self, action):
        response = self.qa_chain(action)
        print(f"Action: {action}\nRespuesta generada: {response}")
        feedback = int(input("Evalúa la respuesta (-1, 0, 1): "))
        upsert_message(action, response, feedback)
        self.done = True if feedback != 0 else False
        self.save_state()
        return response, feedback, self.done


class RAGQLearningAgent:
    def __init__(self, env, q_table_file='q_table.pkl', learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.env = env
        self.q_table_file = q_table_file
        self.q_table = self.load_q_table()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def load_q_table(self):
        try:
            with open(self.q_table_file, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}

    def save_q_table(self):
        with open(self.q_table_file, 'wb') as f:
            pickle.dump(self.q_table, f)

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return state
        return max(self.q_table.get(state, {}), key=self.q_table[state].get, default=state)

    def update_q_table(self, state, action, reward, next_state):
        self.q_table = self.load_q_table()
        
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0

        best_next_action = max(self.q_table.get(next_state, {}).values(), default=0.0)
        td_target = reward + self.discount_factor * best_next_action
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        self.save_q_table()

    def train(self, queries, episodes=5):
        for ep in range(episodes):
            query = np.random.choice(queries)
            state = self.env.reset(query)
            
            while True:
                sleep(1)
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                
                if done:
                    break
            
            self.exploration_rate *= self.exploration_decay
        
queries = ["precio del brunch", "los menores pagan brunch", "¿cuanto sale el daypass pizza?", "Puedo llevar carriola", "cual es el precio del brunch para 3 adultos"]

# env = RAGEnv(Assistant().run)
env = RAGEnv(chat)
agent = RAGQLearningAgent(env)

if False:
    agent.train(queries, episodes=10)
    print("TRAIN FINISHED")

while True:
    inp = input("Chatea aqui: ")
    if inp == "q": break
    
    state = env.reset(inp)
    print(f"Consulta inicial: {state}")
    action = agent.choose_action(state)
    print(f"Action chosen: {action}")
    response, reward, done = env.step(action)
    agent.update_q_table(state, action, reward, response)

    state = response  # Usa la respuesta generada como el nuevo estado