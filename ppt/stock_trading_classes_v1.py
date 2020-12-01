### Classes ###

import pandas as pd
import numpy as np

class DeterministicClassifier():
    '''
    A class to train and score based on a deterministic method to find the desired position under state condition.
    '''
    def __init__(self):
        self.df_state_to_pstn = pd.DataFrame(columns=['pstn'])
        self.state_cols = []
    
    
    
    def choose_pstn(self, df):
        '''
        Given DataFrame df with "long" postion return column, choose the position that maximizes the final total return.
        '''
        if 'rtrn' not in df.columns:
            raise KeyError('Input DataFrame has to contain column "rtrn" for "choose_pstn()" function.')
           
        df['log_rtrn'] = np.log(1.0 + df.rtrn)
        
        if df.log_rtrn.sum()>0.0:
            return 1
        
        elif df.log_rtrn.sum()<0.0:
            return -1
        
        else:
            return 0
    
    
    
    def fit(self, df_train, state_cols):
        '''
        Given DataFrames df_train, determine the desired position under state condition.
        '''
        if 'rtrn' not in df_train.columns:
            raise KeyError('Input DataFrame has to contain column "rtrn" for "fit()" function.')
        
        if len(state_cols)==0:
            raise ValueError('The list of the state columns needs to be specified.')
        
        if not set(state_cols).issubset(set(df_train.columns)):
            raise KeyError('Not all of the state columns are present in the input DataFrame.')
    
        df_state = df_train.groupby(state_cols).size().reset_index().rename(columns={0:'count'})
        df_state.reset_index(drop=False, inplace=True)
        df_state.rename(columns={'index': 'state_idx'}, inplace=True)
        df_train_state = pd.merge(df_train, df_state, how='inner', on=state_cols)
        
        for idx in range(0, df_state.shape[0]):
            df_state.loc[df_state.state_idx==idx, 'pstn'] = \
            self.choose_pstn(df_train_state.loc[df_train_state.state_idx==idx].copy())
        
        df_state['pstn'] = df_state.pstn.astype(int)
        df_state.drop(columns=['state_idx', 'count'], inplace=True)
        
        self.df_state_to_pstn = df_state
        self.state_cols = state_cols
        
        return None
        
    
    
    def predict(self, df_test):
        '''
        Given df_test and fitted self.df_state_to_pstn, andd pstn prediction to df_test.
        '''
        if not set(self.state_cols).issubset(set(df_test.columns)):
            raise KeyError('Not all of the state columns are present in the input DataFrame.')
        
        df_test_pstn = pd.merge(df_test, self.df_state_to_pstn, how='left', on=self.state_cols)
        df_test_pstn.pstn.fillna(0)
        
        return df_test_pstn
        
        
        
        
  
class QLearner():
    '''
    QLearner based reinforcement learning.
    '''
    def __init__(self, gamma=0.9999, penalty=0.001):
        self.gamma = gamma
        self.penalty = penalty
        self.df_Q = pd.DataFrame(columns=['state_code', 'Q_long', 'Q_flat', 'Q_short', 'Q_max', 'pstn', 'pstn_prev'])
        self.state_cols = []
        self.Q_LOOKUP = {1: 'Q_long', 0:'Q_flat', -1:'Q_short'}
        self.iteration_cnt = 0
        self.alpha = 0.2
        self.pstn_same_cnt = 0

        
        
    def codify_state(self, df, state_cols):
        '''
        Create a list with state columns values.
        '''
        df.reset_index(drop=True, inplace=True)
        df['state_code'] = df.apply(lambda x: str([x[col] for col in state_cols]).replace('[','').replace(']',''), axis=1)
        
        return df      
    
    
    
    def find_max_pstn(self, Q_long, Q_flat, Q_short):
        '''
        Given three values, find the max and the corresponding pstn.
        '''
        if (Q_long>=Q_flat) and (Q_long>=Q_short):
            return Q_long, 1
        
        elif (Q_short>=Q_flat) and (Q_short>=Q_long):
            return Q_short, -1
        
        else:
            return Q_flat, 0
        
        
    
    def initialize_Q(self, df_train, state_cols):
        '''
        Create Q table based on df_train and list of state_cols, adding 'holding' (-1, 0, 1) to expand.
        Initialize with vary small random numbers.
        '''
        if 'rtrn' not in df_train.columns:
            raise KeyError('Input DataFrame has to contain column "rtrn" for "fit()" function.')
        
        if len(state_cols)==0:
            raise ValueError('The list of the state columns needs to be specified.')
        
        if not set(state_cols).issubset(set(df_train.columns)):
            raise KeyError('Not all of the state columns are present in the input DataFrame.')
        
        df_state = df_train.groupby(state_cols).size().reset_index().rename(columns={0:'count'})
        df_holding = pd.DataFrame({'holding': [-1, 0, 1]})
        df_Q = pd.merge(df_state.assign(key=0), df_holding.assign(key=0), on='key').drop(['key', 'count'], axis=1)
        state_cols.append('holding')
        df_Q = self.codify_state(df_Q, state_cols)
        
        df_Q[['Q_long', 'Q_flat', 'Q_short']] = pd.DataFrame(np.random.normal(0, 0.0001, (df_Q.shape[0], 3)))
        df_Q['Q_max'] = df_Q.apply(lambda x: self.find_max_pstn(x['Q_long'], x['Q_flat'], x['Q_short'])[0], axis=1)
        df_Q['pstn'] = df_Q.apply(lambda x: self.find_max_pstn(x['Q_long'], x['Q_flat'], x['Q_short'])[1], axis=1)
        df_Q['pstn_prev'] = 0
        
        self.df_Q = df_Q[['state_code', 'Q_long', 'Q_flat', 'Q_short', 'Q_max', 'pstn', 'pstn_prev']]
        self.df_Q.set_index('state_code', inplace=True)
        self.state_cols = state_cols
        
        return None
    
        
    
    def lookup_Q_pstn(self, state_code, holding):
        '''
        Based on state_code and Q table, find Q value and pstn.
        '''
        if holding not in [-1, 0, 1]:
            raise ValueError('"holding" needs to have a value from -1 (short), 0 (flat) or 1 (long).')
        
        state_code = state_code + ', ' + str(holding)
        if state_code not in self.df_Q.index:
            return (0.0, 0)
           
        else:
            return (self.df_Q.loc[state_code, ['Q_max', 'pstn']].values[0], int(self.df_Q.loc[state_code, ['Q_max', 'pstn']].values[1]))

        
        
    def update_Q(self, state_code_in, holding_in, state_code_out, holding_out, rtrn):
        '''
        Update single Q value based on in/out state_code/holding info. Notice action (long/flat/short) determines out holding.
        IMPORTANT: rtrn is the forward 1 day return for in date.
        '''
        if holding_in not in [-1, 0, 1]:
            raise ValueError('"In state holding" needs to have a value from -1 (short), 0 (flat) or 1 (long).')
        
        if holding_out not in [-1, 0, 1]:
            raise ValueError('"Out state holding" needs to have a value from -1 (short), 0 (flat) or 1 (long).')
                
        rtrn *= holding_out
        if holding_in in [-1, 1] and holding_in!=holding_out:
            rtrn -= self.penalty
        Q_max_next, pstn_next = self.lookup_Q_pstn(state_code_out, holding_out)            
        Q_updt = rtrn + self.gamma * Q_max_next

        state_code_in = state_code_in + ', ' + str(holding_in)
        if state_code_in not in self.df_Q.index:
            raise KeyError("In state doesn't exist in the Q table.")
        
        else:
            Q_action = self.Q_LOOKUP[holding_out]
            self.df_Q.loc[state_code_in, Q_action] = (1.0 - self.alpha) * self.df_Q.loc[state_code_in, Q_action] + \
                                                     self.alpha * Q_updt
            
            self.df_Q.loc[state_code_in, ['Q_max', 'pstn']] = self.find_max_pstn(self.df_Q.loc[state_code_in, 'Q_long'],
                                                                                 self.df_Q.loc[state_code_in, 'Q_flat'],
                                                                                 self.df_Q.loc[state_code_in, 'Q_short'])
                      
        return None
    
    
    
    def update_Q_table(self, df_states):
        '''
        Update Q table for all records in df_states, which has state_code, state_code_next, rtrn.
        '''
        if not set(['state_code', 'state_code_next', 'rtrn']).issubset(set(df_states.columns)):
            raise KeyError('Not all of the required columns (state_code, state_code_next, rtrn) are present.')
        
        df_states.reset_index(drop=True, inplace=True)
        for i, row in df_states.iloc[::-1].iterrows():
            for holding_in in [-1, 0, 1]:
                for holding_out in [-1, 0, 1]:
                    self.update_Q(row.state_code, holding_in, row.state_code_next,  holding_out, row.rtrn)
        
        return None
    
    
    def predict(self, df_test):
        '''
        Given df_test and fitted self.df_Q, andd pstn prediction to df_test.
        '''
        
        df_test.reset_index(drop=True, inplace=True)
        self.state_cols.remove('holding')
        
        if not set(self.state_cols).issubset(set(df_test.columns)):
            raise KeyError('Not all of the state columns are present in the input DataFrame.')
        
        df_test_pstn = self.codify_state(df_test, self.state_cols)
        df_test_pstn['pstn'] = 0
        pstn_prev = 0
        for i, row in df_test_pstn.iterrows():
            Q_max, pstn = self.lookup_Q_pstn(row.state_code, pstn_prev)
            df_test_pstn.at[i, 'pstn'] = pstn
            pstn_prev = pstn
            
        self.state_cols.append('holding')
        
        return df_test_pstn
        
        
    
    def fit(self, df_train, state_cols):
        '''
        1. Initialize Q table.
        2. Loop through the df_train backward, .
        3. Each iteration, update Q table based on formula:
            Q'[S,a] = (1-alpha) * Q[S,a] + alpha * (r + gamma * Q_max(S')), with alpha decay choice.
        4. After df_train is completed, check for convergence. If not, feed df_train again until convergence.
        '''
        
        pass


        
        
        
        
        