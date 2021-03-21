# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:35:54 2017

@author: cyou7
"""

import numpy as np
import random as rd

#==============================================================================
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_deriv(x): # x is layer output
#    return sigmoid(x)*(1.0-sigmoid(x))
    return x*(1.0-x)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x): # x is layer output 
    return 1.0 - x**2

#==============================================================================
# from state vector to vehicle array
def state2matrix(state_aux):   # state_aux: 1~3 for HV + binary[1-8] for EV
    VehicleArray =np.zeros((3,3))
    
    VehicleArray[0,:] = state_aux[1:4]
    VehicleArray[2,:] = state_aux[6:9]
    if state_aux[0] == 1:
        VehicleArray[1,0] = 2  # distinguish from binary value, indicate HV
        VehicleArray[1,1:3] = state_aux[4:6]
    elif state_aux[0] == 3:
        VehicleArray[1,2] = 2
        VehicleArray[1,0:2] = state_aux[4:6]
    elif state_aux[0] == 2:
        VehicleArray[1,1] = 2
        VehicleArray[1,0] = state_aux[4]
        VehicleArray[1,2] = state_aux[5]
    return VehicleArray

#==============================================================================
# from vehicle array to state vector
def matrix2state(VehicleArray_aux):
    
    state = np.zeros(9)
    state[1:4] = VehicleArray_aux[0,:]
    state[6:9] = VehicleArray_aux[2,:]
    
    indices = np.asarray(np.where(VehicleArray_aux == 2 )).T  # get position of HV
    
    if indices[0][1] == 0:
        state[0] = 1
        state[4:6] = VehicleArray_aux[1,1:3]
    elif indices[0][1] == 1:
        state[0] = 2
        state[4] = VehicleArray_aux[1,0]
        state[5] = VehicleArray_aux[1,2]
    elif indices[0][1] == 2:
        state[0] = 3
        state[4:6] = VehicleArray_aux[1,0:2]
        
    return state

#==============================================================================
# available actions of each EV
def Available_actions_EV(big_state_matrix,big_pos,HV_lane,NrLanes):  # HV_lane=0 <=> state_aux[0] = 1, HV_lane=right lane <=> state_aux[0] = 3

    aux0 = np.concatenate(([1],big_state_matrix[0,:],[1]))
    aux1 = np.concatenate(([1],big_state_matrix[1,:],[1]))
    aux2 = np.concatenate(([1],big_state_matrix[2,:],[1])) 
    aux3 = np.concatenate(([1],big_state_matrix[3,:],[1]))
    aux4 = np.concatenate(([1],big_state_matrix[4,:],[1]))         
    VehicleArray_aux = np.array([[1,0,0,0,0,0,1],aux0,aux1,aux2,aux3,aux4,[1,0,0,0,0,0,1]])
  
    pos_aux =  big_pos+[1,1] # new position in 7x7 array
    pos_aux_acc = pos_aux + [-1,0]
    if VehicleArray_aux[pos_aux_acc[0],pos_aux_acc[1]] == 0:
        acc = 1
    else:
        acc = 0 
    pos_aux_brk = pos_aux + [1,0]
    if VehicleArray_aux[pos_aux_brk[0],pos_aux_brk[1]] == 0:
        brk = 1
    else:
        brk = 0      
    pos_aux_left = pos_aux + [0,-1]
    if VehicleArray_aux[pos_aux_left[0],pos_aux_left[1]] == 0:
        left = 1
    else:
        left = 0
        
    pos_aux_right = pos_aux + [0,1]
    if VehicleArray_aux[pos_aux_right[0],pos_aux_right[1]] == 0:
        right = 1
    else:
        right = 0   
    
    return acc, brk, left, right

#==============================================================================
#==============================================================================
# add new vehicles randomly to vacant grids after one-step motion of HV
# HV_lane has been changed in object HV, by HV.TakeAction()

def Add_Vehicle_HV_move(state_aux,big_state_matrix,HV_lane,HV_action,NrLanes,EV_name_list,EV_pos_list): # to use only if no collision
    
    VehicleArray = state2matrix(state_aux) # before HV moves
    indices = np.asarray(np.where(big_state_matrix == 2 )).T  # in 5x5 array
    
    EV_name_list_first = []
    EV_name_list_last = []
    EV_pos_row_first = [] # rows of EV_pos_list that are in first row of big_state_matrix
    EV_pos_row_last = [] # rows of EV_pos_list that are in last row of big_state_matrix
    for m in range(len(EV_name_list)):
        if EV_pos_list[m,0] == 0:
            EV_name_list_first = EV_name_list_first + [EV_name_list[m]]
            EV_pos_row_first = EV_pos_row_first + [m]
        elif EV_pos_list[m,0] == 4:
            EV_name_list_last = EV_name_list_last + [EV_name_list[m]]
            EV_pos_row_last = EV_pos_row_last + [m]
#        
#    CurEVs = len(indices_EV)
    new_EV_order = [-1]  # indicate no new EV by default
    add_EV_pos_list = np.array([-1,-1])
    
###############################################################################    
#                   need see how HV_lane effects state    
###############################################################################   

    if HV_action == 'left' and HV_lane==0: #change HV to left boundary 
        state_aux[0] = state_aux[0]-1
        state_aux[3] = 0
        state_aux[5] = 0
        state_aux[8] = 0  #3rd column disappears -> 0
        VehicleArray = state2matrix(state_aux)
        big_state_matrix[2,1] = 0
        big_state_matrix[2,0] = 2

    elif HV_action == 'left' and HV_lane!=0:
        big_state_matrix[2,HV_lane+1] = 0
        big_state_matrix[2,HV_lane] = 2        
        VehicleArray = big_state_matrix[1:NrLanes-1,HV_lane-1:HV_lane+2]
        
    elif HV_action == 'right' and HV_lane==NrLanes-1:  # only change HV
        state_aux[0] = state_aux[0]+1
        state_aux[1] = 0
        state_aux[4] = 0
        state_aux[6] = 0   # 1st column disappears ->0         
        VehicleArray = state2matrix(state_aux)
        big_state_matrix[2,NrLanes-2] = 0
        big_state_matrix[2,NrLanes-1] = 2
        
    elif HV_action == 'right' and HV_lane!=NrLanes-1:
        big_state_matrix[2,HV_lane-1] = 0
        big_state_matrix[2,HV_lane] = 2        
        VehicleArray = big_state_matrix[1:NrLanes-1,HV_lane-1:HV_lane+2]
      
    elif HV_action == 'acc':
#        new_vehicle_Nr = rd.sample(range(NrLanes+1), 1) # maximum nr = NrLanes, full of EVs
        new_vehicle_Nr = rd.sample(range(NrLanes), 1) # not full of EVs, up t0  NrLanes-1 EVs      
        new_vehicle_pos = rd.sample(range(NrLanes), new_vehicle_Nr[0])
        
        big_state_matrix[2,indices[0][1]] = 0
        big_state_matrix[1,indices[0][1]] = 2        
        
        aux = np.array([[0,0,0,0,0]])
        if len(EV_name_list)==0:
            max_EV_order = 0
        else: 
            max_EV_order = max(EV_name_list)
#        EV_name_list = EV_name_list[0:CurEVs-Nr_EV_last]  # remove last row EV orders
#        EV_pos_list = EV_pos_list[0:CurEVs-Nr_EV_last,:]
        EV_name_list = [item for item in EV_name_list if item not in EV_name_list_last]
        EV_pos_list = np.delete(EV_pos_list, (EV_pos_row_last), axis=0)
        
        if new_vehicle_Nr[0] == 0:
            big_state_matrix = np.concatenate((aux,big_state_matrix[0:4,:]))
            EV_pos_list = EV_pos_list + [1,0] 
        else:
            EV_pos_list = EV_pos_list + [1,0] 
            add_EV_pos_list = np.array([], dtype=np.int64).reshape(0,2)
            k = 0
            new_EV_order = []
            for i in new_vehicle_pos: 
                k = k+1
                aux[0][i] = 1
                EV_name_list = [max_EV_order+k] + EV_name_list # append new EV orders to the front
                EV_pos_list = np.concatenate((np.array([[0,i]]),EV_pos_list))
                add_EV_pos_list = np.concatenate((np.array([[0,i]]),add_EV_pos_list))  # added EVs' positions
                new_EV_order =  [max_EV_order+k] + new_EV_order
#            new_EV_order = list(np.asarray(new_vehicle_pos) + max_EV_order) 
            big_state_matrix = np.concatenate((aux,big_state_matrix[0:4,:]))  # assume only to show 5 rows of vehicles 
        
        VehicleArray = np.zeros((3,3))
        if  HV_lane == 0:
            VehicleArray[:,0:2] = big_state_matrix[1:-1,0:2]
        elif HV_lane==NrLanes-1:
            VehicleArray[:,1:3] = big_state_matrix[1:-1,3:5]
        else:
            VehicleArray = big_state_matrix[1:NrLanes-1,HV_lane-1:HV_lane+2]
            
    elif HV_action == 'brk':
#        new_vehicle_Nr = rd.sample(range(NrLanes+1), 1)
        new_vehicle_Nr = rd.sample(range(NrLanes), 1) # up to NrLanes-1 EVs
        new_vehicle_pos = rd.sample(range(NrLanes), new_vehicle_Nr[0])
        
        big_state_matrix[2,indices[0][1]] = 0
        big_state_matrix[3,indices[0][1]] = 2        
        
        aux = np.array([[0,0,0,0,0]])
        if len(EV_name_list)==0:
            max_EV_order = 0
        else: 
            max_EV_order = max(EV_name_list)

        EV_name_list = [item for item in EV_name_list if item not in EV_name_list_first]
        EV_pos_list = np.delete(EV_pos_list, (EV_pos_row_first), axis=0)

        if new_vehicle_Nr[0] == 0:
            big_state_matrix = np.concatenate((big_state_matrix[1:5,:],aux))
            EV_pos_list = EV_pos_list + [-1,0] 
        else:
            EV_pos_list = EV_pos_list + [-1,0] 
#            add_EV_pos_list = np.asarray([])
            add_EV_pos_list = np.array([], dtype=np.int64).reshape(0,2)
            k = 0
            new_EV_order = []
            for i in new_vehicle_pos:
                k = k+1
                aux[0][i] = 1
                EV_name_list.append(max_EV_order+k) # append new EV orders to the end
                EV_pos_list = np.concatenate((EV_pos_list,np.array([[4,i]])))
                add_EV_pos_list = np.concatenate((add_EV_pos_list,np.array([[4,i]])))  # added EVs' positions
                new_EV_order.append(max_EV_order+k)
#            new_EV_order = list(np.asarray(new_vehicle_pos) + max_EV_order) 
            big_state_matrix = np.concatenate((big_state_matrix[1:5,:],aux))  # assume only to show 5 rows of vehicles 
        
        VehicleArray = np.zeros((3,3))
        if  HV_lane == 0:
            VehicleArray[:,0:2] = big_state_matrix[1:-1,0:2]
        elif HV_lane==NrLanes-1:
            VehicleArray[:,1:3] = big_state_matrix[1:-1,3:5]
        else:
            VehicleArray = big_state_matrix[1:NrLanes-1,HV_lane-1:HV_lane+2]
            
    state_aux = matrix2state(VehicleArray)    
    return VehicleArray,state_aux,big_state_matrix,EV_name_list,new_EV_order,EV_pos_list,add_EV_pos_list

#==============================================================================
#==============================================================================
# add new vehicles randomly to vacant grids before motions of all EVs
 
def Add_EVs_front_back(big_state_matrix,HV_lane,NrLanes,EV_name_list_aux,EV_pos_list_aux): # to generate new state s_t+1
    
    new_EV_front_order = [-1]  # indicate no new EV by default
    add_EV_pos_front_list = np.array([-1,-1])
    new_EV_back_order = [-1]  # indicate no new EV by default
    add_EV_pos_back_list = np.array([-1,-1])
    EV_name_list = EV_name_list_aux +[]
    EV_pos_list = EV_pos_list_aux + 0
    
    big_state_matrix2 = big_state_matrix+0  # +0 to decouple VehicleArray_aux2 from VehicleArray_aux 
    
    indices_front = np.asarray(np.where(big_state_matrix[0,:] == 0 )).T  # vacant positions
    indices_back = np.asarray(np.where(big_state_matrix[-1,:] == 0 )).T  # vacant positions
    
    if len(indices_front)>=1:
        Nr_new_vehicle_front = rd.sample(range(len(indices_front)), 1)  # avoid full of EVs, at leat 1 vacant
    else:
        Nr_new_vehicle_front = rd.sample(range(len(indices_front)+1), 1)
    if len(indices_back)>=1:
        Nr_new_vehicle_back = rd.sample(range(len(indices_back)), 1) # avoid full of EVs, at leat 1 vacant
    else:
        Nr_new_vehicle_back = rd.sample(range(len(indices_back)+1), 1)
    
    if len(EV_name_list) == 0:
        max_EV_front_order = 0
    else:
        max_EV_front_order = max(EV_name_list)
        
    if Nr_new_vehicle_front[0] == 0:
        big_state_matrix = big_state_matrix2
    else:        
        new_vehicle_pos_front = rd.sample(range(len(indices_front)), Nr_new_vehicle_front[0]) # take several number from 0~len(indices_front)
        add_EV_pos_front_list = np.array([], dtype=np.int64).reshape(0,2) #np.asarray([])
        k = 0
        new_EV_front_order = []
        for i in new_vehicle_pos_front: 
            k = k+1
            big_state_matrix2[0,indices_front[i]] = 1
#            big_state_matrix3[indices_front[i][0],indices_front[i][1]] = -1
            add_EV_pos_front_list = np.concatenate((np.array([[0,indices_front[i]]]),add_EV_pos_front_list))
            EV_name_list = [max_EV_front_order+k] + EV_name_list
            EV_pos_list = np.concatenate((np.array([[0,indices_front[i]]]),EV_pos_list))
            new_EV_front_order =  [max_EV_front_order+k]+new_EV_front_order
#        new_EV_front_order = list(np.asarray(new_vehicle_pos_front) + max_EV_front_order+k)  
        big_state_matrix = big_state_matrix2

    if len(EV_name_list) == 0:
        max_EV_back_order = 0
    else:
        max_EV_back_order = max(EV_name_list) # front new EV included already
    if Nr_new_vehicle_back[0] == 0:
        big_state_matrix = big_state_matrix2
    else:        
        new_vehicle_pos_back = rd.sample(range(len(indices_back)), Nr_new_vehicle_back[0])
        add_EV_pos_back_list = np.array([], dtype=np.int64).reshape(0,2) #np.asarray([])
        k = 0
        new_EV_back_order = []
        for i in new_vehicle_pos_back: 
            k = k+1
            big_state_matrix2[4,indices_back[i]] = 1
            add_EV_pos_back_list = np.concatenate((add_EV_pos_back_list,np.array([[4,indices_back[i]]])))
            EV_name_list.append(max_EV_back_order+k)
            EV_pos_list = np.concatenate((EV_pos_list,np.array([[4,indices_back[i]]])))
            new_EV_back_order.append(max_EV_back_order+k)
        big_state_matrix = big_state_matrix2   
     
    return big_state_matrix,new_EV_front_order,new_EV_back_order,EV_name_list,EV_pos_list,add_EV_pos_front_list,add_EV_pos_back_list

#==============================================================================
#==============================================================================
# safe actions of HV
def Safe_actions_HV(state_aux):  # HV_lane=0 <=> state_aux[0] = 1, HV_lane=right lane <=> state_aux[0] = 3
    
    VehicleArray_aux = state2matrix(state_aux)
    if state_aux[0] == 1:  # fill 1 to occupy left boundary
        aux0 = np.concatenate(([1],VehicleArray_aux[0,:],[0]))
        aux1 = np.concatenate(([1],VehicleArray_aux[1,:],[0]))
        aux2 = np.concatenate(([1],VehicleArray_aux[2,:],[0]))    
        VehicleArray_aux2 = np.array([[1,0,0,0,0],aux0,aux1,aux2,[1,0,0,0,0]])  #extended to 5x5 with zeros & ones, left boundary considered
        pos_HV = np.array([1,0])
    elif state_aux[0] == 3:
        aux0 = np.concatenate(([0],VehicleArray_aux[0,:],[1]))
        aux1 = np.concatenate(([0],VehicleArray_aux[1,:],[1]))
        aux2 = np.concatenate(([0],VehicleArray_aux[2,:],[1]))    
        VehicleArray_aux2 = np.array([[0,0,0,0,1],aux0,aux1,aux2,[0,0,0,0,1]])  #extended to 5x5 with zeros & ones, right boundary considered
        pos_HV = np.array([1,2])
    elif state_aux[0] == 2:
        aux0 = np.concatenate(([0],VehicleArray_aux[0,:],[0]))
        aux1 = np.concatenate(([0],VehicleArray_aux[1,:],[0]))
        aux2 = np.concatenate(([0],VehicleArray_aux[2,:],[0]))    
        VehicleArray_aux2 = np.array([[0,0,0,0,0],aux0,aux1,aux2,[0,0,0,0,0]])  #no boundary cases
        pos_HV = np.array([1,1])
      
    pos_aux =  pos_HV+[1,1] # new position in 5x5 array (5x5 not necessary, just make it uniform with context)
    
    pos_aux_acc = pos_aux + [-1,0]
    if VehicleArray_aux2[pos_aux_acc[0],pos_aux_acc[1]] == 0:
        acc = 1
    else:
        acc = 0 
    pos_aux_brk = pos_aux + [1,0]
    if VehicleArray_aux2[pos_aux_brk[0],pos_aux_brk[1]] == 0:
        brk = 1
    else:
        brk = 0      
    pos_aux_left = pos_aux + [0,-1]
    if VehicleArray_aux2[pos_aux_left[0],pos_aux_left[1]] == 0:
        left = 1
    else:
        left = 0
        
    pos_aux_right = pos_aux + [0,1]
    if VehicleArray_aux2[pos_aux_right[0],pos_aux_right[1]] == 0:
        right = 1
    else:
        right = 0   
    
    return acc, brk, left, right

#==============================================================================
def Q2Policy(Q_value, ActionSet):
    a,b,c = Q_value.shape
    policy_act = []
    
    for j in range(a):
        column = []
        for i in range(c):
            column.append([])
        policy_act.append(column)
        
        policy_index_aux = np.zeros([a,c])
        policy_index = policy_index_aux.astype(int)
    
    for i in range(a):
        for j in range(c):
            Q_cur_state = Q_value[i,:,j]
            max_Q_cur_state = max(Q_cur_state)            
            best_action_indices = [p for p, q in enumerate(Q_cur_state) if q == max_Q_cur_state]    # best action direved from maximizing Q values of current state

            aux = best_action_indices[0]
            policy_index[i,j] = aux  # just use the first action as policy
            policy_act[i][j] = ActionSet[policy_index[i,j]]
        
    return policy_act, policy_index
#==============================================================================

class NeuralNetwork:

    def __init__(self, layers, S, ActionSet, activation='tanh'):  # make S 2-D,  (3*320,10)
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_deriv = sigmoid_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        # Set weights
        self.weights = []
        
        self.layers = layers
        layers_aux = np.asanyarray(layers)+1
        weight_num = sum(layers_aux[0:-1]*layers[1:len(layers)])
        
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2)) : 3 x 2, with bias in input & hidden layers
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i])) -1
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r) # record all weight matrix: input_row * weight_matrix =  row_output

        self.output_deriv = self.weights + [] # check if it is array or list
        self.output_deriv_matrix = np.zeros((S.shape[0]*len(ActionSet),weight_num))
        

    def TrainNNBackProp(self, X, Y, learning_rate=0.2, epochs=100000): #epochs=5000000
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1]+1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer. [index is up to -2]
        X = temp
        Y = np.array(Y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0]) # i is random int upto lines of X
            a = [X[i]] # pick a row of input, random, with bias (output of input layer)

            for l in range(len(self.weights)):
                hidden_inputs = np.ones([self.weights[l].shape[1] + 1])
                hidden_inputs[0:-1] = self.activation(np.dot(a[l], self.weights[l]))
                a.append(hidden_inputs) # append a, with bias at the end, net_outputs of [input,hidden, output layers]
            error = Y[i] - a[-1][:-1] # a[-1][:-1]: output of the output layer, ignore bias 
            deltas = [error * self.activation_deriv(a[-1][:-1])]
            l = len(a) - 2

            # The last layer before the output is handled separately because of
            # the lack of bias node in output
            deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))

            for l in range(len(a)-3, 0, -1): # we need to begin at the second to last layer
                deltas.append(deltas[-1][:-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))  

            deltas.reverse()
            for i in range(len(self.weights)-1):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta[:,:-1])  # layer.T.dot(delta[:,:-1]) is matrix
                
            # Handle last layer separately because it doesn't have a bias unit
            i+=1
            layer = np.atleast_2d(a[i])
            delta = np.atleast_2d(deltas[i])
            self.weights[i] += learning_rate * layer.T.dot(delta)
            
    def UpdWeiMaxEnt(self, S, ActionSet, mu_D, E_mu, learning_rate=0.2, L1_reg = -0.0, L2_reg = -0.0 ): #S is state table, 2D; Reward_SA: state_index*action*curvature

#==============================================================================
#        X = np.atleast_2d(X)
#        X = np.zeros((S.shape[0]*S.shape[2],S.shape[1]))
#        X[0:S.shape[0],:] = S[:,:,0] 
#        X[S.shape[0]:S.shape[0]*2,:] = S[:,:,1] 
#        X[S.shape[0]*2:S.shape[0]*3,:] = S[:,:,2] 
        
        temp = np.ones([S.shape[0], S.shape[1]+1])
        temp[:, 0:-1] = S  # adding the bias unit to the input layer. [index is up to -2]
        X = temp

        # Reward_SA is useless here, since we don't use error function Error(output)        
#        Y = np.zeros((Reward_SA.shape[0]*Reward_SA.shape[2],Reward_SA.shape[1]))
#        Y[0:Reward_SA.shape[0],:] = Reward_SA[:,:,0] 
#        Y[Reward_SA.shape[0]:Reward_SA.shape[0]*2,:] = Reward_SA[:,:,1] 
#        Y[Reward_SA.shape[0]*2:Reward_SA.shape[0]*3,:] = Reward_SA[:,:,2]  
                

        for j in range(len(ActionSet)):  # actions
            for k in range(X.shape[0]):
                
                output_deriv_aux = []
                
                a = [X[k]] # pick a row of input, random, with bias (output of input layer).  CHECK IF A ROW
                    
                for l in range(len(self.weights)):
                    hidden_inputs = np.ones([self.weights[l].shape[1] + 1])
                    hidden_inputs[0:-1] = self.activation(np.dot(a[l], self.weights[l]))
                    a.append(hidden_inputs) # append a, with bias at the end, net_outputs of [input,hidden, output layers]
    #            error = y[i] - a[-1][:-1] # a[-1][:-1]: output of the output layer, ignore bias 
                FakeOutSwitch = np.zeros(len(ActionSet))
                FakeOutSwitch[j] = 1
                deltas = [FakeOutSwitch * self.activation_deriv(a[-1][:-1])]
                l = len(a) - 2  # index of the last 2nd layer
    
                # The last layer before the output is handled separately because of
                # the lack of bias node in output
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
    
                for l in range(len(a) -3, 0, -1): # we need to begin at the second to last layer
                    deltas.append(deltas[-1][:-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
    
                deltas.reverse()
                
                for i in range(len(self.weights)-1):
                    layer = np.atleast_2d(a[i]) # take i th layer output row vector 
                    delta = np.atleast_2d(deltas[i])
                    self.output_deriv[i] = layer.T.dot(delta[:,:-1]) # this is correct
                    output_deriv_aux=np.concatenate([output_deriv_aux,np.concatenate(nn.output_deriv[i].T)])
                    
                # Handle last layer separately because it doesn't have a bias unit
                i+=1
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.output_deriv[i] = layer.T.dot(delta)
                output_deriv_aux=np.concatenate([output_deriv_aux,np.concatenate(nn.output_deriv[i].T)])
        
                self.output_deriv_matrix[j*X.shape[0]+k,:] =  output_deriv_aux + 0
               
#============================== learn the weights ============================= 

#        rescale = np.exp(learning_rate*(mu_D-E_mu).dot(self.output_deriv_matrix)) # watch dimensions
#        index = 0
#        for i in range(len(self.layers)-1):
#            rows = self.weights[i].shape[0]
#            columns = self.weights[i].shape[1]  
#            Aux = rescale[0,index:index+rows*columns] + 0
#            Aux2 = self.weights[i]+0
#            self.weights[i] *= np.reshape(Aux,(columns,rows)).T 
#            self.weights[i] += L2_reg*Aux2    
#            index += rows*columns
 
        rescale = learning_rate*(mu_D-E_mu).dot(self.output_deriv_matrix) # watch dimensions
        index = 0
        for i in range(len(self.layers)-1):
            rows = self.weights[i].shape[0]
            columns = self.weights[i].shape[1]  
            Aux = rescale[0,index:index+rows*columns] + 0 
            self.weights[i] += ( np.reshape(Aux,(columns,rows)).T+L1_reg*np.sign(self.weights[i])+L2_reg*self.weights[i] ) 
            index += rows*columns           
            
            
    def predict(self, x):
        a = np.array(x)
        for j in range(0, len(self.weights)):
            temp = np.ones(a.shape[0]+1)
            temp[0:-1] = a
            a = self.activation(np.dot(temp, self.weights[j]))
        return a

#===========================  begin define EV  ================================     
class EV(object):
    def __init__(self, Available_actions_fn, big_state_matrix, ActionSet, Pr_actions, big_pos, HV_lane, NrLanes):
        self.Pr_stay = Pr_actions[-1]
        self.Pr_acc = Pr_actions[-2]
        self.Pr_left = Pr_actions[-3]
        self.Pr_right = Pr_actions[-4]
        self.Pr_brake = Pr_actions[-5]      
        self.Available_actions_fn=Available_actions_fn
        self.big_pos = big_pos
#        self.color_index = color_index
#        self.Color = ColorSet[self.color_index]
        self.Action = 'stay'       
        self.lane = big_pos[1]
        
    def TakeAction(self,big_state_matrix,HV_lane,NrLanes,ActionSet,Pr_actions):  # use new state when take action everytime, update lane and pos
        big_pos = self.big_pos
        self.Pr_stay = Pr_actions[-1]
        self.Pr_acc = Pr_actions[-2]
        self.Pr_left = Pr_actions[-3]
        self.Pr_right = Pr_actions[-4]
        self.Pr_brake = Pr_actions[-5]
        
        acc, brk, left, right = self.Available_actions_fn(big_state_matrix,big_pos,HV_lane,NrLanes)
        Pr_sum = sum([brk, right, left, acc, 1]*Pr_actions)  # total = 1 if everything available
       
        self.Pr_stay = self.Pr_stay/Pr_sum
        self.Pr_acc = self.Pr_acc/Pr_sum*acc   # ->0 if acc=0
        self.Pr_left = self.Pr_left/Pr_sum*left
        self.Pr_right = self.Pr_right/Pr_sum*right
        self.Pr_brake = self.Pr_brake/Pr_sum*brk
        
        act = np.random.random() # generate an action number randomly
        eps = 1e-8
        aux1 = self.Pr_stay+eps
        aux2 = aux1+self.Pr_acc+eps
        aux3 = aux2+self.Pr_brake+eps
        aux4 = aux3+self.Pr_left+eps
        aux5 = aux4+self.Pr_right+eps  # aux5 == 1+5*eps
        bins = np.array([0, aux1, aux2, aux3, aux4, aux5])
        index = np.digitize(act,bins)
        self.Action = ActionSet[index-1]
        
        if self.Action=='stay':
            self.big_pos = big_pos
        elif self.Action=='acc':
            self.big_pos = big_pos+[-1,0]
        elif self.Action=='brk':
            self.big_pos = big_pos+[1,0]  
        elif self.Action=='left':
            self.big_pos = big_pos+[0,-1]
            self.lane = self.lane-1
        elif self.Action=='right':
            self.big_pos = big_pos+[0,1]   
            self.lane = self.lane+1
                
#============================= end define EV  ================================= 
#===========================  begin define HV  ================================   
  
class HV_obj(object):
    def __init__(self, Safe_actions_fn, Q_value, Curvature_Class, state, HV_lane, ActionSet, Nr_action, epsilon=1e-3, alpha=0.5, gamma=0.9, NrLanes=5):   # for demonstraion purpose, check available actions so that no collision happens 
                                                       # modify Available_actions_fn with all actions for other purpose
        self.Q_value = Q_value
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.HV_lane = HV_lane
        self.collision = 0
        self.Action = 'stay'
        self.Safe_actions_fn = Safe_actions_fn
        self.Curvature_Class = Curvature_Class
        
        if state[0] == 1:
            self.HV_pos = np.array([1,0])
            self.HV_big_pos = np.array([2,0])
        elif state[0] == 2:
            self.HV_pos = np.array([1,1])
            self.HV_big_pos = np.array([2,HV_lane])            
        else:
            self.HV_pos = np.array([1,2])
            self.HV_big_pos = np.array([2,NrLanes-1])
            
    def TakeAction_HV(self,state,RewardVector,Nr_action,ActionSet):
#            state_row_index = 2**8*(state[0]-1)+sum( np.array([2**7,2**6,2**5,2**4,2**3,2**2,2,1])*state[1:9] )
        if state[0] == 1:
            state_row_index = sum( np.array([2**4,2**3,0,2**2,0,2,1,0])*state[1:9] )
        elif state[0] == 2:
            state_row_index = 2**5+sum( np.array([2**7,2**6,2**5,2**4,2**3,2**2,2,1])*state[1:9] )
        elif state[0] == 3:
            state_row_index = 2**5+2**8+sum( np.array([0,2**4,2**3,0,2**2,0,2,1])*state[1:9] )
            
        Q_cur_state = self.Q_value[int(round(state_row_index)),:,self.Curvature_Class]
        max_Q_cur_state = max(Q_cur_state)
        
        best_action_indices = [i for i, j in enumerate(Q_cur_state) if j == max_Q_cur_state]    # best action direved from maximizing Q values of current state
        
        act = np.random.random() # generate an action number randomly, to use epsilon-greedy
        
        if len(best_action_indices) == Nr_action:
            bins = np.linspace(0,1,Nr_action+1)
            index = np.digitize(act,bins)
            self.Action = ActionSet[index-1]
            action_index = index-1
        else:
            other_action_indices = [item for item in [0,1,2,3,4] if item not in best_action_indices]
            bin1 = np.linspace(0,1-self.epsilon,len(best_action_indices)+1)
            bin2 = np.linspace(1-self.epsilon,1,len(other_action_indices)+1)
            bins = np.concatenate((bin1,bin2[1:len(bin2)]))
            index = np.digitize(act,bins) 
            if index <= len(bin1)-1:
                action_index = best_action_indices[index-1]
                self.Action = ActionSet[action_index]
            else:
                action_index = other_action_indices[index-len(bin1)]
                self.Action = ActionSet[action_index]
                
        self.Q_index = [int(round(state_row_index)),action_index]  # the position of Q to update
        
#        Reward,Feature = self.Reward_fn(state,self.HV_lane,self.Action,self.theta,self.Curvature_Class,NrLanes)
        
        acc, brk, left, right = self.Safe_actions_fn(state)
        Safe_actions_set = ['stay']
        if acc == 1:
            Safe_actions_set.append('acc')
        if brk == 1:
            Safe_actions_set.append('brk')
        if left == 1:
            Safe_actions_set.append('left')
        if right == 1:
            Safe_actions_set.append('right')
            
        if self.Action in Safe_actions_set:
            self.collision = 0
        else:
            self.collision = 1
        
        index_aux = ActionSet.index(self.Action)
        self.ImmediateReward = RewardVector[index_aux]

        pos = self.HV_pos
        big_pos = self.HV_big_pos
        if self.Action=='stay':
            self.HV_pos = pos
            self.HV_big_pos = big_pos
        elif self.Action=='acc':
            self.HV_pos = pos+[-1,0]
            self.HV_big_pos = big_pos+[-1,0]
        elif self.Action=='brk':
            self.HV_pos = pos+[1,0]  
            self.HV_big_pos = big_pos+[1,0]
        elif self.Action=='left':
            self.HV_pos = pos+[0,-1]
            self.HV_big_pos = big_pos+[0,-1]
            self.HV_lane = self.HV_lane-1
        elif self.Action=='right':
            self.HV_pos = pos+[0,1]   
            self.HV_big_pos = big_pos+[0,1]
            self.HV_lane = self.HV_lane+1
                
#=============================== end define HV ================================

if __name__ == '__main__':   #global variable indicates the entry point of the program

    np.random.seed(0)  # make it reproducible
    Nr_action = 5
    Pr_actions = np.random.rand(Nr_action)    
    Pr_actions = Pr_actions/sum(Pr_actions)  # normalize action probabilities
    Pr_actions.sort()  # increasing order
    NrLanes = 5  # road of 5 lanes, 0~4 from left to right
    FixVehicleNr =  False
    
    ActionSet = ['stay','acc','brk','left','right']
    mini_state = np.loadtxt('IRL_temp_folder/save_MinState_table.txt')
    demo_states = np.load('IRL_temp_folder/save_demo_states.npy')   # contain curvature
    demo_actions = np.load('IRL_temp_folder/save_demo_actions.npy')   # action indices
#    print(demo_states.shape)

#   define whole state table    
    S = np.zeros((mini_state.shape[0]*3,mini_state.shape[1]+1))
    S[0:mini_state.shape[0],0:-1] = mini_state
    S[mini_state.shape[0]:mini_state.shape[0]*2,0:-1] = mini_state
    S[mini_state.shape[0]*2:mini_state.shape[0]*3,0:-1] = mini_state  
   
    S[0:mini_state.shape[0],-1] = np.zeros((mini_state.shape[0]))   # left
    S[mini_state.shape[0]:mini_state.shape[0]*2,-1] = np.ones((mini_state.shape[0]))  # straight
    S[mini_state.shape[0]*2:mini_state.shape[0]*3,-1] = 2*np.ones((mini_state.shape[0]))  #right

#   calculate mu_D from demos

#    mu_D_S = np.zeros((1,S.shape[0])) # define a row
#    mu_D_SA = np.zeros((1,S.shape[0]*len(ActionSet))) # define a row
#    for j in range(demo_states.shape[1]):   # number of demo trajectories
#        for i in range(demo_states.shape[0]):     # number of demo steps
#            if demo_states[i,j,0] == 1:
#                state_row_index = sum( np.array([2**4,2**3,0,2**2,0,2,1,0])*demo_states[i,j,1:9] )+mini_state.shape[0]*demo_states[i,j,9]
#            elif demo_states[i,j,0] == 2:
#                state_row_index = 2**5+sum( np.array([2**7,2**6,2**5,2**4,2**3,2**2,2,1])*demo_states[i,j,1:9] )+mini_state.shape[0]*demo_states[i,j,9]
#            elif demo_states[i,j,0] == 3:
#                state_row_index = 2**5+2**8+sum( np.array([0,2**4,2**3,0,2**2,0,2,1])*demo_states[i,j,1:9] )+mini_state.shape[0]*demo_states[i,j,9]
#            mu_D_S[0,int(round(state_row_index))] += 1
#            mu_D_SA[0,int(round(state_row_index))+int(round(demo_actions[i,j]))*S.shape[0]] += 1
#    mu_D_S /= demo_states.shape[1]
#    mu_D_SA /= demo_states.shape[1]

#================================== Mac OS=====================================
    mu_D_S = np.load('IRL_temp_folder/save_mu_D_S.npy')
    mu_D_SA = np.load('IRL_temp_folder/save_mu_D_SA.npy')
#================================= Windows ====================================    
#    mu_D_S = np.load('IRL_temp_folder\save_mu_D_S.npy')
#    mu_D_SA = np.load('IRL_temp_folder\save_mu_D_SA.npy')
#==============================================================================
   
#    mu_D_SA_unify = mu_D_SA+0    
#    for i in range(S.shape[0]):
#        if mu_D_S[0,i] == 0:
#            aux = 1
#        else:
#            aux = mu_D_S[0,i]+0
#        mu_D_SA_unify[0,i] = mu_D_SA[0,i]/aux
#        mu_D_SA_unify[0,i+S.shape[0]] = mu_D_SA[0,i+S.shape[0]]/aux
#        mu_D_SA_unify[0,i+S.shape[0]*2] = mu_D_SA[0,i+S.shape[0]*2]/aux
#        mu_D_SA_unify[0,i+S.shape[0]*3] = mu_D_SA[0,i+S.shape[0]*3]/aux
#        mu_D_SA_unify[0,i+S.shape[0]*4] = mu_D_SA[0,i+S.shape[0]*4]/aux

#================================== Mac OS=====================================
    mu_D_SA_unify = np.load('IRL_temp_folder/mu_D_SA_unify.npy')
#================================= Windows ====================================         
#    mu_D_SA_unify = np.load('IRL_temp_folder\mu_D_SA_unify.npy')
#==============================================================================
    
#   Initialize NN reward function
    nn_layers = [10,20,20,20,5]
    nn = NeuralNetwork(nn_layers,S,ActionSet)
    
#   Try Supervised Learning to get weights

#    Demo_Y = np.zeros((S.shape[0],len(ActionSet)))   
#    for i in range(S.shape[0]):
#        Demo_Y[i,0] = mu_D_SA_unify[0,i]
#        Demo_Y[i,1] = mu_D_SA_unify[0,i+S.shape[0]]     
#        Demo_Y[i,2] = mu_D_SA_unify[0,i+S.shape[0]*2]  
#        Demo_Y[i,3] = mu_D_SA_unify[0,i+S.shape[0]*3]
#        Demo_Y[i,4] = mu_D_SA_unify[0,i+S.shape[0]*4]

#================================== Mac OS=====================================
    Demo_Y = np.load('IRL_temp_folder/Demo_Y.npy') 
#================================= Windows ====================================         
#    Demo_Y = np.load('IRL_temp_folder\Demo_Y.npy') 
#==============================================================================
    # initialize NN
    nn.TrainNNBackProp(S, Demo_Y, 0.1)  
    Pred_Y = np.zeros((S.shape[0],len(ActionSet)))
    for i in range(S.shape[0]):
        Pred_Y[i,:] = nn.predict(S[i,:])
    print('Train error is {:f}'.format(np.linalg.norm(Demo_Y-Pred_Y)))
    print('========= Epoch {:d} begin ========='.format(1))    

#================================== Mac OS===================================== 
#    for i in range(len(nn.weights)):
#        nn.weights[i] = np.load('IRL_temp_folder/save_reward_fcn{}.npy'.format(i))       
#================================= Windows ====================================  
#    for i in range(len(nn.weights)):
#        nn.weights[i] = np.load('IRL_temp_folder\save_reward_fcn{}.npy'.format(i))
#==============================================================================


#==============================================================================    
#============================== begin IRL here ================================
#==============================================================================

    learn_iters = 200 #demo_states.shape[0]
    Curve_Last_Steps = 5
    
#    T_count = np.zeros((S.shape[0],S.shape[0],len(ActionSet)))    # define state transition matrix, learned on-line, keeping visit number to each state-action pair
#================================== Mac OS=====================================
    T_count = np.load('IRL_temp_folder/save_T_count.npy') 
#================================= Windows ====================================         
#    T_count = np.load('IRL_temp_folder\save_T_count.npy') 
#==============================================================================

    # Initialize Q-learning 
    Initial_state = np.array([2,1,0,1,0,1,0,1,0 ])
    Initial_state_matrix = state2matrix(Initial_state)
    aux0 = np.concatenate(([0],Initial_state_matrix[0,:],[0]))
    aux1 = np.concatenate(([0],Initial_state_matrix[1,:],[0]))
    aux2 = np.concatenate(([0],Initial_state_matrix[2,:],[0]))    
    Initial_big_state_matrix = np.array([[1,0,0,0,1],aux0,aux1,aux2,[0,1,0,1,0]])

    Curvature_Class = 0  
   
    Initial_HV_lane = NrLanes-3    # HV_lane range from 0-4
    Nr_action = 5   # total number of actions 

    # initialize E_mu_S: state visit count, E_mu_SA: state-action visit count     
    Initial_E_mu_S = np.zeros((1,S.shape[0])) # define a row   
    if Initial_state[0] == 1:
        Initial_state_row_index = sum( np.array([2**4,2**3,0,2**2,0,2,1,0])*Initial_state[1:9] )+mini_state.shape[0]*Curvature_Class
    elif Initial_state[0] == 2:
        Initial_state_row_index = 2**5+sum( np.array([2**7,2**6,2**5,2**4,2**3,2**2,2,1])*Initial_state[1:9] )+mini_state.shape[0]*Curvature_Class
    elif Initial_state[0] == 3:
        Initial_state_row_index = 2**5+2**8+sum( np.array([0,2**4,2**3,0,2**2,0,2,1])*Initial_state[1:9] )+mini_state.shape[0]*Curvature_Class
    Initial_E_mu_S[0,Initial_state_row_index] = 1
    
    for n in range(learn_iters):

#============================= begin Q-learning ===============================       
        # Initialize Q-learning            
              
        epsilon = 8e-2
        alpha = 0.75
        gamma = 0.5  
        
        Q_value = np.zeros((2**8+2*2**5,Nr_action,3)) # maintain a full Q table       

        # implement Q-learning
        converge = False
        Episode = 0
        criterion_max = 1e-2
        
        while not converge:
            
            Episode_over = False
            step = 0
            
            Find_EV_indices = np.asarray(np.where(Initial_big_state_matrix == 1 )).T
            Initial_EV = len(Find_EV_indices)   # total number of EVs
            EV_name_list = list(range(Initial_EV))   # name list of active EVs
            EV_pos_list = Find_EV_indices
            
            HV = HV_obj(Safe_actions_HV, Q_value, Curvature_Class, Initial_state, Initial_HV_lane, ActionSet, Nr_action, epsilon, alpha, gamma, NrLanes)               
            EV_set={}
            for x in range(0,Initial_EV):
                EV_pos = np.array(Find_EV_indices[x])
                EV_set["EV{}".format(x)]=EV(Available_actions_EV, Initial_big_state_matrix, ActionSet, Pr_actions, EV_pos, Initial_HV_lane, NrLanes)

            state = Initial_state + 0
            big_state_matrix = Initial_big_state_matrix + 0
#            HV.HV_lane = Initial_HV_lane + 0
#            HV.alpha = alpha + 0
            Q_cur = HV.Q_value + 0 
            
            while not Episode_over:
                
                step=step+1
                Curvature_Class_aux = (step//Curve_Last_Steps)%3
                HV.Curvature_Class = Curvature_Class_aux+0
                
                # 
                state_combine = np.zeros(10)
                state_combine[0:-1] = state+0
                state_combine[-1] = HV.Curvature_Class
                
                RewardVector = nn.predict(state_combine)
                HV_pos_before = HV.HV_big_pos + 0
                HV.TakeAction_HV(state,RewardVector,Nr_action,ActionSet)
                HV_pos_after = HV.HV_big_pos + 0

                if HV.collision == 1:
#                   HV.Q_value[HV.Q_index[0],HV.Q_index[1],] = HV.ImmediateReward+0
                    HV.Q_value[HV.Q_index[0],HV.Q_index[1],] = -10
                    Episode_over = True
                else: 
                    # no collison, get T matrix indices for updating
                    if state[0] == 1:
                        state_row_index_pre = sum( np.array([2**4,2**3,0,2**2,0,2,1,0])*state[1:9] )+mini_state.shape[0]*HV.Curvature_Class
                    elif state[0] == 2:
                        state_row_index_pre = 2**5+sum( np.array([2**7,2**6,2**5,2**4,2**3,2**2,2,1])*state[1:9] )+mini_state.shape[0]*HV.Curvature_Class
                    elif state[0] == 3:
                        state_row_index_pre = 2**5+2**8+sum( np.array([0,2**4,2**3,0,2**2,0,2,1])*state[1:9] )+mini_state.shape[0]*HV.Curvature_Class
                    trans_act_index = ActionSet.index(HV.Action)
                    
                    EV_name_list1 = EV_name_list + []
                    EV_pos_list1 = EV_pos_list + 0  
                    VehicleArray,state_aux,big_state_matrix,EV_name_list,new_EV_order,EV_pos_list,add_EV_pos_list = Add_Vehicle_HV_move(state,big_state_matrix,HV.HV_lane,HV.Action,NrLanes,EV_name_list,EV_pos_list)
                    EV_name_list2 = EV_name_list + []
                    EV_pos_list2 = EV_pos_list + 0
                    
#============== begin update Q-values using expected state ====================
                    # expected next state after HV move, only used for calculate Q_max for next state
                    expected_array = VehicleArray+0
                    if HV.Action == 'acc':
                        expected_array[0,:] = np.array([0,0,0])
                    elif HV.Action == 'brk':
                        expected_array[-1,:] = np.array([0,0,0]) 
                    elif HV.Action == 'left':
                        if HV.HV_lane != 0:
                            expected_array[:,0] = np.array([0,0,0]).transpose()
                        else:
                            expected_array[:,-1] = np.array([0,0,0]).transpose()
                    elif HV.Action == 'right':
                        if HV.HV_lane != NrLanes-1:
                            expected_array[:,-1] = np.array([0,0,0]).transpose()
                        else:
                            expected_array[:,0] = np.array([0,0,0]).transpose()
                    expected_state = matrix2state(expected_array)  
                    
                    # update Q value
                    if expected_state[0] == 1:
                        state_row_index = sum( np.array([2**4,2**3,0,2**2,0,2,1,0])*expected_state[1:9] )
                    elif expected_state[0] == 2:
                        state_row_index = 2**5+sum( np.array([2**7,2**6,2**5,2**4,2**3,2**2,2,1])*expected_state[1:9] )
                    elif expected_state[0] == 3:
                        state_row_index = 2**5+2**8+sum( np.array([0,2**4,2**3,0,2**2,0,2,1])*expected_state[1:9] )
                    
                    Q_nxt_state = HV.Q_value[int(round(state_row_index)),:,HV.Curvature_Class]
                    max_Q_nxt_state = max(Q_nxt_state)
#                   HV.alpha = 1/( 1+step )**0.75
                    aux = (1-HV.alpha)*HV.Q_value[HV.Q_index[0],HV.Q_index[1],HV.Curvature_Class]+HV.alpha*(HV.ImmediateReward+HV.gamma*max_Q_nxt_state)
                    HV.Q_value[HV.Q_index[0],HV.Q_index[1],HV.Curvature_Class] = aux
                    
#================ end update Q-values using expected state ====================

                    # Initialize new EVs    
                    if new_EV_order != [-1]:    # new EVs found due to acc or brk
                        k = 0
                        for x in new_EV_order:
                            EV_pos = add_EV_pos_list[k,:]
                            EV_set["EV{}".format(x)]=EV(Available_actions_EV, big_state_matrix, ActionSet, Pr_actions, EV_pos, HV.HV_lane, NrLanes)
                            k = k+1

                    #   reset HV position
                    if HV.Action == 'brk' or HV.Action == 'stay' or HV.Action == 'acc':
                        HV.HV_big_pos = HV_pos_before + 0   # reset HV pos
                        
                    # add new EVs from front and back to vacants 
                    big_state_matrix,new_EV_front_order,new_EV_back_order,EV_name_list,EV_pos_list,add_EV_pos_front_list,add_EV_pos_back_list = Add_EVs_front_back(big_state_matrix,HV.HV_lane,NrLanes,EV_name_list2,EV_pos_list2)       
                    EV_name_list3 = EV_name_list + []
                    EV_pos_list3 = EV_pos_list + 0 
                        
                    # Initialize new EVs    
                    if new_EV_front_order != [-1]:    # new added EVs found to front
                        k = 0
                        for x in new_EV_front_order:
                            EV_pos = add_EV_pos_front_list[k,:]
                            EV_set["EV{}".format(x)]=EV(Available_actions_EV, big_state_matrix, ActionSet, Pr_actions, EV_pos, HV.HV_lane, NrLanes)
                            k = k+1   
                    if new_EV_back_order != [-1]:    # new added EVs found to back
                        k = 0
                        for x in new_EV_back_order:
                            EV_pos = add_EV_pos_back_list[k,:]
                            EV_set["EV{}".format(x)]=EV(Available_actions_EV, big_state_matrix, ActionSet, Pr_actions, EV_pos, HV.HV_lane, NrLanes)
                            k = k+1  
            
                   #   all original EVs take an action, except for newly found EVs, results stored in big array       
                    VehicleArray_aux = np.zeros((7,7))
                    VehicleArray_aux[1:6,1:6] = big_state_matrix + 0

                    VehicleArray_aux2 = VehicleArray_aux + 0
                    big_state_matrix_aux = big_state_matrix + 0  # keep original posiition of EV after moving, avoiding cross in animation
                           
                    k = 0
                    EV_pos_list_aux = np.zeros((len(EV_name_list2),2)).astype(int)
                    for x in EV_name_list2:
                        EV_pos = np.array(EV_pos_list2[k,:])
                        EV_pos_list_aux[k,:] = EV_set["EV{}".format(x)].big_pos + 0 # check if this works properly
                        EV_set["EV{}".format(x)].big_pos = EV_pos+0  # check if this works, update positon of identified EVs since HV moves
                        k = k+1
                        
                    k = 0
                    for x in EV_name_list2:
                        EV_pos = np.array(EV_pos_list2[k,:])
                        EV_pos_aux = EV_pos+[1,1]
                        VehicleArray_aux[EV_pos_aux[0],EV_pos_aux[1]] = 0 # remove original EV that moves in the big array
                        VehicleArray_aux2[EV_pos_aux[0],EV_pos_aux[1]] = -1  # -1 inidicate there was an EV
            
                        EV_set["EV{}".format(x)].TakeAction(big_state_matrix_aux,HV.HV_lane,NrLanes,ActionSet,Pr_actions) # EV takes action
                        EV_pos_aux = EV_set["EV{}".format(x)].big_pos+[1,1] # new pos in big array
                        
                        VehicleArray_aux[EV_pos_aux[0],EV_pos_aux[1]] = 1 # set new pos in big array in big array = 1
                        big_state_matrix = VehicleArray_aux[1:6,1:6] + 0
                        
                        VehicleArray_aux2[EV_pos_aux[0],EV_pos_aux[1]] = 1
                        big_state_matrix_aux = VehicleArray_aux2[1:6,1:6] + 0  # remember that big_state_matrix_aux is not true big_state_matrix
            
                        k = k+1
######################################### Version 2 END ##########################################        
                                
                    # update new state after all EVs moving            
                    VehicleArray = np.zeros((3,3))
                    if  HV.HV_lane == 0:
                        VehicleArray[:,0:2] = big_state_matrix[1:-1,0:2]
                    elif HV.HV_lane==NrLanes-1:
                        VehicleArray[:,1:3] = big_state_matrix[1:-1,3:5]
                    else:
                        VehicleArray = big_state_matrix[1:NrLanes-1,HV.HV_lane-1:HV.HV_lane+2] 
                        
                    state = matrix2state(VehicleArray) # new state after each EV moves   
                        
                    # ready to update T after renewing the state
                    if state[0] == 1:
                        state_row_index_post = sum( np.array([2**4,2**3,0,2**2,0,2,1,0])*state[1:9] )+mini_state.shape[0]*HV.Curvature_Class
                    elif state[0] == 2:
                        state_row_index_post = 2**5+sum( np.array([2**7,2**6,2**5,2**4,2**3,2**2,2,1])*state[1:9] )+mini_state.shape[0]*HV.Curvature_Class
                    elif state[0] == 3:
                        state_row_index_post = 2**5+2**8+sum( np.array([0,2**4,2**3,0,2**2,0,2,1])*state[1:9] )+mini_state.shape[0]*HV.Curvature_Class
                    
                    T_count[int(round(state_row_index_pre)),int(round(state_row_index_post)),trans_act_index] += 1 
            
                    # remove list elements that disappear from the front and back by acc or brk
                    k = 0
                    list_delete = []
                    pos_row_delete = []
                    pos_replace = np.array([], dtype=np.int64).reshape(0,2)
                    for x in EV_name_list2:
                        if EV_set["EV{}".format(x)].big_pos[0]<0 or EV_set["EV{}".format(x)].big_pos[0]>4:
                            list_delete = list_delete + [x]
                            pos_row_delete = pos_row_delete + [k]
                        elif EV_set["EV{}".format(x)].big_pos[0]>=0 and EV_set["EV{}".format(x)].big_pos[0]<=4:   # replace elements in list3
                            pos_replace = np.concatenate((pos_replace,np.array([[EV_set["EV{}".format(x)].big_pos[0],EV_set["EV{}".format(x)].big_pos[1]]])))
                   
                    EV_name_list = [item for item in EV_name_list3 if item not in list_delete] 
                    
                    if new_EV_front_order == [-1]:
                        EV_pos_list = pos_replace
                    else:
                        EV_pos_list = np.concatenate((add_EV_pos_front_list,pos_replace))
                    if new_EV_back_order != [-1]:
                        EV_pos_list = np.concatenate((EV_pos_list,add_EV_pos_back_list))

            Episode = Episode+1        
            criterion = np.linalg.norm(Q_cur-HV.Q_value)
            if Episode//500 > (Episode-1)//500:
                print('Episode:',Episode,'step:',step,'criterion:{:f}'.format(criterion))
            if  criterion<criterion_max and Episode > 5e3:
                converge = True

#================================== Mac OS=====================================
                np.save('IRL_temp_folder/save_T_count.npy',T_count)
#================================= Windows ====================================         
#                np.save('IRL_temp_folder\save_T_count.npy',T_count)
#==============================================================================
                    
        # obtain policy
        policy_act, policy_index = Q2Policy(HV.Q_value, ActionSet)
        
#======================= save train result every epoch ========================
        Q_value = HV.Q_value + 0
#============================= Q learning done ================================
#==============================================================================  
#==============================================================================  

#        policy_index = np.loadtxt('IRL_temp_folder/save_optimal_policy.txt')  # use optimal polisy to verify T
 
#==============================================================================                  
#=================== policy propagation & calculate E_mu ======================    
#==============================================================================           
        
        policy_index_vector = np.reshape(policy_index.T,(S.shape[0]))   # this is a row            
#        E_mu_SA_matrix = np.zeros((S.shape[0]*len(ActionSet),demo_states.shape[0])).T  
        E_mu_SA = np.zeros((1,S.shape[0]*len(ActionSet)))
        for j in range(S.shape[0]):
            E_mu_SA[0,j+S.shape[0]*int(round(policy_index_vector[j]))] = 1   # use the learned policy, unify the states
                
        T = T_count + 0
        for j in range(T_count.shape[0]):  # state
            for k in range(T_count.shape[2]):  #action
                if sum(T_count[j,:,k]) == 0:
                    Aux = 1
                else:
                    Aux = sum(T_count[j,:,k])
                T[j,:,k] = T_count[j,:,k]/Aux  # normalized probability conditioned on S&A
        
# policy is deterministic and unique
    
# update nn.weights using MaxEnt 
        
        nn.UpdWeiMaxEnt(S, ActionSet, mu_D_SA_unify, E_mu_SA, 0.005, 0, -0.0001)
        print('State-Action Visitation Count Error: {:f}'.format((mu_D_SA_unify-E_mu_SA).dot((mu_D_SA_unify-E_mu_SA).T)[0][0]))
        print('========== Epoch {:d} end =========='.format(n+1))
        
        for i in range(len(nn.weights)):
#================================== Mac OS=====================================
            np.save('IRL_temp_folder/save_reward_fcn{}.npy'.format(i),nn.weights[i])
#================================= Windows ====================================         
#            np.save('IRL_temp_folder\save_reward_fcn{}.npy'.format(i),nn.weights[i])
#==============================================================================            
            
        if (mu_D_SA_unify-E_mu_SA).dot((mu_D_SA_unify-E_mu_SA).T)[0][0]<2:
            break
#========================== save final train result  ==========================  
        
#================================== Mac OS=====================================       
    np.savetxt('IRL_temp_folder/save_GenerateDemos_MinState_320_left.txt', HV.Q_value[:,:,0], fmt='%1.4e',newline='\r\n') 
    np.savetxt('IRL_temp_folder/save_GenerateDemos_MinState_320_mid.txt', HV.Q_value[:,:,1], fmt='%1.4e',newline='\r\n') 
    np.savetxt('IRL_temp_folder/save_GenerateDemos_MinState_320_right.txt', HV.Q_value[:,:,2], fmt='%1.4e',newline='\r\n') 

    np.savetxt('IRL_temp_folder/save_optimal_policy.txt', policy_index, fmt='%d',newline='\r\n')   
    np.savetxt('animation/save_optimal_policy.txt', policy_index, fmt='%d',newline='\r\n')
#================================= Windows ====================================     
#    np.savetxt('IRL_temp_folder\save_GenerateDemos_MinState_320_left.txt', HV.Q_value[:,:,0], fmt='%1.4e',newline='\r\n') 
#    np.savetxt('IRL_temp_folder\save_GenerateDemos_MinState_320_mid.txt', HV.Q_value[:,:,1], fmt='%1.4e',newline='\r\n') 
#    np.savetxt('IRL_temp_folder\save_GenerateDemos_MinState_320_right.txt', HV.Q_value[:,:,2], fmt='%1.4e',newline='\r\n') 
#
#    np.savetxt('IRL_temp_folder\save_optimal_policy.txt', policy_index, fmt='%d',newline='\r\n')   
#    np.savetxt('animation\save_optimal_policy.txt', policy_index, fmt='%d',newline='\r\n') 
#==============================================================================     