# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:25:51 2017
@author: CYOU7

"""

from video import make_video
import pygame, sys
from pygame.locals import*
from math import atan
import random as rd
import cvxopt as co
import cvxpy as cp
import math as mh
import matplotlib.pyplot as plt
from numpy import arctan2 as atan2
import numpy as np

np.random.seed(0)  # make it reproducible
Nr_action = 5
ActionSet = ['stay','acc','brk','left','right']
ColorSet = ['BlueCar','RedCar','PinkCar','YellowCar','GreyCar','PurpleCar','BlueTruck','RedTruck','YellowTruck','GreenTruck','GreyTruck']
ColorSet_HV = ['GreenCar_HV']

#Pr_actions = np.random.rand(Nr_action)    
#Pr_actions = Pr_actions/sum(Pr_actions)  # normalize action probabilities
#Pr_actions.sort()  # increasing order
Pr_actions = np.array([0.05, 0.05, 0.05, 0.3, 0.55])

NrLanes = 5  # road of 5 lanes, 0~4 from left to right 
dt = 0.01
extra_dis = 5 # for lane keeping
g = 9.81
Resize = 130/4.5
Period = int(1.0/dt)  # number of steps to add EVs, every 1 sec

maintain_Time = 1.0

display_width = 800 #1000
display_height = 1200 #1000
width = 110
length = 175
car_width = 66
car_length = 130
road_width = 1048
road_length = 2200

WidthSet=[car_width]*6+[74,68,72,75,70]
LengthSet=[car_length]*6+[197,140,197,172,185]

road_margin = road_length-display_height

circuit_edge = 3016
circuit_edge = 3008

R3 = circuit_edge-road_width/2
R1 = R3 - 2*width
R2 = R3 - width
R4 = R3 + width
R5 = R3 + 2*width

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
# available actions of each EV
def Available_actions_EV(state_matrix):  # fake the state matrix for each EV (in center)

    pos_aux = np.array([1,1])
    
    pos_aux_acc = pos_aux + [-1,0]
    if state_matrix[pos_aux_acc[0],pos_aux_acc[1]] == 0:
        acc = 1
    else:
        acc = 0 
    pos_aux_brk = pos_aux + [1,0]
    if state_matrix[pos_aux_brk[0],pos_aux_brk[1]] == 0:
        brk = 1
    else:
        brk = 0      
    pos_aux_left = pos_aux + [0,-1]
    if state_matrix[pos_aux_left[0],pos_aux_left[1]] == 0:
        left = 1
    else:
        left = 0
        
    pos_aux_right = pos_aux + [0,1]
    if state_matrix[pos_aux_right[0],pos_aux_right[1]] == 0:
        right = 1
    else:
        right = 0   

    return acc, brk, left, right
#==============================================================================
#   get state and front vehicle
def get_state_EV(EV_set,EV_name_set,Car):   # suppose car is not in EV_set and EV_name_list
    Car_front=None
    VehicleArray =np.zeros((3,3))
    EV_set_aux={x for x in EV_name_set if abs(EV_set["EV{}".format(x)].lane-Car.lane)<=1 or abs(EV_set["EV{}".format(x)].lane_tg-Car.lane)<=1 or abs(EV_set["EV{}".format(x)].lane-Car.lane_tg)<=1 or abs(EV_set["EV{}".format(x)].lane_tg-Car.lane_tg)<=1}

    VehicleArray[1][1]=2
    
    VehicleArray[0][0]=Car.lane==0 or len({x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0]-width,Car.pos[1]+0.5*Car.car_length-1.5*Car.grid_len*Resize)} ) is not 0
    VehicleArray[1][0]=Car.lane==0 or len({x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0]-width,Car.pos[1]+0.5*Car.car_length-0.5*Car.grid_len*Resize)} ) is not 0
    VehicleArray[2][0]=Car.lane==0 or len({x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0]-width,Car.pos[1]+0.5*Car.car_length+0.5*Car.grid_len*Resize)} ) is not 0

    VehicleArray[0][1]=len( {x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0],Car.pos[1]-Car.grid_len*Resize,Car.grid_len*Resize-Car.car_length)} ) is not 0
    VehicleArray[2][1]=len( {x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0],Car.pos[1]+0.5*Car.car_length+0.5*Car.grid_len*Resize)} ) is not 0

    VehicleArray[0][2]=Car.lane==NrLanes-1 or len({x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0]+width,Car.pos[1]+0.5*Car.car_length-1.5*Car.grid_len*Resize)} ) is not 0
    VehicleArray[1][2]=Car.lane==NrLanes-1 or len({x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0]+width,Car.pos[1]+0.5*Car.car_length-0.5*Car.grid_len*Resize)} ) is not 0
    VehicleArray[2][2]=Car.lane==NrLanes-1 or len({x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0]+width,Car.pos[1]+0.5*Car.car_length+0.5*Car.grid_len*Resize)} ) is not 0

    for x in EV_set_aux:
        EV_set_aux2={x for x in EV_set_aux if (abs(EV_set["EV{}".format(x)].pos[0]-Car.pos[0])<=width/2.0+EV_set["EV{}".format(x)].car_width/2.0 or EV_set["EV{}".format(x)].lane_tg==Car.lane) and EV_set["EV{}".format(x)].pos[1]<Car.pos[1]}
        if len(EV_set_aux2)>0:
            EV_pos_set={EV_set["EV{}".format(x)].pos[1] for x in EV_set_aux2}
            Car_front_name = list(x for x in EV_set_aux2 if EV_set["EV{}".format(x)].pos[1]==max(EV_pos_set))[0]
            Car_front=EV_set["EV{}".format(Car_front_name)]
            
    return VehicleArray,Car_front
#==============================================================================
#   get state and front vehicle
def get_state_HV(EV_set,EV_name_set,Car):   # suppose car is not in EV_set and EV_name_list
    Car_front=None
    VehicleArray =np.zeros((3,3))
    EV_set_aux={x for x in EV_name_set if abs(EV_set["EV{}".format(x)].lane-Car.lane)<=1 or abs(EV_set["EV{}".format(x)].lane_tg-Car.lane)<=1 or abs(EV_set["EV{}".format(x)].lane-Car.lane_tg)<=1 or abs(EV_set["EV{}".format(x)].lane_tg-Car.lane_tg)<=1}

    if Car.lane==0:
        VehicleArray[1][0]=2
        
        VehicleArray[0][0]=len( {x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0],Car.pos[1]-Car.grid_len*Resize,Car.grid_len*Resize-Car.car_length)} ) is not 0
        VehicleArray[2][0]=len( {x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0],Car.pos[1]+0.5*Car.car_length+0.5*Car.grid_len*Resize)} ) is not 0
    
        VehicleArray[0][1]=len( {x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0]+width,Car.pos[1]+0.5*Car.car_length-1.5*Car.grid_len*Resize)} ) is not 0
        VehicleArray[1][1]=len( {x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0]+width,Car.pos[1]+0.5*Car.car_length-0.5*Car.grid_len*Resize)} ) is not 0
        VehicleArray[2][1]=len( {x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0]+width,Car.pos[1]+0.5*Car.car_length+0.5*Car.grid_len*Resize)} ) is not 0
        
    elif Car.lane==NrLanes-1:
        VehicleArray[1][2]=2

        VehicleArray[0][1]=len( {x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0]-width,Car.pos[1]+0.5*Car.car_length-1.5*Car.grid_len*Resize)} ) is not 0
        VehicleArray[1][1]=len( {x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0]-width,Car.pos[1]+0.5*Car.car_length-0.5*Car.grid_len*Resize)} ) is not 0
        VehicleArray[2][1]=len( {x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0]-width,Car.pos[1]+0.5*Car.car_length+0.5*Car.grid_len*Resize)} ) is not 0
    
        VehicleArray[0][2]=len( {x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0],Car.pos[1]-Car.grid_len*Resize,Car.grid_len*Resize-Car.car_length)} ) is not 0
        VehicleArray[2][2]=len( {x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0],Car.pos[1]+0.5*Car.car_length+0.5*Car.grid_len*Resize)} ) is not 0
        
    else:
        VehicleArray[1][1]=2
        
        VehicleArray[0][0]=len( {x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0]-width,Car.pos[1]+0.5*Car.car_length-1.5*Car.grid_len*Resize)} ) is not 0
        VehicleArray[1][0]=len( {x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0]-width,Car.pos[1]+0.5*Car.car_length-0.5*Car.grid_len*Resize)} ) is not 0
        VehicleArray[2][0]=len( {x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0]-width,Car.pos[1]+0.5*Car.car_length+0.5*Car.grid_len*Resize)} ) is not 0
    
        VehicleArray[0][1]=len( {x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0],Car.pos[1]-Car.grid_len*Resize,Car.grid_len*Resize-Car.car_length)} ) is not 0
        VehicleArray[2][1]=len( {x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0],Car.pos[1]+0.5*Car.car_length+0.5*Car.grid_len*Resize)} ) is not 0
    
        VehicleArray[0][2]=len( {x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0]+width,Car.pos[1]+0.5*Car.car_length-1.5*Car.grid_len*Resize)} ) is not 0
        VehicleArray[1][2]=len( {x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0]+width,Car.pos[1]+0.5*Car.car_length-0.5*Car.grid_len*Resize)} ) is not 0
        VehicleArray[2][2]=len( {x for x in EV_set_aux if overlap_area(EV_set["EV{}".format(x)],Car,Car.pos[0]+width,Car.pos[1]+0.5*Car.car_length+0.5*Car.grid_len*Resize)} ) is not 0

    for x in EV_set_aux:
        EV_set_aux2={x for x in EV_set_aux if (abs(EV_set["EV{}".format(x)].pos[0]-Car.pos[0])<=width/2.0+EV_set["EV{}".format(x)].car_width/2.0 or EV_set["EV{}".format(x)].lane_tg==Car.lane) and EV_set["EV{}".format(x)].pos[1]<Car.pos[1]}
        if len(EV_set_aux2)>0:
            EV_pos_set={EV_set["EV{}".format(x)].pos[1] for x in EV_set_aux2}
            Car_front_name = list(x for x in EV_set_aux2 if EV_set["EV{}".format(x)].pos[1]==max(EV_pos_set))[0]
            Car_front=EV_set["EV{}".format(Car_front_name)]
            
    state_HV=matrix2state(VehicleArray)
    
    return state_HV,Car_front

#==============================================================================
# get center and length of EV, return overlap or not
def overlap_area(EV,Car,grid_x,grid_y,delta_y=0.0):
    x_pos_array=[Center_x-width*2.0,Center_x-width,Center_x,Center_x+width,Center_x+width*2.0]
    width_aux = width/2.0+car_width/2.0+0.0 
    if EV.Action=='left' or EV.Action=='right':
        x_center = (EV.pos[0]+x_pos_array[EV.lane_tg])/2.0
        width_aux += abs(EV.pos[0]-x_pos_array[EV.lane_tg])/2.0
    else:
        x_center = EV.pos[0]
    y_center = EV.pos[1]+EV.car_length/2.0-0.5*EV.grid_len*Resize-0.5*(EV.vel-Car.vel)*Car.Dt*Resize
    occu_len = EV.grid_len*Resize+abs(EV.vel-Car.vel)*Car.Dt*Resize
    
    return abs(x_center-grid_x)<width_aux and abs(y_center-grid_y)<(Car.grid_len*Resize+delta_y+occu_len)/2.0

#==============================================================================
# add vehicles randomly 
def add_EVs(EV_set,EV_name_set,Car):
    FrontArray=np.zeros(NrLanes)
    RearArray=np.zeros(NrLanes)
    EV_set_aux1={x for x in EV_name_set if EV_set["EV{}".format(x)].pos[1]+EV_set["EV{}".format(x)].car_length/2.0<=EV_set["EV{}".format(x)].grid_len*Resize}
    EV_set_aux2={x for x in EV_name_set if EV_set["EV{}".format(x)].pos[1]+EV_set["EV{}".format(x)].car_length/2.0>=display_height-EV_set["EV{}".format(x)].grid_len*Resize}
    
    if len(EV_set_aux1) is not 0:
        for i in range(NrLanes):
            FrontArray[i]=( len({x for x in EV_set_aux1 if overlap_area(EV_set["EV{}".format(x)],Car,Center_x+(i-2)*width,-0.5*Car.grid_len*Resize)}) is not 0 )
    if len(EV_name_set)==0:
        last_car=1
    else:
        last_car=max(EV_name_set)
    indices_front = np.asarray(np.where(FrontArray == 0 ))
    if len(indices_front[0])>0:
        Nr_new_vehicle_front = rd.sample(range(len(indices_front[0])-0), 1)  # pick number of cars to fill
    else: 
        Nr_new_vehicle_front=[0]
        
    if Nr_new_vehicle_front[0]>0:
        new_vehicle_pos_front = rd.sample(range(len(indices_front[0])), Nr_new_vehicle_front[0])
        k = 0
        for i in new_vehicle_pos_front: 
            k = k+1
            EV_name_set.add(last_car+k)
            EV_color_index =  np.random.randint(0, high=len(ColorSet))            
            Pos = [Center_x+(indices_front[0][i]-2)*width,-LengthSet[EV_color_index]/2.0]
            EV_lane=indices_front[0][i]
            Vel=13.0
            Dt=0.7
            EV_set["EV{}".format(last_car+k)]=EV(EV_lane, Pos, Vel, Dt, EV_color_index, Curvature_Class)
            Car_dict[EV_set["EV{}".format(last_car+k)].Color](Pos[0],Pos[1],EV_set["EV{}".format(last_car+k)].car_width,EV_set["EV{}".format(last_car+k)].car_length)                 

#=====================================        
    if len(EV_set_aux2) is not 0:
        for i in range(NrLanes):
            RearArray[i]=len( {x for x in EV_set_aux2 if overlap_area(EV_set["EV{}".format(x)],Car,Center_x+(i-2)*width,display_height+0.5*Car.grid_len*Resize)} ) is not 0   
    
    if len(EV_name_set)==0:
        last_car=1
    else:
        last_car=max(EV_name_set)
        
    indices_rear = np.asarray(np.where(RearArray == 0 ))        
    if len(indices_rear[0])>0:
        Nr_new_vehicle_rear = rd.sample(range(len(indices_rear[0])-0), 1)  # pick number of cars to fill
    else:
        Nr_new_vehicle_rear=[0]

    if Nr_new_vehicle_rear[0]>0:
        new_vehicle_pos_rear = rd.sample(range(len(indices_rear[0])), Nr_new_vehicle_rear[0])
        k = 0
        for i in new_vehicle_pos_rear: 
            k = k+1
            EV_name_set.add(last_car+k)
            EV_color_index = np.random.randint(0, high=6)
            Pos = [Center_x+(indices_rear[0][i]-2)*width,display_height+Car.grid_len*Resize-LengthSet[EV_color_index]/2.0]
            EV_lane=indices_rear[0][i]
            Vel=13.0
            Dt=0.7
            EV_set["EV{}".format(last_car+k)]=EV(EV_lane, Pos, Vel, Dt, EV_color_index, Curvature_Class)
            Car_dict[EV_set["EV{}".format(last_car+k)].Color](Pos[0],Pos[1],EV_set["EV{}".format(last_car+k)].car_width,EV_set["EV{}".format(last_car+k)].car_length)  
        
    return EV_set,EV_name_set

#==============================================================================
# define position mapping for curvature plotting
def mapping2curve(plot_straight,plot_straigt2circuit,plot_circuit,plot_circuit2straight,x_aux,y_aux,circuit_x,circuit_y,Curvature_Class,road_y_transit,road_angle_transit,R1,R2,R3,R4,R5,display_width,display_height):
    if plot_straight == True:
        x_aux2 = x_aux+0
        y_aux2 = y_aux+0
        angle_aux2 = 0
    elif plot_straight == False and plot_straigt2circuit == True:
        if y_aux>=circuit_y:
            x_aux2 = x_aux+0
            y_aux2 = y_aux+0
            angle_aux2 = 0
        elif y_aux<circuit_y:
            delta_L = circuit_y-y_aux
            delta_angle = delta_L/R3
            if Curvature_Class==0:
                delta_x = -R3*(1-mh.cos(delta_angle))
                angle_aux2 = delta_angle/np.pi*180
            elif Curvature_Class==2:
                delta_x = R3*(1-mh.cos(delta_angle))
                angle_aux2 = -delta_angle/np.pi*180
                
            delta_y = delta_L-R3*mh.sin(delta_angle)
            x_aux2 = x_aux+delta_x
            y_aux2 = y_aux+delta_y
    elif plot_straigt2circuit == False and plot_circuit==True:
        delta_L = circuit_y-y_aux
        delta_angle = delta_L/R3
        if Curvature_Class==0:
            delta_x = -R3*(1-mh.cos(delta_angle))
            angle_aux2 = delta_angle/np.pi*180
        elif Curvature_Class==2:
            delta_x = R3*(1-mh.cos(delta_angle))
            angle_aux2 = -delta_angle/np.pi*180
            
        delta_y = delta_L-R3*mh.sin(delta_angle)
        x_aux2 = x_aux+delta_x
        y_aux2 = y_aux+delta_y
    elif plot_circuit == False and plot_circuit2straight == True:
        if y_aux>circuit_y: # below circuit bottom
            delta_L = y_aux-circuit_y
            delta_angle = delta_L/R3
            if road_angle_transit>=0:
                delta_x = -R3*(1-mh.cos(delta_angle))
                angle_aux2 = -delta_angle/np.pi*180
            elif road_angle_transit<0:
                delta_x = R3*(1-mh.cos(delta_angle))
                angle_aux2 = delta_angle/np.pi*180
            delta_y = R3*mh.sin(delta_angle)-delta_L
            x_aux2 = x_aux+delta_x
            y_aux2 = y_aux+delta_y 
        elif display_height-y_aux<np.abs(road_angle_transit/180*np.pi)*R3 and y_aux<=circuit_y: # in circuit
            delta_L = circuit_y-y_aux
            delta_angle = delta_L/R3
            if road_angle_transit>=0:
                delta_x = -R3*(1-mh.cos(delta_angle))
                angle_aux2 = delta_angle/np.pi*180
            elif road_angle_transit<0:
                delta_x = R3*(1-mh.cos(delta_angle))
                angle_aux2 = -delta_angle/np.pi*180
            delta_y = delta_L-R3*mh.sin(delta_angle)
            x_aux2 = x_aux+delta_x
            y_aux2 = y_aux+delta_y
        elif display_height-y_aux>=np.abs(road_angle_transit/180*np.pi)*R3: # in straight
            delta_L = (display_height-y_aux)-np.abs(road_angle_transit/180*np.pi)*R3
            if road_angle_transit>=0:
                delta_x = -R3*(1-mh.cos(road_angle_transit/180*np.pi))-delta_L*mh.sin(road_angle_transit/180*np.pi)
            elif road_angle_transit<0:
                delta_x = R3*(1-mh.cos(road_angle_transit/180*np.pi))-delta_L*mh.sin(road_angle_transit/180*np.pi)
            delta_y = -delta_x*mh.tan(road_angle_transit/180*np.pi)
            x_aux2 = x_aux+delta_x
            y_aux2 = y_aux+delta_y
            angle_aux2 = road_angle_transit
                   
    return x_aux2,y_aux2,angle_aux2

#==============================================================================
def bound(x,x1,x2):
    return min( max(x,x1), x2 )

#==============================================================================
def path_planning(pos0,k_max,Lane_width,direction): #direction>0 -> left
    K=(( (1+144/Lane_width**2/k_max**2)**0.5-1 )/2)**0.5
    Length=K*Lane_width
    D0 = (1+K**2)**0.5/4/K*Lane_width
    L0 = 2*K*D0
    
    XI0_offset=pos0[0]/Resize
    YI0_offset=pos0[1]/Resize
    
    XI0=0
    YI0=0
    XIf=XI0-direction*D0
    YIf=YI0+L0
    
    P0 = np.zeros(2)
    P0[0] = XI0
    P0[1] = YI0
    
    P4 = np.zeros(2)
    P4[0] = XIf
    P4[1] = YIf

    P2 = (P0+P4)/2   
    
    Arr = L0*0.25
    
    P1 = P0+[0, Arr]
    P3 = P4-[0, Arr]   
    
    dtau = 0.02
    Number = int(round(1/dtau))+1
    tau = np.linspace(0, 1, num=Number)  

    Aux1 = (1-tau)**4*P0[0]+4*(1-tau)**3*tau*P1[0]+6*(1-tau)**2*tau**2*P2[0]+4*(1-tau)*tau**3*P3[0]+tau**4*P4[0]
    Aux2 = (1-tau)**4*P0[1]+4*(1-tau)**3*tau*P1[1]+6*(1-tau)**2*tau**2*P2[1]+4*(1-tau)*tau**3*P3[1]+tau**4*P4[1]

    Curve_AB = np.zeros((Number,2))
    Curve_AB[:,0] = Aux1
    Curve_AB[:,1] = Aux2    

    D_x = 4*(1-tau)**3*(P1[0]-P0[0])+12*(1-tau)**2*tau*(P2[0]-P1[0])+12*tau**2*(1-tau)*(P3[0]-P2[0])+4*tau**3*(P4[0]-P3[0])
    D_y = 4*(1-tau)**3*(P1[1]-P0[1])+12*(1-tau)**2*tau*(P2[1]-P1[1])+12*tau**2*(1-tau)*(P3[1]-P2[1])+4*tau**3*(P4[1]-P3[1])
    
    DD_x = 12*(1-tau)**2*(P2[0]-2*P1[0]+P0[0])+24*tau*(1-tau)*(P3[0]-2*P2[0]+P1[0])+12*tau**2*(P4[0]-2*P3[0]+P2[0])   
    DD_y = 12*(1-tau)**2*(P2[1]-2*P1[1]+P0[1])+24*tau*(1-tau)*(P3[1]-2*P2[1]+P1[1])+12*tau**2*(P4[1]-2*P3[1]+P2[1])
    
    ka = (D_x*DD_y-D_y*DD_x)/( (D_x**2+D_y**2)**1.5 )
    
    tau_ex = np.linspace(0, 2, num=int(round(2/dtau))+1)/2
    mid=int((len(tau)+1)/2-1)
    Aux_1 = np.append(Aux1[mid:0:-1],Aux1)
    Aux_1 = np.append(Aux_1,Aux1[len(tau)-2:mid-1:-1])

    Aux_2 = np.append(-Aux2[mid:0:-1],Aux2)
    Aux_2 = np.append(Aux_2,2*L0-Aux2[len(tau)-2:mid-1:-1])    
    Curve_AB_Ex = np.zeros((int(round(2/dtau))+1,2)) 
    Curve_AB_Ex[:,0] = Aux_1
    Curve_AB_Ex[:,1] = Aux_2   
 
    ka_Ex = np.append(ka[mid:0:-1],ka)
    ka_Ex = np.append(ka_Ex,ka[len(tau)-2:mid-1:-1])   

# coordinate transform
    aux_1 = Aux_1+direction*D0/2
    aux_2 = Aux_2+L0/2;
    
#    theta = np.pi/2-atan(L0/D0/2);  #Cooridnate CW
    theta = atan(1/K)*direction
    
    Aux_1 = aux_1*mh.cos(theta)-aux_2*mh.sin(theta);
    Aux_2 = aux_1*mh.sin(theta)+aux_2*mh.cos(theta);
    Curve_AB_Ex_Rotate = np.zeros((int(round(2/dtau))+1,2)) 
    Curve_AB_Ex_Rotate[:,0] = Aux_1+XI0_offset
    Curve_AB_Ex_Rotate[:,1] = Aux_2+YI0_offset    

# transformed D_x, D_y
    Aux_1 = np.append(-D_x[mid:0:-1],D_x)
    aux_1 = np.append(Aux_1,-D_x[len(tau)-2:mid-1:-1])    

    Aux_2 = np.append(D_y[mid:0:-1],D_y)
    aux_2 = np.append(Aux_2,D_y[len(tau)-2:mid-1:-1]) 
       
    Aux_1 = aux_1*mh.cos(theta)-aux_2*mh.sin(theta)
    Aux_2 = aux_1*mh.sin(theta)+aux_2*mh.cos(theta)
    psi_t = atan2(Aux_2,Aux_1)
    
#    plt.plot(Curve_AB_Ex_Rotate[:,0], Curve_AB_Ex_Rotate[:,1], 'ro')
#    plt.show()
    
    return tau_ex, Curve_AB_Ex_Rotate, ka_Ex, psi_t, Length
    
    
#===============================================
class EV(object):
    def __init__(self, EV_lane, Pos, Vel, Dt, color_index, Curvature_Class, m=850, h=0.5, lf=1.5, lr=0.9, Iz=1401):
        self.Pr_stay = Pr_actions[-1]
        self.Pr_acc = Pr_actions[-2]
        self.Pr_left = Pr_actions[-3]
        self.Pr_right = Pr_actions[-4]
        self.Pr_brake = Pr_actions[-5]      
        
        self.lane = EV_lane
        self.lane_tg = EV_lane
        self.pos = Pos #[bit]
        self.vel = Vel*1.6/3.6 #[m/s]
        self.vel_tg = Vel*1.6/3.6 #[m/s]
        self.acc = 0.0
        self.Dt = Dt
        self.Action = 'stay'
        self.in_action = False
        self.Curvature_Class = Curvature_Class
        self.color_index = color_index
        self.Color = ColorSet[self.color_index]
        self.car_width=WidthSet[self.color_index]
        self.car_length=LengthSet[self.color_index]  
        self.grid_len=self.car_length/Resize+self.vel*self.Dt        
        self.V_max = 40*1.6/3.6   #Resize=130/4.5
        self.V_min = 25*1.6/3.6
        self.acc_up = 5; #m/s^2
        self.acc_low = -12; #m/s^2

        self.VehicleArray=np.zeros((3,3))        
        self.move_count = 0
        self.move_count_pure_speeding = 0
        
#        self.nu = -0.2
#        self.nu1 = -0.2 
#        self.nu2 = -0.1  
        self.nu = -1
        self.nu1 = -2 
        self.nu2 = -1 
        
        self.nu = -5
        self.nu1 = -10 
        self.nu2 = -5  

        self.nu = -5
        self.nu1 = -20 
        self.nu2 = -10              

##  vehicle dynamics parameters
        self.m = m
        self.h = h
        self.lf = lf
        self.lr = lr
        self.Iz = Iz
        self.Iwr = 1.2
        self.r_F = 0.311
        self.r_R = 0.311
        self.L_axis = 2.4
        self.ls = 0.01

        self.Cf = 42000
        self.Cr = 81000    
        self.nt = 0.225
        
        self.B = 20
        self.C = 1.6  # C<2
        self.D = 1.3 #0.715
        self.E = 0.0  #-0.064
        self.Sh = 0.0 #-0.011
        self.Sv = 0.0 #0.01        

        self.Vx = self.vel+0.0
        self.Vy = 0.0
        self.r = 0.0
        self.psi = np.pi/2   
        self.wR = self.vel/self.r_R-1e-3

#####  define vehicle dynamics
    def vehicle_dynamics(self,HV,delta,Tr_aux,dt):
        Vf=[self.Vx,  self.Vy, 0]+ np.cross([0, 0, self.r], [self.lf, 0, self.r_F-self.h]); 
        Vr=[self.Vx,  self.Vy, 0]+ np.cross([0, 0, self.r], [-self.lr, 0, self.r_R-self.h]);
# wheel velocity in wheel-fixed frame
        Tran=np.array([[mh.cos(delta), mh.sin(delta), 0], [-mh.sin(delta), mh.cos(delta), 0], [0, 0, 1]]) 
        Vf=Tran.dot(Vf)  
        
        VFx=Vf[0]; 
        VFy=Vf[1]; 
        VRx=Vr[0]; 
        VRy=Vr[1];  
 
# bound the the control torque         
        beta_R=atan(VRy/VRx)
        s_R = np.tan(np.pi/2/self.C)/self.B
        if abs(mh.tan(beta_R))>s_R:
            beta_R=mh.atan(s_R/2.0)
        mu_R = self.D
      
        fRx_up = self.m*g*self.lf/( self.L_axis*s_R*(1/mh.cos(beta_R))**2/mu_R/( ( (1/mh.cos(beta_R))**2*s_R**2-(np.tan(beta_R))**2 )**0.5+(np.tan(beta_R))**2 )-self.h )
        fRx_low = -self.m*g*self.lf/( self.L_axis*s_R*(1/mh.cos(beta_R))**2/mu_R/( ( (1/mh.cos(beta_R))**2*s_R**2-(np.tan(beta_R))**2 )**0.5-(np.tan(beta_R))**2 )+self.h )
        Tr_up = fRx_up*self.r_R*(self.m+self.Iwr/self.r_R**2)/self.m
        Tr_low = fRx_low*self.r_R*(self.m+self.Iwr/self.r_R**2)/self.m
   
        Tr = bound( Tr_aux,0.95*Tr_low,0.95*Tr_up )
    
        if abs(self.Vx)<0.05: #abs(self.wR*self.r_R)<0.05:  # case of low velocity

            dot_Vy=0;  
            dot_r=0;    
            dot_XI=0;       
            dot_wR=Tr/self.Iwr;            
            dot_Vx=0.8*dot_wR*self.r_R;              
        else:                  
            sFx=0
            sFy=VFy/VFx
            sF= ( sFx**2+sFy**2 )**0.5
     
            sRx=( VRx-self.wR*self.r_R )/abs( self.wR*self.r_R )
            sRy=VRy/abs( self.wR*self.r_R ) 
            sR= ( sRx**2+sRy**2 )**0.5;
     
            mu_sF=self.D*mh.sin( self.C*atan( self.B*( (1-self.E)*(sF+self.Sh)+self.E/self.B*atan(self.B*(sF+self.Sh)) ) ) ) +self.Sv;
            mu_sR=self.D*mh.sin( self.C*atan( self.B*( (1-self.E)*(sR+self.Sh)+self.E/self.B*atan(self.B*(sR+self.Sh)) ) ) ) +self.Sv;
     
            if sF==0:  # no slip at front wheel
                fFx=0;
                fFy=0;                  
                muRx=-sRx*mu_sR/sR;
                muRy=-sRy*mu_sR/sR;                 
    #       calculate rear tire forces            
                fFz=( self.lr-self.h*muRx )*self.m*g/( self.L_axis-self.h*muRx );
                fRz=self.m*g-fFz;     
                fRx=muRx*fRz;
                fRy=muRy*fRz;                 
            else:             
                muFx=-sFx*mu_sF/sF;
                muFy=-sFy*mu_sF/sF;  
                muRx=-sRx*mu_sR/sR;
                muRy=-sRy*mu_sR/sR; 
     
    #       calculate tire forces
                fFz=( self.lr-self.h*muRx )*self.m*g/( self.L_axis+self.h*( muFx*mh.cos( delta )-muFy*mh.sin( delta )- muRx) );
                fRz=self.m*g-fFz;
     
                fFx=muFx*fFz;
                fFy=muFy*fFz;
                fRx=muRx*fRz;
                fRy=muRy*fRz;            

            dot_Vx=( fFx*mh.cos( delta )-fFy*mh.sin( delta )+fRx )/self.m+self.Vy*self.r;
            dot_Vy=( fFx*mh.sin( delta )+fFy*mh.cos( delta )+fRy )/self.m-self.Vx*self.r;  
            dot_r=( ( fFy*mh.cos( delta )+fFx*mh.sin( delta ) )*self.lf-fRy*self.lr )/self.Iz;
     
            T_rotation=np.array([ [mh.cos(self.psi), mh.sin(self.psi)], [-mh.sin(self.psi), mh.cos(self.psi)] ])
            dot_pos=T_rotation.transpose().dot([ self.Vx, self.Vy ])
     
            dot_XI=dot_pos[0];
#            dot_YI=dot_pos[1];   # do not update longitudinal pos for HV
            
            dot_wR=(Tr-fRx*self.r_R)/self.Iwr;
            
        self.Vx += dot_Vx*dt
        self.Vy += dot_Vy*dt
        self.psi += self.r*dt
        self.wR += dot_wR*dt
        self.wR = max(1e-1,self.wR)
        
        self.r += dot_r*dt
        self.pos[0] += dot_XI*dt*Resize
        self.pos[1] -= (self.Vx-HV.Vx)*dt*Resize
        
        self.Vx=bound(self.Vx,self.V_min,self.V_max)
        self.vel = self.Vx+0.0
        
##### define controller
    def ORT(self): 
##        Vx_aux = 15.0
#        Vx_aux = max(self.Vx, 5)
#        A=np.matrix([[-(self.Cf+self.Cr)/Vx_aux/self.m,-1+(self.lr*self.Cr-self.lf*self.Cf)/self.m/Vx_aux**2,0,0],[(self.lr*self.Cr-self.lf*self.Cf)/self.Iz,-(self.lr**2*self.Cr+self.lf**2*self.Cf)/self.Iz/Vx_aux,0,0],[-Vx_aux,-self.ls,0,Vx_aux],[0,-1,0,0]])
#        B=np.matrix([self.Cf/self.m/Vx_aux,self.lf*self.Cf/self.Iz,0,0]).T
#        E=np.matrix([0,0,Vx_aux*self.ls,Vx_aux]).T
#        C=np.matrix([0,0,1,0])
#        
#        rows = A.shape[0]
#        P = cp.Variable((rows, rows), PSD=True)
#        S = cp.Variable((1,rows))
#        PI = cp.Variable((rows,1))        
#
#        Gamma = cp.Variable()
#        epsi = cp.Variable()
#        I = np.identity(rows)
#
#        cons1 = epsi>=0
#        cons2 = -(A*P+B*S+P.T*A.T+S.T*B.T)>>1e0
#        cons3 = cp.bmat([[-I, A*PI+B*Gamma+E],[PI.T*A.T+Gamma.T*B.T+E.T, -epsi*np.eye(1)]])<<1e-10
#        cons4 = cp.bmat([[-epsi*np.eye(1), C*PI],[PI.T*C.T, -np.identity(1)]])<<1e-10
#        cons5 = P>>1e1
##        cons6 = epsi-1e-5<=0  
#        
#        optprob = cp.Problem(cp.Minimize(epsi), constraints=[cons1, cons2, cons3, cons4, cons5])
#        result = optprob.solve()
#        
#        aux1 = S.value.dot(np.linalg.pinv(P.value))
#        aux2 = Gamma.value-aux1.dot(PI.value)
#        self.G = aux1[0]
#        self.H = aux2[0][0]
        
#        self.G = np.array([-0.171908272474611,-0.0194559249021821,0.0686712004081447,0.972969648170903])
#        self.H = 4.58197066088387 
 
        self.G = np.array([-3.464761346737013525e+00,-7.421571597432523593e-01,1.132624286600968144e+00,4.632126864955187884e+00])
        self.H = 14.436298476877123
        
#        self.G = np.array([-0.171908282309310,-0.0194559256488496,0.0686712014555834,0.972969665762630])
#        self.H = 4.46361884075154
        
##### implement action
    def maintain_fcn(self,car_front,HV,steps=np.int(3*maintain_Time/dt)):
        if self.in_action:  
            self.move_count+=1
            self.grid_len=self.car_length/Resize+self.vel*self.Dt
            
#            yaw_err = self.psi-np.pi/2
            delta_y=self.pos[0]/Resize+self.ls*mh.cos(self.psi)-(Center_x+(self.lane-2)*width)/Resize;    
            delta_psi = np.pi/2-self.psi;     
#            self.ORT() 
            delta = self.G.dot(np.array([ atan(self.Vy/self.Vx), self.r, delta_y, delta_psi ])) # check math   
            
            if car_front is not None:
                dist=-(car_front.pos[1]-self.pos[1])/Resize
            
            if car_front==None:
                v_err=self.vel-self.vel_tg
#                self.acc = bound( self.nu*v_err,self.acc_low,self.acc_up )
                self.acc = self.nu*v_err
            elif self.vel>=car_front.vel and dist<2*self.grid_len:   
                self.vel_tg=car_front.vel+0.0                
                s_err=-dist+self.grid_len
                v_err=self.vel-self.vel_tg
#                self.acc = bound( self.nu1*v_err+self.nu2*s_err,self.acc_low,self.acc_up )
                self.acc = self.nu1*v_err+self.nu2*s_err
            else: #if self.vel<car_front.vel:
                v_err=self.vel-self.vel_tg
#                self.acc = bound( self.nu*v_err,self.acc_low,self.acc_up )
                self.acc = self.nu*v_err
            self.acc = bound( self.acc,self.acc_low,self.acc_up )    
            Tr_aux = self.acc*self.r_R*(self.m+self.Iwr/self.r_R**2)  
            self.vehicle_dynamics(HV,delta,Tr_aux,dt)    
#            self.vel+=self.acc*dt
#            self.vel=bound(self.vel,self.V_min,self.V_max)
#            self.pos[1] -= (self.vel-HV.vel)*dt*Resize  # not update y_pos
            if ((car_front!=None and (self.vel-car_front.vel)**2<0.25) or self.move_count>=steps) and delta_psi**2<25e-4 and self.r**2<1e-2:
                self.move_count=0
                self.in_action=False   
    
    def acc_fcn(self,car_front,HV,acc,steps=np.int(3*maintain_Time/dt)):
        if self.in_action:
#            yaw_err = self.psi-np.pi/2
            delta_y=self.pos[0]/Resize+self.ls*mh.cos(self.psi)-(Center_x+(self.lane-2)*width)/Resize;    
            delta_psi = np.pi/2-self.psi;     
#            self.ORT() 
            delta = self.G.dot(np.array([ atan(self.Vy/self.Vx), self.r, delta_y, delta_psi ])) # check math             
            
            if acc or car_front==None: 
                self.move_count_pure_speeding+=1 
                self.grid_len=self.car_length/Resize+self.vel*self.Dt
                self.vel_tg=self.V_max
                v_err=self.vel-self.vel_tg
                self.acc = self.nu*v_err
#                self.acc=self.acc_up 
            else:  # provide the car making acc==0
                self.move_count+=1
                self.grid_len=self.car_length/Resize+self.vel*self.Dt
                dist=-(car_front.pos[1]-self.pos[1])/Resize
                self.vel_tg=car_front.vel                 
                s_err=-dist+self.grid_len
                v_err=self.vel-self.vel_tg
                self.acc = self.nu1*v_err+self.nu2*s_err                
#            self.vel+=self.acc*dt
            self.acc = bound( self.acc,self.acc_low,self.acc_up )    
            Tr_aux = self.acc*self.r_R*(self.m+self.Iwr/self.r_R**2)  
            self.vehicle_dynamics(HV,delta,Tr_aux,dt)
#            self.vel=bound(self.vel,self.V_min,self.V_max)
#            self.pos[1] -= (self.vel-HV.vel)*dt*Resize # not update y_pos
            if (v_err**2<0.25 or self.move_count_pure_speeding>=steps or self.move_count>=steps) and delta_psi**2<25e-4 and self.r**2<1e-2:
                self.move_count_pure_speeding=0
                self.move_count=0
                self.in_action=False
                
    def brk_fcn(self,car_front,HV,steps=np.int(maintain_Time/dt)):
        if self.in_action:
            self.move_count+=1
#            yaw_err = self.psi-np.pi/2
            delta_y=self.pos[0]/Resize+self.ls*mh.cos(self.psi)-(Center_x+(self.lane-2)*width)/Resize;    
            delta_psi = np.pi/2-self.psi;     
#            self.ORT() 
            delta = self.G.dot(np.array([ atan(self.Vy/self.Vx), self.r, delta_y, delta_psi ])) # check math
             
            if car_front is not None and self.vel>car_front.vel:
                self.grid_len=self.car_length/Resize+self.vel*self.Dt
                dist=-(car_front.pos[1]-self.pos[1])/Resize
                vel_aux=max(self.V_min,min(car_front.vel,self.vel_tg*0.8))                
                s_err=-dist+self.grid_len
                v_err=self.vel-vel_aux
                self.acc = self.nu1*v_err+self.nu2*s_err
#                self.acc = bound( self.nu1*v_err+self.nu2*s_err,self.acc_low,self.acc_up )  
            else:
                self.grid_len=self.car_length/Resize+self.vel*self.Dt
                v_err=self.vel-self.vel_tg*0.8
                self.acc = self.nu*v_err  
#                self.acc = bound( self.nu*v_err,self.acc_low,self.acc_up )                
#            self.vel+=self.acc*dt
            self.acc = bound( self.acc,self.acc_low,self.acc_up )    
            Tr_aux = self.acc*self.r_R*(self.m+self.Iwr/self.r_R**2)  
            self.vehicle_dynamics(HV,delta,Tr_aux,dt)
#            self.vel=bound(self.vel,self.V_min,self.V_max)            
#            self.pos[1] -= (self.vel-HV.vel)*dt*Resize # not update y_pos
            if (v_err**2<0.25 or self.move_count>=steps) and delta_psi**2<25e-4 and self.r**2<1e-2:
                self.move_count=0
                self.in_action=False    

    def left_fcn(self,car_front,HV):
        if self.in_action:
            lat_err = self.pos[0]/Resize-self.Curve_AB_Ex_Rotate[-1,0]
            yaw_err = self.psi-np.pi/2
            pos_pred = np.zeros(2)
            pos_pred[0]=self.pos[0]/Resize+self.ls*mh.cos(self.psi)
            pos_pred[1]=(2*self.Curve_AB_Ex_Rotate[0,1]-self.pos[1]/Resize)+self.ls*mh.sin(self.psi)
            travel_dis = pos_pred[1]-self.Curve_AB_Ex_Rotate[0,1]
            if travel_dis<self.Length:
                x_ref = np.interp(pos_pred[1], self.Curve_AB_Ex_Rotate[:,1], self.Curve_AB_Ex_Rotate[:,0])
                rho_ref = np.interp(pos_pred[1], self.Curve_AB_Ex_Rotate[:,1], self.ka_Ex)
                psi_ref = np.interp(pos_pred[1], self.Curve_AB_Ex_Rotate[:,1], self.psi_t)
            else:
                x_ref = self.Curve_AB_Ex_Rotate[-1,0]
                rho_ref = 0.0
                psi_ref = np.pi/2
                
            delta_y=pos_pred[0]-x_ref;    
            delta_psi = psi_ref-self.psi;
#            self.ORT()                
            delta = self.G.dot(np.array([ atan(self.Vy/self.Vx), self.r, delta_y, delta_psi ]))+self.H*rho_ref # check math            
            
            Tr_aux = 0.0
            self.grid_len=self.car_length/Resize+self.vel*self.Dt
            if car_front is not None and self.vel>car_front.vel:
                dist=-(car_front.pos[1]-self.pos[1])/Resize
                self.vel_tg=car_front.vel+0                
                s_err=-dist+self.grid_len
                v_err=self.vel-self.vel_tg
                self.acc = self.nu1*v_err+self.nu2*s_err
                self.acc = bound( self.acc,self.acc_low,self.acc_up )
                Tr_aux = self.acc*self.r_R*(self.m+self.Iwr/self.r_R**2) 
                
            self.vehicle_dynamics(HV,delta,Tr_aux,dt)           

            self.Curve_AB_Ex_Rotate[:,1]=self.Curve_AB_Ex_Rotate[:,1]+HV.Vx*dt
            if lat_err**2<=1e-2 and yaw_err**2<25e-4 and self.r**2<1e-2:#travel_dis>=extra_dis+self.Length: #lat_err**2<=1e-2:
#                self.move_count=0
                self.in_action=False
                self.lane = self.lane_tg+0                


    def right_fcn(self,car_front,HV):
        if self.in_action:
            lat_err = self.pos[0]/Resize-self.Curve_AB_Ex_Rotate[-1,0]
            yaw_err = self.psi-np.pi/2
            pos_pred = np.zeros(2)
            pos_pred[0]=self.pos[0]/Resize+self.ls*mh.cos(self.psi)
            pos_pred[1]=(2*self.Curve_AB_Ex_Rotate[0,1]-self.pos[1]/Resize)+self.ls*mh.sin(self.psi)
            travel_dis = pos_pred[1]-self.Curve_AB_Ex_Rotate[0,1]
            if travel_dis<self.Length:
                x_ref = np.interp(pos_pred[1], self.Curve_AB_Ex_Rotate[:,1], self.Curve_AB_Ex_Rotate[:,0])
                rho_ref = np.interp(pos_pred[1], self.Curve_AB_Ex_Rotate[:,1], self.ka_Ex)
                psi_ref = np.interp(pos_pred[1], self.Curve_AB_Ex_Rotate[:,1], self.psi_t)
            else:
                x_ref = self.Curve_AB_Ex_Rotate[-1,0]
                rho_ref = 0.0
                psi_ref = np.pi/2
                
            delta_y=pos_pred[0]-x_ref;    
            delta_psi = psi_ref-self.psi;  
#            self.ORT()               
            delta = self.G.dot(np.array([ atan(self.Vy/self.Vx), self.r, delta_y, delta_psi ]))+self.H*rho_ref # check math            
            
            Tr_aux = 0.0
            self.grid_len=self.car_length/Resize+self.vel*self.Dt
            if car_front is not None and self.vel>car_front.vel:
                dist=-(car_front.pos[1]-self.pos[1])/Resize
                self.vel_tg=car_front.vel+0                
                s_err=-dist+self.grid_len
                v_err=self.vel-self.vel_tg
                self.acc = self.nu1*v_err+self.nu2*s_err
                self.acc = bound( self.acc,self.acc_low,self.acc_up )
                Tr_aux = self.acc*self.r_R*(self.m+self.Iwr/self.r_R**2) 
                
            self.vehicle_dynamics(HV,delta,Tr_aux,dt) 
#            self.vel=bound(self.vel,self.V_min,self.V_max) 
            
#            self.Curve_AB_Ex_Rotate[:,1]+=HV.Vx*dt
            self.Curve_AB_Ex_Rotate[:,1]=self.Curve_AB_Ex_Rotate[:,1]+HV.Vx*dt
            if lat_err**2<=1e-2 and yaw_err**2<25e-4 and self.r**2<1e-2:#travel_dis>=extra_dis+self.Length: #lat_err**2<=1e-2:
#                self.move_count=0
                self.in_action=False
                self.lane = self.lane_tg+0              
        
    def TakeAction(self,state,car_front,HV):  # use new state when take action everytime, update lane and pos

        self.Pr_stay = Pr_actions[-1]
        self.Pr_acc = Pr_actions[-2]
        self.Pr_left = Pr_actions[-3]
        self.Pr_right = Pr_actions[-4]
        self.Pr_brake = Pr_actions[-5]
        
        acc, brk, left, right = Available_actions_EV(state)
        Pr_sum = sum([brk, right, left, acc, 1]*Pr_actions)  # total = 1 if everything available
       
        self.Pr_stay = self.Pr_stay/Pr_sum
        self.Pr_acc = self.Pr_acc/Pr_sum*acc   # ->0 if acc=0
        self.Pr_left = self.Pr_left/Pr_sum*left
        self.Pr_right = self.Pr_right/Pr_sum*right
        self.Pr_brake = self.Pr_brake/Pr_sum*brk

        if not self.in_action:          
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

            if self.Action=='left' or self.Action=='right':
                if self.Action=='left':
                    direction = 1.0
                else: 
                    direction = -1.0
                k_max = min(1.0/150, 0.25*self.D*g/self.Vx**2)
                self.tau_ex, self.Curve_AB_Ex_Rotate, self.ka_Ex, self.psi_t, self.Length = path_planning(self.pos,k_max,width/Resize,direction)
 
            self.ORT()
           
            self.vel_tg = self.vel+0.0
            self.in_action = True
            
            
        if self.Action=='stay':
            self.maintain_fcn(car_front,HV)
        elif self.Action=='acc':  
            self.acc_fcn(car_front,HV,acc)
        elif self.Action=='brk':    
            self.brk_fcn(car_front,HV)                
        elif self.Action=='left':
            self.lane_tg = self.lane-1
            self.left_fcn(car_front,HV) 
        else: 
            self.lane_tg = self.lane+1
            self.right_fcn(car_front,HV) 
            
#===============================================
class HV_obj(object):
    def __init__(self, HV_lane, Pos, Vel, Dt, Policy, color_index, Curvature_Class, m=850, h=0.5, lf=1.5, lr=0.9, Iz=1401):   # for demonstraion purpose, check available actions so that no collision happens 
                                                       # modify Available_actions_fn with all actions for other purpose
        self.lane = HV_lane
        self.lane_tg = HV_lane
        self.pos = Pos #[bit]
        self.vel = Vel*1.6/3.6 #[m/s]
        self.vel_tg = Vel*1.6/3.6 #[m/s]
        self.acc = 0.0
        self.Dt = Dt
        self.Action = 'stay'
        self.in_action = False
        self.Curvature_Class = Curvature_Class
        self.color_index = color_index
        self.Color = ColorSet_HV[self.color_index]
        self.car_width=car_width
        self.car_length=car_length
        self.grid_len=self.car_length/Resize+self.vel*self.Dt        
        self.Policy = Policy
        self.V_max = 45*1.6/3.6   #Resize=130/4.5
        self.V_min = 18*1.6/3.6
        self.acc_up = 5; #m/s^2
        self.acc_low = -12; #m/s^2

        self.state=9*[0]        
        self.move_count = 0
        self.move_count_pure_speeding = 0
        
#        self.nu = -0.1
#        self.nu1 = -0.2 
#        self.nu2 = -0.1
        self.nu = -1
        self.nu1 = -2 
        self.nu2 = -1 
        
        self.nu = -5
        self.nu1 = -10
        self.nu2 = -5    
        
        self.nu = -5
        self.nu1 = -20 
        self.nu2 = -10          
##  vehicle dynamics parameters
        self.m = m
        self.h = h
        self.lf = lf
        self.lr = lr
        self.Iz = Iz
        self.Iwr = 1.2
        self.r_F = 0.311
        self.r_R = 0.311
        self.L_axis = 2.4
        self.ls = 0.01

        self.Cf = 42000
        self.Cr = 81000    
        self.nt = 0.225
        
        self.B = 20
        self.C = 1.6  # C<2
        self.D = 1.2 #0.715
        self.E = 0.0  #-0.064
        self.Sh = 0.0 #-0.011
        self.Sv = 0.0 #0.01

        self.Vx = self.vel+0.0
        self.Vy = 0.0
        self.r = 0.0
        self.psi = np.pi/2   
        self.wR = self.vel/self.r_R-1e-3
        
    def vehicle_dynamics(self,delta,Tr_aux,dt):
        Vf=[self.Vx,  self.Vy, 0]+ np.cross([0, 0, self.r], [self.lf, 0, self.r_F-self.h]); 
        Vr=[self.Vx,  self.Vy, 0]+ np.cross([0, 0, self.r], [-self.lr, 0, self.r_R-self.h]);
# wheel velocity in wheel-fixed frame
        Tran=np.array([[mh.cos(delta), mh.sin(delta), 0], [-mh.sin(delta), mh.cos(delta), 0], [0, 0, 1]]) 
        Vf=Tran.dot(Vf)  
        
        VFx=Vf[0]; 
        VFy=Vf[1]; 
        VRx=Vr[0]; 
        VRy=Vr[1];   
        
# bound the the control torque         
        beta_R=atan(VRy/VRx)
        s_R = np.tan(np.pi/2/self.C)/self.B
        if abs(mh.tan(beta_R))>s_R:
            beta_R=mh.atan(s_R/2.0)
        mu_R = self.D
      
        fRx_up = self.m*g*self.lf/( self.L_axis*s_R*(1/mh.cos(beta_R))**2/mu_R/( ( (1/mh.cos(beta_R))**2*s_R**2-(np.tan(beta_R))**2 )**0.5+(np.tan(beta_R))**2 )-self.h )
        fRx_low = -self.m*g*self.lf/( self.L_axis*s_R*(1/mh.cos(beta_R))**2/mu_R/( ( (1/mh.cos(beta_R))**2*s_R**2-(np.tan(beta_R))**2 )**0.5-(np.tan(beta_R))**2 )+self.h )
        Tr_up = fRx_up*self.r_R*(self.m+self.Iwr/self.r_R**2)/self.m
        Tr_low = fRx_low*self.r_R*(self.m+self.Iwr/self.r_R**2)/self.m
        Tr = bound( Tr_aux,0.95*Tr_low,0.95*Tr_up ) 
        if self.wR<0:
            stop=0
            
        if abs(self.Vx)<0.05: #abs(self.wR*self.r_R)<0.05:  # case of low velocity

            dot_Vy=0;  
            dot_r=0;    
            dot_XI=0;
#            dot_YI=0;        
            dot_wR=Tr/self.Iwr;            
            dot_Vx=0.8*dot_wR*self.r_R;              
        else:                  
            sFx=0
            sFy=VFy/VFx
            sF= ( sFx**2+sFy**2 )**0.5   
            sRx=( VRx-self.wR*self.r_R )/abs( self.wR*self.r_R )
            sRy=VRy/abs( self.wR*self.r_R ) 
            sR= ( sRx**2+sRy**2 )**0.5;
     
            mu_sF=self.D*mh.sin( self.C*atan( self.B*sF ) ) 
            mu_sR=self.D*mh.sin( self.C*atan( self.B*sR ) ) 
     
            if sF==0:  # no slip at front wheel
                fFx=0;
                fFy=0;                  
                muRx=-sRx*mu_sR/sR;
                muRy=-sRy*mu_sR/sR;                 
    #       calculate rear tire forces            
                fFz=( self.lr-self.h*muRx )*self.m*g/( self.L_axis-self.h*muRx );
                fRz=self.m*g-fFz;     
                fRx=muRx*fRz;
                fRy=muRy*fRz;                 
            else:             
                muFx=-sFx*mu_sF/sF;
                muFy=-sFy*mu_sF/sF;  
                muRx=-sRx*mu_sR/sR;
                muRy=-sRy*mu_sR/sR; 
     
    #       calculate tire forces
                fFz=( self.lr-self.h*muRx )*self.m*g/( self.L_axis+self.h*( muFx*mh.cos( delta )-muFy*mh.sin( delta )- muRx) );
                fRz=self.m*g-fFz;
     
                fFx=muFx*fFz;
                fFy=muFy*fFz;
                fRx=muRx*fRz;
                fRy=muRy*fRz;            

            dot_Vx=( fFx*mh.cos( delta )-fFy*mh.sin( delta )+fRx )/self.m+self.Vy*self.r;
            dot_Vy=( fFx*mh.sin( delta )+fFy*mh.cos( delta )+fRy )/self.m-self.Vx*self.r;  
            dot_r=( ( fFy*mh.cos( delta )+fFx*mh.sin( delta ) )*self.lf-fRy*self.lr )/self.Iz;
     
            T_rotation=np.array([ [mh.cos(self.psi), mh.sin(self.psi)], [-mh.sin(self.psi), mh.cos(self.psi)] ])
            dot_pos=T_rotation.transpose().dot([ self.Vx, self.Vy ])
     
            dot_XI=dot_pos[0];
#            dot_YI=dot_pos[1];   # do not update longitudinal pos for HV
            
            dot_wR=(Tr-fRx*self.r_R)/self.Iwr;
            
        self.Vx += dot_Vx*dt
        self.Vy += dot_Vy*dt
        self.psi += self.r*dt
        if self.wR + dot_wR*dt<0:
            stop=0       
        self.wR += dot_wR*dt
        self.wR = max(1e-1,self.wR)
        
        self.r += dot_r*dt
        self.pos[0] += dot_XI*dt*Resize 
        
        self.Vx=bound(self.Vx,self.V_min,self.V_max)
        self.vel = self.Vx+0.0
        
##### define controller
    def ORT(self):       
##        Vx_aux = 10.0
#        Vx_aux = max(self.Vx, 5)
#        A=np.matrix([[-(self.Cf+self.Cr)/Vx_aux/self.m,-1+(self.lr*self.Cr-self.lf*self.Cf)/self.m/Vx_aux**2,0,0],[(self.lr*self.Cr-self.lf*self.Cf)/self.Iz,-(self.lr**2*self.Cr+self.lf**2*self.Cf)/self.Iz/Vx_aux,0,0],[-Vx_aux,-self.ls,0,Vx_aux],[0,-1,0,0]])
#        B=np.matrix([self.Cf/self.m/Vx_aux,self.lf*self.Cf/self.Iz,0,0]).T
#        E=np.matrix([0,0,Vx_aux*self.ls,Vx_aux]).T        
#        C=np.matrix([0,0,1,0])
#        
#        rows = A.shape[0]
#        P = cp.Variable((rows, rows), PSD=True)
#        S = cp.Variable((1,rows))
#        PI = cp.Variable((rows,1))        
#
#        Gamma = cp.Variable()
#        epsi = cp.Variable()
#        I = np.identity(rows)
#
#        cons1 = epsi>=0
#        cons2 = -(A*P+B*S+P.T*A.T+S.T*B.T)>>1e0
#        cons3 = cp.bmat([[-I, A*PI+B*Gamma+E],[PI.T*A.T+Gamma.T*B.T+E.T, -epsi*np.eye(1)]])<<1e-10
#        cons4 = cp.bmat([[-epsi*np.eye(1), C*PI],[PI.T*C.T, -np.identity(1)]])<<1e-10
#        cons5 = P>>1e1
##        cons6 = epsi-1e-5<=0  
#        
#        optprob = cp.Problem(cp.Minimize(epsi), constraints=[cons1, cons2, cons3, cons4, cons5])
#        result = optprob.solve()
#
#        aux1 = S.value.dot(np.linalg.pinv(P.value))
#        aux2 = Gamma.value-aux1.dot(PI.value)
#        self.G = aux1[0]
#        self.H = aux2[0][0] 
        
#        self.G = np.array([-0.0260928258891248,0.0152326746505565,0.0468479504998531,0.672122306996471])
#        self.H = 2.19301870160333          
        self.G = np.array([-3.464761346737013525e+00,-7.421571597432523593e-01,1.132624286600968144e+00,4.632126864955187884e+00])
        self.H = 14.436298476877123
        
#        self.G = np.array([-0.171908282309310,-0.0194559256488496,0.0686712014555834,0.972969665762630])
#        self.H = 4.46361884075154
        
##### implement action
        
    def maintain_fcn(self,car_front,steps=np.int(maintain_Time/dt)):
        if self.in_action:  
            self.move_count+=1
            self.grid_len=self.car_length/Resize+self.vel*self.Dt

#            yaw_err = self.psi-np.pi/2
            delta_y=self.pos[0]/Resize+self.ls*mh.cos(self.psi)-(Center_x+(self.lane-2)*width)/Resize;    
            delta_psi = np.pi/2-self.psi;     
#            self.ORT() 
            delta = self.G.dot(np.array([ atan(self.Vy/self.Vx), self.r, delta_y, delta_psi ])) # check math  
            
            if car_front is not None:
                dist=-(car_front.pos[1]-self.pos[1])/Resize
            
            if car_front==None:
                v_err=self.vel-self.vel_tg
                self.acc = self.nu*v_err
            elif self.vel>=car_front.vel and dist<2*self.grid_len:  
                self.vel_tg=car_front.vel                 
                s_err=-dist+self.grid_len
                v_err=self.vel-self.vel_tg
                self.acc = self.nu1*v_err+self.nu2*s_err
            else: #if self.vel<car_front.vel:
                v_err=self.vel-self.vel_tg
                self.acc = self.nu*v_err
                
            self.acc = bound( self.acc,self.acc_low,self.acc_up )   
            Tr_aux = self.acc*self.r_R*(self.m+self.Iwr/self.r_R**2)     
            self.vehicle_dynamics(delta,Tr_aux,dt)
#            self.vel=bound(self.vel,self.V_min,self.V_max)
#            if ((car_front!=None and (self.vel-car_front.vel)**2<0.25) or self.move_count>=steps) and yaw_err**2<25e-4:
            if self.move_count>=steps and delta_psi**2<25e-4 and self.r**2<1e-2:    
                self.move_count=0
                self.in_action=False   

    
    def acc_fcn(self,car_front,acc,steps=np.int(maintain_Time/dt)):
        if self.in_action:
            
#            yaw_err = self.psi-np.pi/2
            delta_y=self.pos[0]/Resize+self.ls*mh.cos(self.psi)-(Center_x+(self.lane-2)*width)/Resize;    
            delta_psi = np.pi/2-self.psi;     
#            self.ORT() 
            delta = self.G.dot(np.array([ atan(self.Vy/self.Vx), self.r, delta_y, delta_psi ])) # check math            

            if acc or car_front==None: 
                self.move_count_pure_speeding+=1
                self.grid_len=self.car_length/Resize+self.vel*self.Dt
                self.vel_tg=self.V_max
                v_err=self.vel-self.vel_tg
                self.acc = self.nu*v_err
            else:  # provide the car making acc==0
                self.move_count+=1
                self.grid_len=self.car_length/Resize+self.vel*self.Dt
                dist=-(car_front.pos[1]-self.pos[1])/Resize
                self.vel_tg=car_front.vel+0                
                s_err=-dist+self.grid_len
                v_err=self.vel-self.vel_tg
                self.acc = self.nu1*v_err+self.nu2*s_err 
                                
            self.acc = bound( self.acc,self.acc_low,self.acc_up )    
            Tr_aux = self.acc*self.r_R*(self.m+self.Iwr/self.r_R**2)  
            self.vehicle_dynamics(delta,Tr_aux,dt)
#            self.vel=bound(self.vel,self.V_min,self.V_max)
            if (v_err**2<0.25 or self.move_count_pure_speeding>=steps or self.move_count>=steps) and delta_psi**2<25e-4 and self.r**2<1e-2:
                self.move_count_pure_speeding=0
                self.move_count=0
                self.in_action=False  
                               
    def brk_fcn(self,car_front,steps=np.int(maintain_Time/dt)):
        if self.in_action:
            self.move_count+=1
            
#            yaw_err = self.psi-np.pi/2
            delta_y=self.pos[0]/Resize+self.ls*mh.cos(self.psi)-(Center_x+(self.lane-2)*width)/Resize;    
            delta_psi = np.pi/2-self.psi;     
#            self.ORT() 
            delta = self.G.dot(np.array([ atan(self.Vy/self.Vx), self.r, delta_y, delta_psi ])) # check math 
            
            if car_front is not None and self.vel>car_front.vel:
                self.grid_len=self.car_length/Resize+self.vel*self.Dt
                dist=-(car_front.pos[1]-self.pos[1])/Resize
                vel_aux=max(self.V_min,min(car_front.vel,self.vel_tg*0.8))                
                s_err=-dist+self.grid_len
                v_err=self.vel-vel_aux
                self.acc = self.nu1*v_err+self.nu2*s_err  
            else:
                self.grid_len=self.car_length/Resize+self.vel*self.Dt
                v_err=self.vel-self.vel_tg*0.8
                self.acc = self.nu*v_err
            
            self.acc = bound( self.acc,self.acc_low,self.acc_up )
            Tr_aux = self.acc*self.r_R*(self.m+self.Iwr/self.r_R**2)  
            self.vehicle_dynamics(delta,Tr_aux,dt)
#            self.vel=bound(self.vel,self.V_min,self.V_max)
            if (v_err**2<0.25 or self.move_count>=steps) and delta_psi**2<25e-4 and self.r**2<1e-2:
                self.move_count=0
                self.in_action=False    

                
    def left_fcn(self,car_front):
        if self.in_action:
#            self.move_count+=1
#            self.pos[0]-=width/steps
            lat_err = self.pos[0]/Resize-self.Curve_AB_Ex_Rotate[-1,0]
            yaw_err = self.psi-np.pi/2
            pos_pred = np.zeros(2)
            pos_pred[0]=self.pos[0]/Resize+self.ls*mh.cos(self.psi)
            pos_pred[1]=(2*self.Curve_AB_Ex_Rotate[0,1]-self.pos[1]/Resize)+self.ls*mh.sin(self.psi)
            travel_dis = pos_pred[1]-self.Curve_AB_Ex_Rotate[0,1]
            if travel_dis<self.Length:
                x_ref = np.interp(pos_pred[1], self.Curve_AB_Ex_Rotate[:,1], self.Curve_AB_Ex_Rotate[:,0])
                rho_ref = np.interp(pos_pred[1], self.Curve_AB_Ex_Rotate[:,1], self.ka_Ex)
                psi_ref = np.interp(pos_pred[1], self.Curve_AB_Ex_Rotate[:,1], self.psi_t)
            else:
                x_ref = self.Curve_AB_Ex_Rotate[-1,0]
                rho_ref = 0.0
                psi_ref = np.pi/2
                
            delta_y=pos_pred[0]-x_ref;    
            delta_psi = psi_ref-self.psi;     
#            self.ORT() 
            delta = self.G.dot(np.array([ atan(self.Vy/self.Vx), self.r, delta_y, delta_psi ]))+self.H*rho_ref # check math            
            
            Tr_aux = 0.0  
            self.grid_len=self.car_length/Resize+self.vel*self.Dt            
            if car_front is not None and self.vel>car_front.vel:
                dist=-(car_front.pos[1]-self.pos[1])/Resize
                self.vel_tg=car_front.vel+0                
                s_err=-dist+self.grid_len
                v_err=self.vel-self.vel_tg
                self.acc = self.nu1*v_err+self.nu2*s_err
                self.acc = bound( self.acc,self.acc_low,self.acc_up )
                Tr_aux = self.acc*self.r_R*(self.m+self.Iwr/self.r_R**2)
           
            self.vehicle_dynamics(delta,Tr_aux,dt) 
#            self.vel=bound(self.vel,self.V_min,self.V_max)            
#            self.Curve_AB_Ex_Rotate[:,1]+=self.Vx*dt
            self.Curve_AB_Ex_Rotate[:,1]=self.Curve_AB_Ex_Rotate[:,1]+HV.Vx*dt
            if lat_err**2<=1e-2 and yaw_err**2<25e-4 and self.r**2<1e-2:#travel_dis>=extra_dis+self.Length: #lat_err**2<=1e-2:
#                self.move_count=0
                self.in_action=False
                self.lane = self.lane_tg+0                  
             

    def right_fcn(self,car_front):
        if self.in_action:
            lat_err = self.pos[0]/Resize-self.Curve_AB_Ex_Rotate[-1,0]
            yaw_err = self.psi-np.pi/2
            pos_pred = np.zeros(2)
            pos_pred[0]=self.pos[0]/Resize+self.ls*mh.cos(self.psi)
            pos_pred[1]=(2*self.Curve_AB_Ex_Rotate[0,1]-self.pos[1]/Resize)+self.ls*mh.sin(self.psi)
            travel_dis = pos_pred[1]-self.Curve_AB_Ex_Rotate[0,1]
            if travel_dis<self.Length:
                x_ref = np.interp(pos_pred[1], self.Curve_AB_Ex_Rotate[:,1], self.Curve_AB_Ex_Rotate[:,0])
                rho_ref = np.interp(pos_pred[1], self.Curve_AB_Ex_Rotate[:,1], self.ka_Ex)
                psi_ref = np.interp(pos_pred[1], self.Curve_AB_Ex_Rotate[:,1], self.psi_t)
            else:
                x_ref = self.Curve_AB_Ex_Rotate[-1,0]
                rho_ref = 0.0
                psi_ref = np.pi/2
                
            delta_y=pos_pred[0]-x_ref;    
            delta_psi = psi_ref-self.psi;
#            self.ORT()                 
            delta = self.G.dot(np.array([ atan(self.Vy/self.Vx), self.r, delta_y, delta_psi ]))+self.H*rho_ref # check math            
            
            Tr_aux = 0.0
            self.grid_len=self.car_length/Resize+self.vel*self.Dt            
            if car_front is not None and self.vel>car_front.vel:
                dist=-(car_front.pos[1]-self.pos[1])/Resize
                self.vel_tg=car_front.vel+0                
                s_err=-dist+self.grid_len
                v_err=self.vel-self.vel_tg
                self.acc = self.nu1*v_err+self.nu2*s_err
                self.acc = bound( self.acc,self.acc_low,self.acc_up )
                Tr_aux = self.acc*self.r_R*(self.m+self.Iwr/self.r_R**2)
           
            self.vehicle_dynamics(delta,Tr_aux,dt) 
#            self.vel=bound(self.vel,self.V_min,self.V_max)            
#            self.Curve_AB_Ex_Rotate[:,1]+=self.Vx*dt
            self.Curve_AB_Ex_Rotate[:,1]=self.Curve_AB_Ex_Rotate[:,1]+HV.Vx*dt
            if  lat_err**2<=1e-2 and yaw_err**2<25e-4 and self.r**2<1e-2:#travel_dis>=extra_dis+self.Length: #lat_err**2<=1e-2:
                self.in_action=False
                self.lane = self.lane_tg+0                     
        
    def TakeAction_HV(self,state,car_front):
        if not self.in_action:
            if state[0] == 1:
                state_row_index = sum( np.array([2**4,2**3,0,2**2,0,2,1,0])*state[1:9] )
            elif state[0] == 2:
                state_row_index = 2**5+sum( np.array([2**7,2**6,2**5,2**4,2**3,2**2,2,1])*state[1:9] )
            elif state[0] == 3:
                state_row_index = 2**5+2**8+sum( np.array([0,2**4,2**3,0,2**2,0,2,1])*state[1:9] )
                
            best_action_index = self.Policy[int(round(state_row_index)),self.Curvature_Class].astype(int)       
            self.Action = ActionSet[best_action_index]
            self.vel_tg = self.vel+0.0
            self.in_action = True
            
            if self.Action=='left' or self.Action=='right':
                if self.Action=='left':
                    direction = 1.0
                else: 
                    direction = -1.0
                k_max = k_max = min(1.0/150, 0.25*self.D*g/self.Vx**2)
                self.tau_ex, self.Curve_AB_Ex_Rotate, self.ka_Ex, self.psi_t, self.Length = path_planning(self.pos,k_max,width/Resize,direction)
            
            self.ORT()
            
        acc,brk,lf,rt=Safe_actions_HV(state)
                       
        if self.Action=='stay':
            self.maintain_fcn(car_front)
        elif self.Action=='acc':  
            self.acc_fcn(car_front,acc)
        elif self.Action=='brk':    
            self.brk_fcn(car_front)                
        elif self.Action=='left':
            self.lane_tg = self.lane-1
            self.left_fcn(car_front) 
        else: 
            self.lane_tg = self.lane+1
            self.right_fcn(car_front)             
            
            
#===============================================

pygame.init()

gameDisplay = pygame.display.set_mode((display_width,display_height),pygame.RESIZABLE)
pygame.display.set_caption('Multi lane animation')

black = (0,0,0)
white = (255,255,255)

clock = pygame.time.Clock()
crashed = False

myfont = pygame.font.SysFont('Comic Sans MS', 50)

GreenCar_HV = pygame.image.load('ImageFolder/GreenCar_HV.png')
GreenCar = pygame.image.load('ImageFolder/GreenCar.png')
BlueCar = pygame.image.load('ImageFolder/BlueCar.png')
PinkCar = pygame.image.load('ImageFolder/PinkCar.png')
GreyCar = pygame.image.load('ImageFolder/GreyCar.png')
PurpleCar = pygame.image.load('ImageFolder/PurpleCar.png')
RedCar = pygame.image.load('ImageFolder/RedCar.png')
YellowCar = pygame.image.load('ImageFolder/YellowCar.png')

GreenTruck = pygame.image.load('ImageFolder/GreenTruck.png')
BlueTruck = pygame.image.load('ImageFolder/BlueTruck.png')
GreyTruck = pygame.image.load('ImageFolder/GreyTruck.png')
RedTruck = pygame.image.load('ImageFolder/RedTruck.png')
YellowTruck = pygame.image.load('ImageFolder/YellowTruck.png')


Straight = pygame.image.load('ImageFolder/straight.png')
Circuit = pygame.image.load('ImageFolder/circuit1.png')  # need png, rotation keeps trnasparency
Circuit = pygame.image.load('ImageFolder/circuit3.png')

LeftTurn = pygame.image.load('ImageFolder/LeftTurn3.png')
RightTurn = pygame.image.load('ImageFolder/RightTurn3.png')

def green_car_HV(x,y,car_width,car_length,car_angle=0):
    GreenCar_HV_rota = pygame.transform.rotate(GreenCar_HV,car_angle)
    gameDisplay.blit(GreenCar_HV_rota, (x-car_width/2*np.cos(car_angle/180*np.pi)-car_length/2*np.sin(np.abs(car_angle/180*np.pi)),y-car_length/2*np.cos(car_angle/180*np.pi))-car_width/2*np.sin(np.abs(car_angle/180*np.pi)))
    
def green_car(x,y,car_width,car_length,car_angle=0):
    GreenCar_rota = pygame.transform.rotate(GreenCar,car_angle)
    gameDisplay.blit(GreenCar_rota, (x-car_width/2*np.cos(car_angle/180*np.pi)-car_length/2*np.sin(np.abs(car_angle/180*np.pi)),y-car_length/2*np.cos(car_angle/180*np.pi))-car_width/2*np.sin(np.abs(car_angle/180*np.pi)))
def blue_car(x,y,car_width,car_length,car_angle=0):
    BlueCar_rota = pygame.transform.rotate(BlueCar,car_angle)
    gameDisplay.blit(BlueCar_rota, (x-car_width/2*np.cos(car_angle/180*np.pi)-car_length/2*np.sin(np.abs(car_angle/180*np.pi)),y-car_length/2*np.cos(car_angle/180*np.pi))-car_width/2*np.sin(np.abs(car_angle/180*np.pi)))
def pink_car(x,y,car_width,car_length,car_angle=0):
    PinkCar_rota = pygame.transform.rotate(PinkCar,car_angle)
    gameDisplay.blit(PinkCar_rota, (x-car_width/2*np.cos(car_angle/180*np.pi)-car_length/2*np.sin(np.abs(car_angle/180*np.pi)),y-car_length/2*np.cos(car_angle/180*np.pi))-car_width/2*np.sin(np.abs(car_angle/180*np.pi))) 
def grey_car(x,y,car_width,car_length,car_angle=0):
    GreyCar_rota = pygame.transform.rotate(GreyCar,car_angle)
    gameDisplay.blit(GreyCar_rota, (x-car_width/2*np.cos(car_angle/180*np.pi)-car_length/2*np.sin(np.abs(car_angle/180*np.pi)),y-car_length/2*np.cos(car_angle/180*np.pi))-car_width/2*np.sin(np.abs(car_angle/180*np.pi)))
def purple_car(x,y,car_width,car_length,car_angle=0):
    PurpleCar_rota = pygame.transform.rotate(PurpleCar,car_angle)
    gameDisplay.blit(PurpleCar_rota, (x-car_width/2*np.cos(car_angle/180*np.pi)-car_length/2*np.sin(np.abs(car_angle/180*np.pi)),y-car_length/2*np.cos(car_angle/180*np.pi))-car_width/2*np.sin(np.abs(car_angle/180*np.pi)))
def red_car(x,y,car_width,car_length,car_angle=0):
    RedCar_rota = pygame.transform.rotate(RedCar,car_angle)
    gameDisplay.blit(RedCar_rota, (x-car_width/2*np.cos(car_angle/180*np.pi)-car_length/2*np.sin(np.abs(car_angle/180*np.pi)),y-car_length/2*np.cos(car_angle/180*np.pi))-car_width/2*np.sin(np.abs(car_angle/180*np.pi)))
def yellow_car(x,y,car_width,car_length,car_angle=0):
    YellowCar_rota = pygame.transform.rotate(YellowCar,car_angle)
    gameDisplay.blit(YellowCar_rota, (x-car_width/2*np.cos(car_angle/180*np.pi)-car_length/2*np.sin(np.abs(car_angle/180*np.pi)),y-car_length/2*np.cos(car_angle/180*np.pi))-car_width/2*np.sin(np.abs(car_angle/180*np.pi)))

def green_truck(x,y,truck_width,truck_length,car_angle=0):
    GreenTruck_rota = pygame.transform.rotate(GreenTruck,car_angle)
    gameDisplay.blit(GreenTruck_rota, (x-truck_width/2*np.cos(car_angle/180*np.pi)-truck_length/2*np.sin(np.abs(car_angle/180*np.pi)),y-truck_length/2*np.cos(car_angle/180*np.pi))-truck_width/2*np.sin(np.abs(car_angle/180*np.pi)))
def blue_truck(x,y,truck_width,truck_length,car_angle=0):
    BlueTruck_rota = pygame.transform.rotate(BlueTruck,car_angle)
    gameDisplay.blit(BlueTruck_rota, (x-truck_width/2*np.cos(car_angle/180*np.pi)-truck_length/2*np.sin(np.abs(car_angle/180*np.pi)),y-truck_length/2*np.cos(car_angle/180*np.pi))-truck_width/2*np.sin(np.abs(car_angle/180*np.pi)))
def grey_truck(x,y,truck_width,truck_length,car_angle=0):
    GreyTruck_rota = pygame.transform.rotate(GreyTruck,car_angle)
    gameDisplay.blit(GreyTruck_rota, (x-truck_width/2*np.cos(car_angle/180*np.pi)-truck_length/2*np.sin(np.abs(car_angle/180*np.pi)),y-truck_length/2*np.cos(car_angle/180*np.pi))-truck_width/2*np.sin(np.abs(car_angle/180*np.pi)))
def red_truck(x,y,truck_width,truck_length,car_angle=0):
    RedTruck_rota = pygame.transform.rotate(RedTruck,car_angle)
    gameDisplay.blit(RedTruck_rota, (x-truck_width/2*np.cos(car_angle/180*np.pi)-truck_length/2*np.sin(np.abs(car_angle/180*np.pi)),y-truck_length/2*np.cos(car_angle/180*np.pi))-truck_width/2*np.sin(np.abs(car_angle/180*np.pi)))
def yellow_truck(x,y,truck_width,truck_length,car_angle=0):
    YellowTruck_rota = pygame.transform.rotate(YellowTruck,car_angle)
    gameDisplay.blit(YellowTruck_rota, (x-truck_width/2*np.cos(car_angle/180*np.pi)-truck_length/2*np.sin(np.abs(car_angle/180*np.pi)),y-truck_length/2*np.cos(car_angle/180*np.pi))-truck_width/2*np.sin(np.abs(car_angle/180*np.pi)))
    
def straight(x,y,width,length,road_angle=0):   # left: angle>0  right:angle<0, (x,y): bottom line center
    Straight_rota = pygame.transform.rotate(Straight,road_angle)
    if road_angle>=0:
        gameDisplay.blit(Straight_rota, (x-width/2*np.cos(road_angle/180*np.pi)-length*np.sin(road_angle/180*np.pi),y-length*np.cos(road_angle/180*np.pi)-width/2*np.sin(road_angle/180*np.pi))) 
    else:
        gameDisplay.blit(Straight_rota, (x-width/2*np.cos(-road_angle/180*np.pi),y-length*np.cos(-road_angle/180*np.pi)-width/2*np.sin(-road_angle/180*np.pi))) 
   
def circuit(x,y,circuit_edge,circuit_angle=0):  # left: angle>0  right:angle<0, (x,y): quadrant circuit center
    if circuit_angle>=0:
        angle_aux = circuit_angle%90 + 0
        Circuit_rota1 = pygame.transform.rotate(Circuit,angle_aux)
        Circuit_rota2 = pygame.transform.rotate(Circuit,-(90-angle_aux)+180)
        gameDisplay.blit(Circuit_rota1, (x-circuit_edge*np.sin(angle_aux/180*np.pi),y-circuit_edge*np.cos(angle_aux/180*np.pi)-circuit_edge*np.sin(angle_aux/180*np.pi)))    
        gameDisplay.blit(Circuit_rota2, (x-circuit_edge*np.cos(angle_aux/180*np.pi)-circuit_edge*np.sin(angle_aux/180*np.pi),y-circuit_edge*np.cos(angle_aux/180*np.pi)))
    else:
        angle_aux = circuit_angle%90 + 0
        Circuit_rota1 = pygame.transform.rotate(Circuit,angle_aux)
        Circuit_rota2 = pygame.transform.rotate(Circuit,-(90-angle_aux))
        gameDisplay.blit(Circuit_rota1, (x-circuit_edge*np.sin(angle_aux/180*np.pi),y-circuit_edge*np.cos(angle_aux/180*np.pi)-circuit_edge*np.sin(angle_aux/180*np.pi)))    
        gameDisplay.blit(Circuit_rota2, (x,y-circuit_edge*np.sin(angle_aux/180*np.pi))) 
   
def left_turn(x,y):
    gameDisplay.blit(LeftTurn, (x,y))  
def right_turn(x,y):
    gameDisplay.blit(RightTurn, (x,y)) 

Car_dict = {'GreenCar':green_car,'BlueCar':blue_car,'PinkCar':pink_car,'GreyCar':grey_car,'PurpleCar':purple_car,'RedCar':red_car,'YellowCar':yellow_car,'GreenTruck':green_truck,'BlueTruck':blue_truck,'GreyTruck':grey_truck,'RedTruck':red_truck,'YellowTruck':yellow_truck}
Car_dict_HV = {'GreenCar_HV':green_car_HV}

dx = -2
dy = 0
Road_x0 = display_width * 0.5
Road_y0 = display_height #* 0.5

circuit_x = display_width-circuit_edge
circuit_y = display_height

Center_x =  display_width * 0.5 + dx
Center_y = display_height * 0.5 + dy

#=====================================================================
Initial_state = [2,1,0,1,0,0,0,0,1 ]
Initial_state_matrix = state2matrix(Initial_state)
aux0 = np.concatenate(([0],Initial_state_matrix[0,:],[0]))
aux1 = np.concatenate(([0],Initial_state_matrix[1,:],[0]))
aux2 = np.concatenate(([0],Initial_state_matrix[2,:],[0]))    
Initial_big_state_matrix = np.array([[1,0,0,0,0],aux0,aux1,aux2,[0,0,0,1,0]])

Initial_HV_lane = NrLanes-3    # HV_lane range from 0-4
Nr_action = 5   # total number of actions    

Find_EV_indices = np.asarray(np.where(Initial_big_state_matrix == 1 )).T
Initial_EV = len(Find_EV_indices)   # total number of EVs
EV_name_set = set(np.array(range(Initial_EV))+1)   # name list of active EVs, start from 1, reserve 0 for HV (faked EV)
EV_pos_list = Find_EV_indices
# epsilon = 1e-10
    
#========================= Plot Initial Figure ================================ 
straight(Road_x0,Road_y0,road_width,road_length,0)

Curvature_Class = 0
HV_color_index = 0
Policy = np.loadtxt('save_optimal_policy.txt')
#Policy = np.loadtxt('save_optimal_policy_tailgating.txt')
Pos=np.array([Center_x,Center_y])
Vel=10.0
Dt=1.2
HV = HV_obj(Initial_HV_lane, Pos, Vel, Dt, Policy, HV_color_index, Curvature_Class)           
Car_dict_HV[HV.Color](Center_x,Center_y,HV.car_width,HV.car_length)

PositionTable = np.zeros((5,5,2))
for i in range(5):
    for j in range(5):
        PositionTable[i,j,0] = Center_x+(j-2)*width
        PositionTable[i,j,1] = Center_y+(i-2)*HV.grid_len*Resize

EV_set={}
EV_set["EV{}".format(0)]=HV

i=0
for x in EV_name_set:
    EV_pos = np.array(Find_EV_indices[i])
    Pos = PositionTable[EV_pos[0],EV_pos[1],:]
    EV_color_index =  np.random.randint(0, high=len(ColorSet))
    EV_lane=EV_pos[1]
    Vel=12.0
    Dt=1.2
    EV_set["EV{}".format(x)]=EV(EV_lane, Pos, Vel, Dt, EV_color_index, Curvature_Class)
    Car_dict[EV_set["EV{}".format(x)].Color](Pos[0],Pos[1],EV_set["EV{}".format(x)].car_width,EV_set["EV{}".format(x)].car_length)
    i+=1
    
pygame.display.update()
clock.tick(20)

#==============================================================================

state = Initial_state

step = 0
plot_after_time = 0.1 #sec
plot_after_steps = int(plot_after_time/dt)
angle_scale = 1

x_left = display_width * 0.01 
y_left = display_height * 0.45

x_right = display_width * 0.855
y_right = display_height * 0.45 

Curve_Last_Steps = 300*plot_after_steps

save_screen = make_video(gameDisplay)  # initiate the video generator
video = False  # at start: video not active
plot_straight = True
plot_straigt2circuit = False
plot_circuit = False
plot_circuit2straight = False

count = 0
travel_dist_aux=0
road_x_transit = 0
road_y_transit = 0
road_angle_transit = 0
circuit_x = 0
circuit_y = 0

road_x_offset1 = 11 # corner to straigt transition error
road_x_offset2 = 10

circuit_x_offset1 = 110 #10 # straight to corner transition error
circuit_y_offset1 = 0
circuit_x_offset2 = 120 #20
circuit_y_offset2 = 0

while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True
        elif event.type == VIDEORESIZE:
            gameDisplay = pygame.display.set_mode((event.w, event.h),pygame.RESIZABLE)
        # toggle video on/off by clicking 'v' on keyboard #
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_v:
            video = not video
            
    # update curvature first       
    Curvature_Class_aux = (step//Curve_Last_Steps)%4
    if Curvature_Class_aux == 3:
        HV.Curvature_Class = 1
    else:
        HV.Curvature_Class = Curvature_Class_aux+0
        
    if plot_straight == True: 
        travel_dist_aux+=HV.vel*dt*Resize
        modeNr = travel_dist_aux//road_length
        if step%plot_after_steps==0:
            straight(Road_x0,Road_y0-road_length*modeNr+travel_dist_aux,road_width,road_length)
            straight(Road_x0,Road_y0-road_length*(modeNr+1)+travel_dist_aux,road_width,road_length)

        if HV.Curvature_Class != 1:
            plot_straight = False
            plot_straigt2circuit = True
            residue_travel = travel_dist_aux%road_length-road_margin #position of window top related to road connection line            
            travel_dist_aux=0
            
    if plot_straigt2circuit == True and plot_straight == False:
        if residue_travel <= 0:             
            if HV.Curvature_Class == 0:
                circuit_x = display_width-circuit_edge+circuit_x_offset1
                angle = -1e-2
            elif HV.Curvature_Class == 2:
                circuit_x = circuit_edge-circuit_x_offset2
                angle = 90-1e-2 #make show possible
            circuit_y = residue_travel+travel_dist_aux
            travel_dist_aux+=HV.vel*dt*Resize
            if circuit_y >= display_height:
                circuit_y = display_height+0
                plot_straigt2circuit = False
                plot_circuit = True
                travel_dist_aux=0
            if step%plot_after_steps==0:
                circuit(circuit_x,circuit_y,circuit_edge,angle)
                straight(Road_x0,circuit_y+road_length,road_width,road_length)
        elif residue_travel > 0:             
            if HV.Curvature_Class == 0:
                circuit_x = display_width-circuit_edge+circuit_x_offset1
                angle = -1e-2
            elif HV.Curvature_Class == 2:
                circuit_x = circuit_edge-circuit_x_offset2
                angle = 90-1e-2 #make show possible
            circuit_y = -(road_length-residue_travel)+travel_dist_aux #vel_step*count
            travel_dist_aux+=HV.vel*dt*Resize #count = count+1
            if circuit_y >= display_height:
                circuit_y = display_height+0
                plot_straigt2circuit = False
                plot_circuit = True
                travel_dist_aux=0  
            if step%plot_after_steps==0:    
                circuit(circuit_x,circuit_y,circuit_edge,angle)
                straight(Road_x0,circuit_y+road_length,road_width,road_length) 
                straight(Road_x0,circuit_y+2*road_length,road_width,road_length)
        angle = 0
    
    if plot_straigt2circuit == False and plot_circuit == True:
        if HV.Curvature_Class == 0:
            circuit_x = display_width-circuit_edge+circuit_x_offset1
            angle = angle-HV.vel*dt*Resize/R3/np.pi*180*angle_scale
        elif  HV.Curvature_Class == 2:
            circuit_x = circuit_edge-circuit_x_offset2
            angle = angle+HV.vel*dt*Resize/R3/np.pi*180*angle_scale
        circuit_y = display_height
        
        if step%plot_after_steps==0:
            circuit(circuit_x,circuit_y,circuit_edge,angle)
            
        if HV.Curvature_Class == 1:
           plot_circuit = False
           plot_circuit2straight = True
           phi = np.arcsin(display_height/R3)/np.pi*180
           angle_pre_transit = angle+0 # circuit angle
           phi_transit = phi+0
   
    if plot_circuit == False and plot_circuit2straight == True:
        circuit_y = display_height
        if count != 0:
            phi_transit = phi_transit-HV.vel*dt*Resize/R3/np.pi*180*angle_scale
        if angle_pre_transit<0: # CW, left turn
            circuit_x = display_width-circuit_edge+circuit_x_offset1
            angle = angle-HV.vel*dt*Resize/R3/np.pi*180*angle_scale
            count = count+1
            if angle-angle_pre_transit<=-phi:
                angle = angle_pre_transit-phi
                phi_transit = 0
                count = 0
                plot_circuit2straight = False
                plot_straight = True 
            road_x_transit = circuit_x+R3*mh.cos(phi_transit/180*np.pi)+road_x_offset1      
            road_angle_transit = phi_transit+0

        elif angle>0: # CCW, right turn
            circuit_x = circuit_edge-circuit_x_offset2
            angle = angle+HV.vel*dt*Resize/R3/np.pi*180*angle_scale
            count = count+1
            if angle-angle_pre_transit>=phi:
                angle = angle_pre_transit+phi
                phi_transit = 0
                count = 0
                plot_circuit2straight = False
                plot_straight = True
            road_x_transit = circuit_x-R3*mh.cos(phi_transit/180*np.pi)-road_x_offset2  
            road_angle_transit = -phi_transit+0
            
        road_y_transit = display_height-R3*mh.sin(phi_transit/180*np.pi) 
        if step%plot_after_steps==0:
            circuit(circuit_x,circuit_y,circuit_edge,angle)
            straight(road_x_transit,road_y_transit,road_width,road_length,road_angle_transit)

    state_HV,car_front=get_state_HV(EV_set,EV_name_set,HV)
    HV.TakeAction_HV(state_HV,car_front)
    if step%plot_after_steps==0:    
        x_aux2,y_aux2,angle_aux2 = mapping2curve(plot_straight,plot_straigt2circuit,plot_circuit,plot_circuit2straight,HV.pos[0],HV.pos[1],circuit_x,circuit_y,HV.Curvature_Class,road_y_transit,road_angle_transit,R1,R2,R3,R4,R5,display_width,display_height)
        Car_dict_HV[HV.Color](x_aux2,y_aux2,HV.car_width,HV.car_length,angle_aux2+HV.psi/np.pi*180-90) 
    
    EV_name_set0 = EV_name_set|set()
    for x in EV_name_set:
        EV_name_set_aux = EV_name_set|set()
        EV_name_set_aux.remove(x)
        EV_name_set_aux.add(0)  # introduce HV      
        VehicleArray_EV,car_front=get_state_EV(EV_set,EV_name_set_aux,EV_set["EV{}".format(x)])
        EV_set["EV{}".format(x)].TakeAction(VehicleArray_EV,car_front,HV)
        x_aux=EV_set["EV{}".format(x)].pos[0]
        y_aux=EV_set["EV{}".format(x)].pos[1]
        if step%plot_after_steps==0:
            x_aux2,y_aux2,angle_aux2 = mapping2curve(plot_straight,plot_straigt2circuit,plot_circuit,plot_circuit2straight,x_aux,y_aux,circuit_x,circuit_y,HV.Curvature_Class,road_y_transit,road_angle_transit,R1,R2,R3,R4,R5,display_width,display_height)
            angle_aux3=angle_aux2+EV_set["EV{}".format(x)].psi/np.pi*180-90
            Car_dict[EV_set["EV{}".format(x)].Color](x_aux2,y_aux2,EV_set["EV{}".format(x)].car_width,EV_set["EV{}".format(x)].car_length,angle_aux3)
        y_low=-EV_set["EV{}".format(x)].grid_len*Resize-EV_set["EV{}".format(x)].car_length/2.0
        y_up=display_height+EV_set["EV{}".format(x)].grid_len*Resize+EV_set["EV{}".format(x)].car_length/2.0
        if y_aux<y_low or y_aux>y_up:
            EV_name_set0.remove(x)
    EV_name_set=EV_name_set0|set()
        
    if step%Period==0:
        EV_set,EV_name_set=add_EVs(EV_set,EV_name_set,HV)
### end ### HV takes actions and plot new vehicles positions ##########  
       
############################## add road signs begin ###########################   
    if step%plot_after_steps==0:    
        if HV.Curvature_Class == 0:
            left_turn(x_right,y_right)
        elif HV.Curvature_Class == 2:
            right_turn(x_left,y_left)   

############################## add road signs end #############################       
        textsurface = myfont.render('Speed {} km/h'.format(round(HV.vel*3.6,2)), False, (0, 0, 0))
        if HV.Curvature_Class == 0:
            gameDisplay.blit(textsurface,(500,20)) 
        else:
            gameDisplay.blit(textsurface,(10,20))
###############################################################################        
        pygame.display.update()  
        print('Simulated step: {}'.format(step))
        
    clock.tick(30)
    if video:
        next(save_screen)  # call the generator
        
    step = step + 1
    

pygame.quit()
quit()

