import random
from math import sqrt
import xlwings as xw
import pandas as pd
import time

start_time = time.time()

random.seed(510420)

number_of_trials = 10000

walls_U_min = 0.02
walls_U_max = 0.18

min_width=3
max_width=50

min_length=13
max_length=50

floors_max=8

min_floor_heigth=2
max_floor_heigth=3.5

empty=0
inf_rate_min = 0.5
inf_rate_max = 6  

wb = xw.Book('ZEBRA(extra).xlsx')
sht = wb.sheets("extra") # Get input page open

headers = ['Walls U Value', 'Walls Area', 'Floor U Value', 'Floor Area', 'Roof U value', 'Roof Area', 'Maintenance Factor', 
                      'Mid-pane U value', 'Av Frame Width','Glazing g', 'Frame U value', 'Psi Factor', 'Infiltration Rate', 'Ventilation Rate',
                      'Shelter Factor', 'TFA', 'Volume', 'Occupants', 'Metabolic Gains', 'Electrical Gains', 'MVHR efficiency', 'W1H', 'W1W', 'W1A', 'W1B', 'W1dir',
                      'W2H', 'W2W', 'W2A', 'W2B', 'W2dir', 'W3H', 'W3W', 'W3A', 'W3B', 'W3dir', 'W4H', 'W4W', 'W4A', 'W4B', 'W4dir',
                      'Linear TB coef', 'length of TB','Point TB coef', 'Number of point TB', 'Floors']            
results = ['Total_heating_E', 'Cooling', 'Heat_n_cooling', 'Heating', 'Cooling2', 'Total_primary_energy_use']
headers.extend(results)

finalList = []

def rand(x,y):
    val = round(random.uniform(x,y),5)
    return val

i1=0
for item in range (number_of_trials):
    
        print (number_of_trials - i1)
        i1 += 1
        
        # Materials
        wall_U_value = rand(walls_U_min, walls_U_max)
        floor_U_value = rand(walls_U_min, walls_U_max)
        roof_U_value = random.uniform(walls_U_min,walls_U_max)
       
        # Geometry            
        w=random.uniform(min_width, max_width) #width
        l=random.uniform(min_length, max_length) #length
        floor_area = w*l        
        floors = random.randint(1,floors_max) #number of storyes          
        
        TFA = random.uniform(0.9,0.99)*floor_area
        if floors != 1: TFA = TFA + rand(0.7,1)*rand(0.9,0.99)*(floors - 1)*floor_area

        roof_area = floor_area*random.uniform(1,1.3)                                              
        floor_height = random.uniform(min_floor_heigth, max_floor_heigth)
        volume = TFA*floor_height*floors*rand(1,1.5)       
        walls_area = 2*(w+l)*floor_height*floors*rand(1,1.7) #to account to the fact that the same floor can give different wall area depending on plan
                                       
        tot_windows_area = walls_area   
        
        while tot_windows_area >= walls_area: #make sure that the total window area is less than that of the wall
            window_to_TFA_ratio=rand(0.15,0.4)
            tot_windows_area=window_to_TFA_ratio*TFA  
            continue  
        wall_area = walls_area - tot_windows_area
        
        Total_surface_A = wall_area + roof_area
        
        #Windows
        maint= rand (0.5, 0.9)
        frame_U = rand (0.05, 3)
        psi = rand(0.03, 0.07)
        mid_pane_U_value = rand(0.85,5)
        frame_width= rand(0.5,0.1)*floors
        glazing_g = rand(0.3,0.85)
        
        w1_A = w2_A = w3_A = w4_A = 0  
        
        i = 1
        while i > 0.05: #randomize each window area given the total area of all windows 
            w1_A += rand(0,i/4)
            w2_A += rand(0,i/4)
            w3_A += rand(0,i/4)
            w4_A += rand(0,i/4)       
            i = tot_windows_area - (w1_A + w2_A + w3_A + w4_A)
            
        class windows:          
            def __init__(self, area, direction):
                self.area = area
                self.alpha = rand(0, 90)
                self.betha = rand(0, 180)
                self.direction = direction
                ratio=rand(0.2,1)          
                o=random.randint(0, 1)
                if o==1: #randomises which is greate, width or height 
                    self.width = ratio*sqrt(area)
                    self.height = area/self.width
                else:
                    self.height = ratio*sqrt(area)
                    self.width = area/self.height
                self.list = [self.height, self.width, self.alpha, self.betha, self.direction]    
                                            
        tilt = random.randint(0,89)          
        w1 = windows(w1_A, 0+tilt)
        w2 = windows(w2_A, 90+tilt)
        w3 = windows(w3_A, 180+tilt)
        w4 = windows(w4_A, 270+tilt) 
              
        windows = w1.list + w2.list + w3.list + w4.list   
        
        #Air 
        infiltration_rate = rand(inf_rate_min, inf_rate_max)
        ventilation_rate=rand(5, 30)
        shelter_fac=rand(0.03, 0.1)
        
        occupants = random.uniform(0.02*TFA,0.25*TFA)  
        
    
        # Gains
        met_gain = random.uniform(1,4.5)
        elec_gain = random.uniform(0.7,1.5)
        MVHR = random.uniform(0.7,0.95)
        
        #thermal bridges 
        TH_lin_c = rand(-0.1,0.4)
        
        TH_pt_c = rand(0, 30)
        
        TH_lengh = rand(0,1)*2*(w+l)*floors*2
        
        TH_nb = rand(0,1)*Total_surface_A
        
        floors = float(floors)
        
        TH = [TH_lin_c, TH_lengh, TH_pt_c, TH_nb]        
        row = [wall_U_value, wall_area, floor_U_value, floor_area, roof_U_value, roof_area, maint, mid_pane_U_value, frame_width, 
                   glazing_g, frame_U, psi, infiltration_rate,ventilation_rate, shelter_fac, TFA, volume, occupants, met_gain, elec_gain, 
                   MVHR]
        row.extend(windows)
        row.extend(TH)
        row.append(floors)
        
        sht.range('A6').value = row #add numbers to the input row     
        results = sht.range('A12:F12').value    
        row.extend(results)
        
        row = [round(num, 4) for num in row] 
        
        finalList.append(row)
                
resultsTable = pd.DataFrame(finalList, columns=['Walls U Value', 'Walls Area', 'Floor U Value', 'Floor Area', 'Roof U value', 'Roof Area', 'Maintenance Factor', 
                      'Mid-pane U value', 'Av Frame Width','Glazing g', 'Frame U value', 'Psi Factor', 'Infiltration Rate', 'Ventilation Rate',
                      'Shelter Factor', 'TFA', 'Volume', 'Occupants', 'Metabolic Gains', 'Electrical Gains', 'MVHR efficiency', 'W1H', 'W1W', 'W1A', 'W1B', 'W1dir',
                      'W2H', 'W2W', 'W2A', 'W2B', 'W2dir', 'W3H', 'W3W', 'W3A', 'W3B', 'W3dir', 'W4H', 'W4W', 'W4A', 'W4B', 'W4dir',
                      'Linear TB coef', 'length of TB','Point TB coef', 'Number of point TB', 'Floors', 'Total Heating E', 'Cooling', 'Heat_n_cooling', 'Heating', 'Cooling2', 'Total_primary_energy_use'])   
resultsTable_reduced = resultsTable.iloc[:, :-5]

resultsTable_reduced.to_csv('realistic_sample.csv', index=False)