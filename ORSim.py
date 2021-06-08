'''

Designed by PSU Hershey Capstone Group Spring 2021

Nathan Kurtz, Aaron Harter, Drew Bennison, Kevin Gardner, Michelle Soeganto, Olivia Kucenski, Praharsh Verma

The objective of this project is to use data science and statistical techniques to create a prototype simulation of the scheduling and utilization of hospital operating rooms (OR)

The ultimate goal is to be able to test different implementation strategies:
    * Bumping cases for emergent cases
    * Scheduling optimization strategies

Input Data
The code takes in .xlsx files formatted like the following file we were provided:
    * Scheduled Cases - "OR_Model_Final_PSH.xlsx"
    * Cancelled Cases* - "2019_Cancelled_Cases_Complete_Clean.xlxs"

This code will take in the aforementioned input files, and output the planned and simulated schedules.

'''


# load required packages - if any are not found, open anaconda prompt  and use "conda install pandas" OR "pip install pandas"
#or use the requirements.txt file
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from sklearn.neighbors import KernelDensity
from numpy import array, linspace
import numpy as np
from matplotlib.pyplot import plot
import random
from matplotlib import colors as mcolors
import openpyxl
pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', 100)

# read in the data from the input files
dt = pd.read_excel("OR_Model_Final_PSH.xlsx", engine='openpyxl')
cancelled_cases = pd.read_excel("2019_Cancelled_Cases_Complete_Clean.xlsx", engine='openpyxl')

'''

Parses and Organizes Data from Scheduled Cases Dataset (OR_Model_Final_PSH.xlsx)

'''

###################### Clean Actual Cases Data #########################################################
# create some new columns in the data for other useful stats
# scheduled vs. actual case length
dt['scheduled_case_duration'] = dt['SCH_END'] - dt['SCH_START']
dt['actual_case_duration'] = dt['OUT_ROOM_TIME'] - dt['IN_ROOM_TIME']
# scheduled vs. actual case length in seconds
dt['actual_case_duration_seconds'] = (dt['OUT_ROOM_TIME'] - dt['IN_ROOM_TIME']).dt.total_seconds()
dt['scheduled_case_duration_seconds'] = (dt['SCH_END'] - dt['SCH_START']).dt.total_seconds()
# scheduled vs. actual case length in minutes
dt['actual_case_duration_minute'] = pd.to_numeric(dt['actual_case_duration_seconds'])/60
dt['scheduled_case_duration_minute'] = pd.to_numeric(dt['scheduled_case_duration_seconds'])/60
# make columns for just time regardless of date - will be used later to model variability in starting time
dt['scheduled_start_time'] = dt['SCH_START'].apply(lambda x : datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").time())
dt['actual_start_time'] = dt['IN_ROOM_TIME'].apply(lambda x : datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").time())
#add month column
dt['SCH_START_MONTH'] = dt['SCH_START'].dt.strftime('%b')
dt['ACTUAL_START_MONTH'] = dt['IN_ROOM_TIME'].dt.strftime('%b')
#calculate difference between actual and planned numbers
dt['actual_minus_scheduled_case_duration_minute'] = dt['actual_case_duration_minute'] - dt['scheduled_case_duration_minute']
dt['actual_minus_scheduled_case_start_time'] = (dt['IN_ROOM_TIME'] - dt['SCH_START']).dt.total_seconds()/60
#fix capitalization
dt['WEEKDAY'] = dt['WEEKDAY'].str.capitalize()

def classify(inp):
    '''
    Function defines cases as follows:
    1) Stat/Emergent
    2) Urgent
    3) Priority
    -1) Non-emergency classification
    '''
    
    if inp < 1: #emergency
        return 1
    elif inp < 6:
        return 2 #urgent
    elif inp < 24:
        return 3 #priority
    else:
        return 0 #elective
    
def day_of_week(inp):
    if inp == 0:
        return "Mon"
    elif inp == 1:
        return "Tue"
    elif inp == 2:
        return "Wed"
    elif inp == 3:
        return "Thu"
    elif inp == 4:
        return "Fri"
    elif inp == 5:
        return "Sat"
    else:
        return "Sun"
    
dt['ACTUAL_WEEKDAY'] = (dt['IN_ROOM_TIME']).dt.dayofweek.apply(day_of_week)
dt['calculated_add_on_hours']=(dt['SCH_START']-dt['ORIG_SCH_DATE']).apply(lambda x : x.seconds/360)
dt['new_emergency_classification'] = dt['calculated_add_on_hours'].apply(classify)
dt['cancelled_flag'] = 0

'''

Parses and Organizes Data from Cancelled Cases Dataset (2019_Cancelled_Cases_Complete_Clean.xlxs)

'''

################################# Clean Cancelled Cases Data ##############################################################
#rename columns of cancelled data and make new columns similar to what we did with the regular data
cancelled_cases = cancelled_cases.rename(columns={"Case Number Formatted": "CASE_NBR", "Scheduled OR Number": "SCH_OR",
                                                  "Service Line": "Service_Line", "Service Line Dept": "Service_Line_Dept",
                                                "Cancelled Date and time": "CANCELLED_DATE", "Scheduled Start  Date and Time": "SCH_START"})

cancelled_cases['scheduled_case_duration'] = cancelled_cases['SCH_END'] - cancelled_cases['SCH_START']
cancelled_cases['scheduled_case_duration_seconds'] = (cancelled_cases['SCH_END'] - cancelled_cases['SCH_START']).dt.total_seconds()
cancelled_cases['scheduled_case_duration_minute'] = pd.to_numeric(cancelled_cases['scheduled_case_duration_seconds'])/60
cancelled_cases['scheduled_start_time'] = cancelled_cases['SCH_START'].apply(lambda x : datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").time())
cancelled_cases['SCH_START_MONTH'] = cancelled_cases['SCH_START'].dt.strftime('%b')

#Add cancelled flag, week day, and case date    
cancelled_cases['cancelled_flag'] = 1
cancelled_cases['WEEKDAY'] = (cancelled_cases['SCH_START']).dt.dayofweek.apply(day_of_week)
cancelled_cases['CASE_DATE'] = cancelled_cases['SCH_START'].dt.floor("D") #.dt.date.

'''

Combines Cancelled and Scheduled Cases into one Pandas DataFrame

'''

#Make the combined data set
combined_data = pd.concat([dt, cancelled_cases])


'''

Filters OR Rooms as well as retains the unique set of OR rooms from the combined_data DataFrame

'''

###################### Ignore rooms in simulation and create a full list of rooms to use #######################################
#Find the list of rooms to use during the analysis and sort them by order
all_or_rooms = combined_data.SCH_OR.unique()
ignore_or = ["At Bedside", "MOR CATH 01", "MPR 01", "SPR 04", "CHPR 02", "CHPR 01", "MOR Add On 2", "CHOR Add On 1",
            "MOR Standby 1", "MOR Standby 2", "CHOR Standby 1", "CHPR Add On 1","MOR Add On 1"]
all_or_rooms = list(set(all_or_rooms).difference(ignore_or))
all_or_rooms = [x for x in all_or_rooms if str(x) != 'nan']
all_or_rooms.sort(reverse=False)


'''

This is the bulk of the logic. Takes in all formatted data and performs scheduling logic.

'''

############################# Define the main class ########################################################################
class HersheyORSim:
    #set initial variables
    def __init__(self, selected_month="Jan", selected_weekday="Mon", selected_cutoff_time=.2916666):
    	#cut off time determines the hours before midnight when we are generating the schedule
    	#default is 5 p.m.
        self.selected_month = selected_month
        self.selected_weekday = selected_weekday
        self.selected_cutoff_time = selected_cutoff_time
    #plan schedule function
    def planSchedule(self):
        d = {'CASE_NBR': [], 'SCH_START': [], 'SCH_END': [], 'SCH_OR': [], 'Service_Line': [], 'CANCELLED': []}
        scheduled_cases = pd.DataFrame(data=d)
        
        for or_room in all_or_rooms:
            # filter data based on selections
            or_single = combined_data[(combined_data.SCH_OR == or_room) & (combined_data.SCH_START_MONTH == self.selected_month) & (combined_data.WEEKDAY == self.selected_weekday)]

            # filter out cases that were scheduled not before cutoff time the previous day (these are emergency cases mostly) or were cancelled after the cutoff time
            or_single = or_single[((or_single.ORIG_SCH_DATE < (or_single.CASE_DATE-timedelta(days=self.selected_cutoff_time))) | pd.isnull(or_single.ORIG_SCH_DATE)) & ((or_single.CANCELLED_DATE > or_single.CASE_DATE-timedelta(days=self.selected_cutoff_time)) | (or_single.cancelled_flag ==0))]
            # skip room if no data
            if len(or_single) == 0:
                continue
            
            # rank the cases by time each day - we can use this to find the first, second, third, etc. case of each day
            or_single['case_order_scheduled'] = or_single.groupby("CASE_DATE")["SCH_START"].rank("dense", ascending=True)
            
            ##################################################################################################
            #Find the 90th percentile time a case has been scheduled to end in a room for use later
            max_time_or = or_single['SCH_END'].apply(lambda x : datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").time())
            max_time_or = max_time_or.sort_values(ascending=True, ignore_index=True)
            max_time_or = max_time_or.iloc[round(.9*len(max_time_or))-1]

            # Start by seeing how many cases there will be that day
            num_cases = or_single[['CASE_DATE', 'CASE_NBR']].groupby(['CASE_DATE']).agg(['count']).reset_index()
            num_cases = num_cases[['CASE_NBR']]
            
            # the next few lines create a KDE of the number of cases that will be in the OR that day
            num_cases_numpy = num_cases.to_numpy().reshape(-1, 1)
            num_cases_kde = KernelDensity(kernel='gaussian', bandwidth=.3).fit(num_cases_numpy)
            # pull a random draw from the distribution - rerunning this block will create a new draw each time
            num_cases_draw = round(num_cases_kde.sample(1)[0][0])

            #################################################################################################

            days_cases = []
            days_cancelled = []

            # for each of the cases that is scheduled for that day, what is that case?
            for i in range(1, int(num_cases_draw)+1):
                # what is the probability of each type of case being the ith case of the day?
                or_single_sl_prob_start = or_single[(or_single.case_order_scheduled == i)]
                or_single_sl_prob = or_single_sl_prob_start[['Service_Line', 'CASE_NBR', 'cancelled_flag']].groupby(['Service_Line', 'cancelled_flag']).agg(['count']).reset_index().sort_values(by=('CASE_NBR', 'count'), ascending=False)
                total_cases = len(or_single_sl_prob_start)
                or_single_sl_prob['prob'] = or_single_sl_prob[['CASE_NBR']]/total_cases

                # random number, pick which type of case with be the ith of the day based on historical probability
                ids = list(range(0, len(or_single_sl_prob)))
                or_single_sl_prob['ids'] = ids
                
                for id_x in ids:
                    cs = sum(or_single_sl_prob['prob'][or_single_sl_prob['ids'] < id_x+1])
                    or_single_sl_prob.iloc[id_x, 3] = cs

                random_num = random.uniform(0, 1)

                for id_x in ids:
                    if random_num < or_single_sl_prob.iloc[id_x, 3]:
                        case_type = or_single_sl_prob.iloc[id_x,0]
                        cancelled = or_single_sl_prob.iloc[id_x,1]
                        break
                        
                        
                #check if case_type that was just drawn is the same as the one before it - 94% of the time it should
                #if it's the same, just continue
                if i != 1:
                    if case_type != days_cases[i-2]:
                        random_num = random.uniform(0, 1)
                        if random_num < .06:
                            case_type = case_type
                        else:
                            case_type = days_cases[i-2]
            
                        
                days_cases.append(case_type)
                days_cancelled.append(cancelled)
            ################################################################################################

            # What time will the first case start? Random draw from discrete distribution of historical probabilities
            or_single_first_cases = or_single[(or_single.case_order_scheduled == 1)]
            first_case_start_time = or_single_first_cases[['scheduled_start_time']]

            first_case_start_time = or_single_first_cases[['scheduled_start_time', 'CASE_NBR']].groupby(['scheduled_start_time']).agg(['count']).reset_index().sort_values(by=('CASE_NBR', 'count'), ascending=False)
            total_cases = len(or_single_first_cases)
            first_case_start_time['prob'] = first_case_start_time[['CASE_NBR']]/total_cases

            ids = list(range(0, len(first_case_start_time)))
            first_case_start_time['ids'] = ids

            for id_x in ids:
                cs = sum(first_case_start_time['prob'][first_case_start_time['ids'] < id_x+1])
                first_case_start_time.iloc[id_x, 3] = cs

            random_num = random.uniform(0, 1)

            for id_x in ids:
                if random_num < first_case_start_time.iloc[id_x, 3]:
                    starting_time_winner = first_case_start_time.iloc[id_x,0]
                    break

            #date formatting for month
            if self.selected_month == "Jan":
                month_number = 1
            elif self.selected_month == "Feb":
                month_number = 2
            elif self.selected_month == "Mar":
                month_number = 3
            elif self.selected_month == "Apr":
                month_number = 4
            elif self.selected_month == "May":
                month_number = 5
            elif self.selected_month == "Jun":
                month_number = 6
            elif self.selected_month == "Jul":
                month_number = 7
            elif self.selected_month == "Aug":
                month_number = 8
            elif self.selected_month == "Sep":
                month_number = 9
            elif self.selected_month == "Oct":
                month_number = 10
            elif self.selected_month == "Nov":
                month_number = 11
            else:
                month_number = 12
                
            #format starting time and max case end time based on the selected month
            starting_time = datetime.combine(date(2020, month_number, 1), starting_time_winner)
            max_cases_or_time = datetime.combine(date(2020, month_number, 1), max_time_or)

            ##################################################################################################
            #distribution of time between cases to draw from
            or_single_copy = or_single.copy()
            or_single_copy['case_order_scheduled'] = or_single_copy['case_order_scheduled'] - 1
            or_single_time_between = or_single.merge(or_single_copy, how='left', on=['CASE_DATE', 'case_order_scheduled', 'SCH_OR'])
            or_single_time_between = or_single_time_between[['SCH_START_x', 'SCH_END_x', 'SCH_OR', 'SCH_START_y', 'SCH_END_y']]
            or_single_time_between = or_single_time_between[or_single_time_between.SCH_START_y.notnull()]
            or_single_time_between['time_between_surgeries_scheduled'] = (or_single_time_between['SCH_START_y'] - or_single_time_between['SCH_END_x']).dt.total_seconds()/60
            or_single_time_between

            scheduled_time_between = or_single_time_between[['time_between_surgeries_scheduled']]
            scheduled_time_between_numpy = scheduled_time_between.to_numpy().reshape(-1, 1)
            
            #check if there are enough cases to build this data, otherwise set it to just be 10 minutes
            if len(scheduled_time_between) == 0:
                scheduled_time_between_numpy = np.array([10,10,10,10]).reshape(-1, 1)
            kde_time_between_cases = KernelDensity(kernel='gaussian', bandwidth=.02).fit(scheduled_time_between_numpy)

            ##################################################################################################
            #for each case number in that OR
            for case_number in range(0, len(days_cases)):
                # starting case
                if case_number == 0:
                    # with starting case, generate distrubution of that surgery length as originally scheduled
                    or_single_department_line = or_single[(or_single.Service_Line == days_cases[case_number])]
                    X = or_single_department_line[['scheduled_case_duration_minute']]
                    X2 = X.to_numpy().reshape(-1, 1)
                    kde = KernelDensity(kernel='gaussian', bandwidth=.3).fit(X2)
                    #draw case length
                    case_length = round(kde.sample(1)[0][0])
                    #add the first case onto the schedule
                    temp_d = pd.DataFrame(data={'CASE_NBR': [case_number+1], 'SCH_START': [starting_time], 'SCH_END': [starting_time + timedelta(minutes=case_length)], 'SCH_OR': [or_room], 'Service_Line': [days_cases[case_number]], 'CANCELLED': [days_cancelled[case_number]]})
                    scheduled_cases = scheduled_cases.append(temp_d)
                else: 
                    # distribution of turnover time
                    scheduled_time_between_draw = round(kde_time_between_cases.sample(1)[0][0])
                    # add turnover time to scheduled end of last case --> this is scheduled start
                    next_case_scheduled_start = scheduled_cases.iloc[len(scheduled_cases)-1,2] + timedelta(minutes= round(kde_time_between_cases.sample(1)[0][0]))

                    # random pull for surgery length --> scheduled end
                    or_single_department_line = or_single[(or_single.Service_Line == days_cases[case_number])]
                    X = or_single_department_line[['scheduled_case_duration_minute']]
                    X2 = X.to_numpy().reshape(-1, 1)
                    kde = KernelDensity(kernel='gaussian', bandwidth=.3).fit(X2)
                    case_length = round(kde.sample(1)[0][0])
                                       
                    #make sure case length isn't too late in the night
                    if next_case_scheduled_start + timedelta(minutes=case_length) < max_cases_or_time:
                        temp_d = pd.DataFrame(data={'CASE_NBR': [case_number+1], 'SCH_START': [next_case_scheduled_start], 'SCH_END': [next_case_scheduled_start + timedelta(minutes=case_length)], 'SCH_OR': [or_room], 'Service_Line': [days_cases[case_number]], 'CANCELLED': [days_cancelled[case_number]]})
                        temp_d['SCH_START'] = temp_d['SCH_START'].dt.round("5min") # round start to nearest 5 minutes
                        scheduled_cases = scheduled_cases.append(temp_d)
                    #try to redraw 5 times to see if we get a shorter case; if we don't, move on
                    else:
                        num_retries = 0
                        while num_retries <5:
                            case_length = round(kde.sample(1)[0][0])
                            if next_case_scheduled_start + timedelta(minutes=case_length) < max_cases_or_time:
                                temp_d = pd.DataFrame(data={'CASE_NBR': [case_number+1], 'SCH_START': [next_case_scheduled_start], 'SCH_END': [next_case_scheduled_start + timedelta(minutes=case_length)], 'SCH_OR': [or_room], 'Service_Line': [days_cases[case_number]], 'CANCELLED': [days_cancelled[case_number]]})
                                temp_d['SCH_START'] = temp_d['SCH_START'].dt.round("5min") # round start to nearest 5 minutes
                                scheduled_cases = scheduled_cases.append(temp_d)
                                break
                            else:
                                num_retries += 1
                        
                        if num_retries == 5:
                            break
                         


        return(scheduled_cases)
    
    #Finds, formats, and returns an actual Hershey planned schedule for simulating
    def selectRealSchedule(self, selected_date):
        selected_date = datetime.strptime(selected_date, '%Y-%m-%d')

        # filter data based on selections
        scheduled_cases = combined_data[(combined_data.CASE_DATE == selected_date)]

        # filter out cases that were scheduled not before cutoff time the previous day (these are emergency cases mostly) or were cancelled after the cutoff time
        scheduled_cases = scheduled_cases[((scheduled_cases.ORIG_SCH_DATE < (scheduled_cases.CASE_DATE-timedelta(days=self.selected_cutoff_time))) | pd.isnull(scheduled_cases.ORIG_SCH_DATE)) & ((scheduled_cases.CANCELLED_DATE > scheduled_cases.CASE_DATE-timedelta(days=self.selected_cutoff_time)) | (scheduled_cases.cancelled_flag == 0))]
        
        #only keep cases that happened in one of the used OR rooms
        scheduled_cases = scheduled_cases[(scheduled_cases.SCH_OR.isin(all_or_rooms))]
        #set cancelled to 0 because we don't know any information about cancelled cases at this time
        scheduled_cases['CANCELLED'] = 0 
        #add in the new case number for order and sort values before returning
        scheduled_cases['CASE_NBR'] = scheduled_cases.groupby("SCH_OR")["SCH_START"].rank("dense", ascending=True)
        scheduled_cases = scheduled_cases.sort_values(by=['SCH_OR', 'SCH_START'])
        scheduled_cases = scheduled_cases[['CASE_NBR', 'SCH_START', 'SCH_END', 'SCH_OR', 'Service_Line',
                                          'CANCELLED']]
        return scheduled_cases
    
    #simulates a planned schedule
    def simulateSchedule(self, planned_schedule):
        #store the actual simulated schedule
        d = {'CASE_NBR': [], 'SCH_START': [], 'SCH_END': [], 'SCH_OR': [], 'Service_Line': [], 'CANCELLED': [],
            "IN_ROOM_TIME": [], "OUT_ROOM_TIME": [], "OR_USED":[]}
        simulated_cases = pd.DataFrame(data=d)
        
        #go through each OR room
        for or_room in all_or_rooms:
            temp_or_room = planned_schedule[planned_schedule.SCH_OR==or_room]
            #if OR room has not scheduled cases, pass and continue onto the next one
            if len(temp_or_room) == 0:
                continue
            
            
            #Filter for the correct room, day, month, non-emergency and non-cancelled
            or_single_simulated = combined_data[(combined_data.OR_USED == or_room) & (combined_data.ACTUAL_START_MONTH == self.selected_month) & (combined_data.ACTUAL_WEEKDAY == self.selected_weekday) & (combined_data.cancelled_flag == 0 & (combined_data.calculated_add_on_hours > 6))]
            
            #calculate a few columns that are useful to use
            or_single_simulated['IN_ROOM_TIME'] = pd.to_datetime(or_single_simulated['IN_ROOM_TIME'])
            or_single_simulated['OUT_ROOM_TIME'] = pd.to_datetime(or_single_simulated['OUT_ROOM_TIME'])
            or_single_simulated['case_order_actual'] = or_single_simulated.groupby("CASE_DATE")["IN_ROOM_TIME"].rank("dense", ascending=True)
            
            #go through each case in that room
            for index, row in temp_or_room.iterrows():
                #check if it's the first case of the day in that OR - if it is,
                #draw from the KDE from same OR, where case number == 1
                if row['CASE_NBR'] == 1:
                    #check if the case is cancelled
                    #first case of the day, cancelled
                    if row['CANCELLED'] == 1:
                        #add the results to the data frame
                        temp_d = pd.DataFrame(data={'CASE_NBR': [row['CASE_NBR']], 'SCH_START': [row['SCH_START']], 'SCH_END': [row['SCH_END']], 'SCH_OR': [row['SCH_OR']], 'Service_Line': [row['Service_Line']], 'CANCELLED': [row['CANCELLED']],
                                               "IN_ROOM_TIME":[np.nan], "OUT_ROOM_TIME":[np.nan], "OR_USED":[""]})
                    
                        simulated_cases = simulated_cases.append(temp_d)
                    #first case of the day, not cancelled
                    else:
                        # with starting case, generate distrubution of actual-scheduled case start time ##########################
                        or_single_first_case = or_single_simulated[(or_single_simulated.case_order_actual == 1)]

                        X = or_single_first_case[['actual_minus_scheduled_case_start_time']]
                        X2 = X.to_numpy().reshape(-1, 1)
                        #if no data, make the start time difference distribution
                        if len(or_single_first_case) == 0:
                            X2 = np.array([0,0,-5,5,10,-10]).reshape(-1, 1)
                        #draw difference in case start time if there's data
                        kde = KernelDensity(kernel='gaussian', bandwidth=.3).fit(X2)
                        start_time_difference = round(kde.sample(1)[0][0])
                        
                        #make sure the start time for the first case is reasonable
                        while start_time_difference < -90 or start_time_difference > 90:
                            start_time_difference = round(kde.sample(1)[0][0])
                        ######################################## CASE LENGTH #####################################################
                        # random pull for surgery length --> actual length/difference from planned
                        or_single_department_line = or_single_simulated[(or_single_simulated.Service_Line == row['Service_Line'])]
                        X = or_single_department_line[['actual_minus_scheduled_case_duration_minute']]
                        X2 = X.to_numpy().reshape(-1, 1)
                        if len(or_single_department_line) == 0:
                            X2 = np.array([0,0,0,0]).reshape(-1, 1)
                        kde = KernelDensity(kernel='gaussian', bandwidth=.3).fit(X2)
                        case_length_modifier = round(kde.sample(1)[0][0])
                        
                        #get the actual scheduled case length of this case
                        scheduled_case_length = (row.loc['SCH_END'] - row.loc['SCH_START']).total_seconds()/60
                          
                        #make sure the length of the case is positive; redraw case_length_modifier if not    
                        while scheduled_case_length + case_length_modifier < 10:
                            case_length_modifier = round(kde.sample(1)[0][0])

                        #add the results to the data frame
                        temp_d = pd.DataFrame(data={'CASE_NBR': [row['CASE_NBR']], 'SCH_START': [row['SCH_START']], 'SCH_END': [row['SCH_END']], 'SCH_OR': [row['SCH_OR']], 'Service_Line': [row['Service_Line']], 'CANCELLED': [row['CANCELLED']],
                                                   "IN_ROOM_TIME":[row['SCH_START'] + timedelta(minutes=start_time_difference)], "OUT_ROOM_TIME":[row['SCH_START'] + timedelta(minutes=start_time_difference+scheduled_case_length+case_length_modifier)],
                                                    "OR_USED":[row['SCH_OR']]})

                        simulated_cases = simulated_cases.append(temp_d)
                #second case or later
                else:
                    #second case or later, cancelled
                    if row['CANCELLED'] == 1:
                        temp_d = pd.DataFrame(data={'CASE_NBR': [row['CASE_NBR']], 'SCH_START': [row['SCH_START']], 'SCH_END': [row['SCH_END']], 'SCH_OR': [row['SCH_OR']], 'Service_Line': [row['Service_Line']], 'CANCELLED': [row['CANCELLED']],
                                               "IN_ROOM_TIME":[np.nan], "OUT_ROOM_TIME":[np.nan], "OR_USED":""})
                    
                        simulated_cases = simulated_cases.append(temp_d)
                    #second case or later, not cancelled 
                    else:
                        #bring in previous case data
                        previous_case = simulated_cases.iloc[len(simulated_cases)-1,]
                        #previous case was cancelled
                        if previous_case['CANCELLED'] >=1:
                            #previous case was cancelled, and it could be any number case of the day
                            #take scheduled start time add to it turnover time modifier, but let this modifier be zero or negative
                            #random pull for turnover time
                            #calculate turnover time distribution to draw from
                            or_single_simulated_copy = or_single_simulated[or_single_simulated.Service_Line == row['Service_Line']].copy()
                            or_single_simulated_copy['case_order_actual'] = or_single_simulated_copy['case_order_actual'] - 1
                            or_single_time_between = or_single_simulated[or_single_simulated.Service_Line == row['Service_Line']].merge(or_single_simulated_copy, how='left', on=['CASE_DATE', 'case_order_actual', 'OR_USED'])
                            or_single_time_between = or_single_time_between[['IN_ROOM_TIME_x', 'OUT_ROOM_TIME_x', 'OR_USED', 'IN_ROOM_TIME_y', 'OUT_ROOM_TIME_y', 'SCH_START_x', 'SCH_END_x', 'SCH_START_y', 'SCH_END_y', 'Service_Line_x']]
                            or_single_time_between = or_single_time_between[or_single_time_between.IN_ROOM_TIME_y.notnull()]
                            or_single_time_between['time_between_surgeries_actual'] = (or_single_time_between['IN_ROOM_TIME_y'] - or_single_time_between['OUT_ROOM_TIME_x']).dt.total_seconds()/60
                            or_single_time_between['time_between_surgeries_scheduled'] = (or_single_time_between['SCH_START_y'] - or_single_time_between['SCH_END_x']).dt.total_seconds()/60
                            or_single_time_between['actual_minus_expected_turnover_time'] = or_single_time_between['time_between_surgeries_actual'] - or_single_time_between['time_between_surgeries_scheduled']
                            #return or_single_time_between

                            actual_minus_expected_time_between = or_single_time_between[['actual_minus_expected_turnover_time']]
                            actual_minus_expected_time_between_numpy = actual_minus_expected_time_between.to_numpy().reshape(-1, 1)

                            #check if there are enough cases to build this data, otherwise set it to just be 10 minutes
                            if len(actual_minus_expected_time_between_numpy) == 0:
                                actual_minus_expected_time_between_numpy = np.array([10,10,10,10]).reshape(-1, 1)
                            kde_time_between_cases_actual = KernelDensity(kernel='gaussian', bandwidth=.3).fit(actual_minus_expected_time_between_numpy)
                            
                            actual_time_between_draw = round(kde_time_between_cases_actual.sample(1)[0][0])

                            #check if the turnover time is too extreme - limit of max 60 minutes since the previous case was cancelled
                            too_extreme_count = 0
                            if actual_time_between_draw > 60:
                                while too_extreme_count < 5:
                                    actual_time_between_draw = round(kde_time_between_cases_actual.sample(1)[0][0])
                                    if actual_time_between_draw <= 60:
                                        too_exteme_count = 10
                                    else:
                                        too_extreme_count +=1
                            #if we couldn't get a draw less than 2 hours, manually set turnover time
                            if too_extreme_count == 5:
                                actual_time_between_draw = 25
                            
                                
                            #case duration draw
                            or_single_department_line = or_single_simulated[(or_single_simulated.Service_Line == row['Service_Line'])]
                            X = or_single_department_line[['actual_minus_scheduled_case_duration_minute']]
                            X2 = X.to_numpy().reshape(-1, 1)
                            if len(or_single_department_line) == 0:
                                X2 = np.array([0,0,0,0]).reshape(-1, 1)
                            kde = KernelDensity(kernel='gaussian', bandwidth=.3).fit(X2)
                            case_length_modifier = round(kde.sample(1)[0][0])
                                
                            #make sure the length of the case is positive
                            scheduled_case_length = (row.loc['SCH_END'] - row.loc['SCH_START']).total_seconds()/60
                            
                            #redraw case length modifier if it will result in a case ending before it began
                            while scheduled_case_length + case_length_modifier < 10:
                                case_length_modifier = round(kde.sample(1)[0][0])
                            
                            #add things up and append
                            temp_d = pd.DataFrame(data={'CASE_NBR': [row['CASE_NBR']], 'SCH_START': [row['SCH_START']], 'SCH_END': [row['SCH_END']], 'SCH_OR': [row['SCH_OR']], 'Service_Line': [row['Service_Line']], 'CANCELLED': [row['CANCELLED']],
                                                           "IN_ROOM_TIME":[row['SCH_START'] + timedelta(minutes=actual_time_between_draw)], "OUT_ROOM_TIME":[row['SCH_START'] + timedelta(minutes=actual_time_between_draw+case_length_modifier+scheduled_case_length)],
                                                        "OR_USED":[row['SCH_OR']]})

                            simulated_cases = simulated_cases.append(temp_d)

                            
                        else: #continue as normal, previous case wasn't cancelled and neither was this one
                        
                            #random pull for turnover time
                            #calculate turnover time distribution to draw from
                            or_single_simulated_copy = or_single_simulated[or_single_simulated.Service_Line == row['Service_Line']].copy()
                            or_single_simulated_copy['case_order_actual'] = or_single_simulated_copy['case_order_actual'] - 1
                            or_single_time_between = or_single_simulated[or_single_simulated.Service_Line == row['Service_Line']].merge(or_single_simulated_copy, how='left', on=['CASE_DATE', 'case_order_actual', 'OR_USED'])
                            or_single_time_between = or_single_time_between[['IN_ROOM_TIME_x', 'OUT_ROOM_TIME_x', 'OR_USED', 'IN_ROOM_TIME_y', 'OUT_ROOM_TIME_y', 'SCH_START_x', 'SCH_END_x', 'SCH_START_y', 'SCH_END_y', 'Service_Line_x']]
                            or_single_time_between = or_single_time_between[or_single_time_between.IN_ROOM_TIME_y.notnull()]
                            or_single_time_between['time_between_surgeries_actual'] = (or_single_time_between['IN_ROOM_TIME_y'] - or_single_time_between['OUT_ROOM_TIME_x']).dt.total_seconds()/60
                            or_single_time_between['time_between_surgeries_scheduled'] = (or_single_time_between['SCH_START_y'] - or_single_time_between['SCH_END_x']).dt.total_seconds()/60
                            or_single_time_between['actual_minus_expected_turnover_time'] = or_single_time_between['time_between_surgeries_actual'] - or_single_time_between['time_between_surgeries_scheduled']
                            #return or_single_time_between

                            actual_minus_expected_time_between = or_single_time_between[['actual_minus_expected_turnover_time']]
                            actual_minus_expected_time_between_numpy = actual_minus_expected_time_between.to_numpy().reshape(-1, 1)

                            #check if there are enough cases to build this data, otherwise set it to just be 10 minutes
                            if len(actual_minus_expected_time_between_numpy) == 0:
                                actual_minus_expected_time_between_numpy = np.array([0,5,10,15,20,25]).reshape(-1, 1)
                            kde_time_between_cases_actual = KernelDensity(kernel='gaussian', bandwidth=.3).fit(actual_minus_expected_time_between_numpy)

                            actual_time_between_draw = round(kde_time_between_cases_actual.sample(1)[0][0])
                            
                            #check if the turnover time is too extreme
                            too_extreme_count = 0
                            if actual_time_between_draw > 120:
                                while too_extreme_count < 5:
                                    actual_time_between_draw = round(kde_time_between_cases_actual.sample(1)[0][0])
                                    if actual_time_between_draw <= 120:
                                        too_exteme_count = 10
                                    else:
                                        too_extreme_count +=1
                            #if we couldn't get a draw less than 2 hours, manually set turnover time
                            if too_extreme_count == 5:
                                actual_time_between_draw = 25
                            
                            #bring in previous row and check that the changeover time doesn't make the situation impossible
                            simulated_turnover_time = (row.loc['SCH_START'] - previous_case['SCH_END']).total_seconds()/60 + actual_time_between_draw
                            if simulated_turnover_time < 5:
                                simulated_turnover_time = 30
                                
                                
                            #case duration draw
                            or_single_department_line = or_single_simulated[(or_single_simulated.Service_Line == row['Service_Line'])]
                            X = or_single_department_line[['actual_minus_scheduled_case_duration_minute']]
                            X2 = X.to_numpy().reshape(-1, 1)
                            if len(or_single_department_line) == 0:
                                X2 = np.array([0,0,0,0]).reshape(-1, 1)
                            kde = KernelDensity(kernel='gaussian', bandwidth=.3).fit(X2)
                            case_length_modifier = round(kde.sample(1)[0][0])

    
                            #make sure the length of the case is positive
                            scheduled_case_length = (row.loc['SCH_END'] - row.loc['SCH_START']).total_seconds()/60
                            
                            while scheduled_case_length + case_length_modifier < 10:
                                case_length_modifier = round(kde.sample(1)[0][0])
                                
                            #add things up and append
                            temp_d = pd.DataFrame(data={'CASE_NBR': [row['CASE_NBR']], 'SCH_START': [row['SCH_START']], 'SCH_END': [row['SCH_END']], 'SCH_OR': [row['SCH_OR']], 'Service_Line': [row['Service_Line']], 'CANCELLED': [row['CANCELLED']],
                                                           "IN_ROOM_TIME":[previous_case['OUT_ROOM_TIME'] + timedelta(minutes=simulated_turnover_time)], "OUT_ROOM_TIME":[previous_case['OUT_ROOM_TIME'] + timedelta(minutes=simulated_turnover_time+case_length_modifier+scheduled_case_length)],
                                                       "OR_USED":[row['SCH_OR']]})

                            simulated_cases = simulated_cases.append(temp_d)
                    
                    
        
        
        return simulated_cases

'''

Take the simulated case data and visualize it in a chart that gets saved to your working directory.

'''

def visualizeSchedule(schedule, show_rooms=all_or_rooms):
    '''
    Input: final schedule that includes columns for both planned schedule and actual schedule, list of rooms to show
    Output: graphs of planned vs. actual schedule, saves them to your folder
    '''
    
    #make the list of what rooms should be shown based on user input
    rooms_in_data = list(set(schedule.SCH_OR.unique()))
    new_show_rooms = []
    for room in rooms_in_data:
        if room in show_rooms:
            new_show_rooms.append(room)
    
    #place the rooms in alphabetical order        
    show_rooms = new_show_rooms
    show_rooms = [x for x in show_rooms if str(x) != 'nan']
    show_rooms.sort(reverse=False)
    
    sample_day = schedule.sort_values(by="SCH_OR")
    
    #set line colors for departments
    line_color = {
        "Urology" : "green",
        "Ortho" : "blue",
        "Otolaryngology": "orange",
        "Trauma Surgery": "red",
        "Neurosurgery": "purple",
        "Plastic Surgery": "brown",
        "Vascular Surgery": "pink",
        "OB/Gyn": "gray",
        "Ophthalmology Surgery": "olive",
        "MIS/Bariatric Surgery": "cyan",
        "GSSSO - HPB": "goldenrod",
        "GSSSO - GSO": "magenta",
        "Pediatric Surgery": "teal",
        "Colorectal Surgery": "black",
        "CT Surgery":"tomato",
        "Thoracic Surgery":"sienna",
        "GI-Adult":"darkgoldenrod",
        "PEDS CT Surgery":"forestgreen",
        "Transplant Surgery": "darkseagreen",
        "Dental Surgery":"aquamarine",
        "Miscellaneous Surgery": "darkslategray",
        "Gift of Life":"deepskyblue",
        "Pain": "dodgerblue",
        "Pulmonary - Adult":"rebeccapurple",
        "GI-Peds":"indigo",
        "Pulmonary - Peds":"hotpink"
    }
    
    
    #initialize the graph with a point that will not be on it
    point_plot =[
      go.Scatter(x=["2010-02-05 01:00:00", "2010-02-05 01:00:00"],
                 y=["Standby", "Standby"],
                 name="",
                 line=dict(color="black"))]
    
    #find date of first case of the day to set the proper range
    start_range = sample_day.iloc[0,1].floor("D")
    end_range = start_range + timedelta(minutes=1600)
    
    make_title = "Planned OR schedule"
    layout = go.Layout(title=go.layout.Title(text=make_title,x=0.5),
            xaxis={'title':'Time','range':[start_range, end_range]},
            yaxis={'title':'OR'},
            height = 900)

    fig = go.Figure(data=point_plot, layout=layout)


    departments_used = []

    #loop through each case and add it to the chart
    for i in range(0, len(sample_day)):
        get_start = sample_day['SCH_START'].iloc[i]
        get_end = sample_day['SCH_END'].iloc[i]
        get_operating_room = sample_day['SCH_OR'].iloc[i]
        get_department = sample_day['Service_Line'].iloc[i]
        
        if get_operating_room not in show_rooms:
            continue

        get_line_color = line_color[get_department]
        line_color1 = dict(color=get_line_color)

        if get_department in departments_used:
            fig.add_trace(go.Scatter(x=[get_start, get_end],
                 y=[get_operating_room, get_operating_room],
                 name=get_department,
                 line=line_color1,
                showlegend=False))

        else:
            departments_used.append(get_department)
            fig.add_trace(go.Scatter(x=[get_start, get_end],
                y=[get_operating_room, get_operating_room],
                 name=get_department,
                 line=line_color1))
     
    #simulated end of day schedule viz
    #same technique as above but use IN/OUT Room Time and actual OR Used
    sample_day = sample_day.sort_values(by="OR_USED")
    
    point_plot1 = [
      go.Scatter(x=["2010-02-05 01:00:00", "2010-02-05 01:00:00"],
                 y=["Standby", "Standby"],
                 name="",
                 line=dict(color="black"))]

    make_title1 = "Simulated OR schedule"
    layout1 = go.Layout(title=go.layout.Title(text=make_title1,x=0.5),
            xaxis={'title':'Time','range':[start_range, end_range]},
            yaxis={'title':'OR'},
            height = 900)

    fig1 = go.Figure(data=point_plot1, layout=layout1)


    departments_used1 = []

    for i in range(0, len(sample_day)):
        get_start = sample_day['IN_ROOM_TIME'].iloc[i]
        get_end = sample_day['OUT_ROOM_TIME'].iloc[i]
        get_operating_room = sample_day['OR_USED'].iloc[i]
        get_department = sample_day['Service_Line'].iloc[i]
        
        if get_operating_room not in show_rooms:
            continue

        get_line_color = line_color[get_department]
        line_color1 = dict(color=get_line_color)

        if get_department in departments_used1:
            fig1.add_trace(go.Scatter(x=[get_start, get_end],
                 y=[get_operating_room, get_operating_room],
                 name=get_department,
                 line=line_color1,
                showlegend=False))

        else:
            departments_used1.append(get_department)
            fig1.add_trace(go.Scatter(x=[get_start, get_end],
                y=[get_operating_room, get_operating_room],
                 name=get_department,
                 line=line_color1))

    #update figure height based on number of OR rooms being shown
    if len(show_rooms) < 6:
        height_needed = 400
    else:
        height_needed = 400 + 14*(len(show_rooms)-6)

    #update the fig dimensions based on the above logic
    fig.update_layout(
    autosize=False,
    width=1000,
    height=height_needed,)
    
    fig1.update_layout(
    autosize=False,
    width=1000,
    height=height_needed,)


    #write the images to the current directory
    fig.write_image("planned_schedule.png")
    fig1.write_image("simulated_schedule.png")
    return fig, fig1