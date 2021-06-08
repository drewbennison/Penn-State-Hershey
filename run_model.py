#import the model code
import ORSim

print("Package successfully loaded...")

#create a class and generate a planned and simulated schedule
example_class = ORSim.HersheyORSim(selected_month = "Apr", selected_weekday = "Tue")
planned_schedule = example_class.planSchedule()
simulated_schedule = example_class.simulateSchedule(planned_schedule)
print(simulated_schedule)
#visualize the schedule
ORSim.visualizeSchedule(simulated_schedule)

#select a historical day and simulate that schedule
example_class_2 = ORSim.HersheyORSim()
planned_schedule_2 = example_class_2.selectRealSchedule("2019-04-18")
simulated_schedule_2 = example_class_2.simulateSchedule(planned_schedule_2)
#visualize the schedule
#ORSim.visualizeSchedule(simulated_schedule_2)

#simulate a day 10 times
for simulation in range(0,10):
	print("Running simulation", simulation)
	example_class_3 = ORSim.HersheyORSim(selected_month = "May", selected_weekday = "Tue")
	planned_schedule_3 = example_class_3.planSchedule()
	simulated_schedule_3 = example_class_3.simulateSchedule(planned_schedule_3)
	print("Here are the first few cases of simulated schedule", simulation)
	print(simulated_schedule_3.head(5))
