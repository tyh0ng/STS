import csv

csv_file = open('results-spanish.csv', 'r', encoding='utf-8-sig')
read_csv = csv.reader(csv_file)
all_scores = []
for row in read_csv:
	for value in row:
		all_scores.append([float(value)])

    
# name of csv file  
filename = "final_spanish_results.csv"
    
# writing to csv file  
with open(filename, 'w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
        
    csvwriter.writerows(all_scores) 
