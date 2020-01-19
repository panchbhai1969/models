import csv

def dataset_csv_to_example_list(dataset_path):
  """
  Function to parse dataset.csv file into a list of list of values in a row.
  """
  examples_list = []
  i=0
  with open(dataset_path) as f:
    csv_f = csv.reader(f)
    
    for row in csv_f:
      if i==0: # skipping first row
        i = 1 
        continue
      examples_list.append(row)
  return examples_list