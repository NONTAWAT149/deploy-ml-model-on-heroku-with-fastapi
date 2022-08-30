import requests

input_data = {'age': 76,
                 'workclass': 'Private',
                 'fnlgt': 124191,
                 'education': 'Masters',
                 'education-num': 14,
                 'marital-status': 'Married-civ-spouse',
                 'occupation': 'Exec-managerial',
                 'relationship': 'Husband',
                 'race': 'White',
                 'sex': 'Male',
                 'capital-gain': 0,
                 'capital-loss': 0,
                 'hours-per-week': 40,
                 'native-country': 'United-States'}

response = requests.post('https://pl-udacity-project3.herokuapp.com/prediction/', json=input_data)

print("status code: ", response.status_code)
print("response: ", response.json())