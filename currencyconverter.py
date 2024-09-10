#Step 1: Import libraries
import requests 

#Define API Key, URL, and Currencies covered
API_KEY = 'fca_live_id4xoE9bSKIfr4l7pQfe2BFhRs3VWdGpwdz4J44f'
BASE_URL = f'https://api.freecurrencyapi.com/v1/latest?apikey={API_KEY}'

CURRENCIES = ['USD', 'CAD', 'EUR', 'AUD', 'CNY']

#Step 2: Run program
#Define a function to check if the currency is in the list of valid currencies to be converted
def convert_currency(base):
    currencies = ','.join(CURRENCIES)
    url = f'{BASE_URL}&base_currency={base}&currencies={currencies}'
    try:
        response = requests.get(url)
        data = response.json()
        return data['data']
    except:
        print('Invalid currency.')
        return None
#Get input from user on what currency to convert
while True:
    base = input('Enter the base currency (q for quit): ').upper()

    if base == 'Q':
        break

    data = convert_currency(base)
    if not data:
        continue

    del data[base]
#Prints converted currency value
    for ticker, value in data.items():
        print(f'{ticker}: {value}')

