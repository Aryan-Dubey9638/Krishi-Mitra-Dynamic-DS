import requests
import pandas as pd
from bs4 import BeautifulSoup
import io

def fetch_agmarknet():
    url = "https://agmarknet.gov.in/SearchCmmMkt.aspx"
    print(f"Fetching from {url}...")
    # Initial GET to get ViewState
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    
    r = session.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    
    viewstate = soup.find('input', {'id': '__VIEWSTATE'})['value']
    viewstategenerator = soup.find('input', {'id': '__VIEWSTATEGENERATOR'})['value']
    eventvalidation = soup.find('input', {'id': '__EVENTVALIDATION'})['value']
    
    # Example payload to search for Tomato across all states
    payload = {
        '__EVENTTARGET': 'ctl00$cphBody$Button1',
        '__EVENTARGUMENT': '',
        '__VIEWSTATE': viewstate,
        '__VIEWSTATEGENERATOR': viewstategenerator,
        '__EVENTVALIDATION': eventvalidation,
        'ctl00$cphBody$cboYear': '2024',
        'ctl00$cphBody$cboMonth': 'September',
        'ctl00$cphBody$cboState': 'MH', # Maharashtra
        'ctl00$cphBody$cboCommodity': '78', # Tomato
        'ctl00$cphBody$Button1': 'Submit'
    }
    
    r2 = session.post(url, data=payload)
    print("Post status:", r2.status_code)
    
    tables = pd.read_html(r2.text)
    for i, t in enumerate(tables):
        if len(t) > 5:
            print(t.head())
            t.to_csv("test_agmarknet.csv", index=False)
            break

if __name__ == "__main__":
    fetch_agmarknet()
