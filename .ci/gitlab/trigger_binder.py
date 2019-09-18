import requests
import sys

url = sys.argv[1]
headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:53.0) Gecko/20100101 Firefox/53.0'}

try:
    response = requests.get(url, allow_redirects=True, timeout=10., headers=headers, stream=True)
    for _ in response.iter_content():
        # we don't care what the reposne is, unless it's an http error
        pass
except requests.exceptions.Timeout as e:
    # The timeout's triggered if nothing is received every x seconds
    print('timeout. assuming normal trigger')
    sys.exit(0)
except requests.exceptions.ConnectionError as e:
    print('Connect failed')
    print(e)
    sys.exit(-2)
if not response.ok:
    print(f'trigger {url} not successful: {response.status_code}')
    sys.exit(-1)
print('Trigger success')