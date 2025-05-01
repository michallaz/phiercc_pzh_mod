import requests
import subprocess
import sys
import time

SPEC="pubmlst_campylobacter_seqdef"
DATABASE='C. jejuni / C. coli cgMLST v2'

# find scheme link
scheme_link = ''
scheme_table = requests.get('https://rest.pubmlst.org/db/' f'{SPEC}' + '/schemes')
for scheme in scheme_table.json()['schemes']:
    if DATABASE == scheme['description']:
        scheme_link =  scheme['scheme']

print(f'Downloading profiles')
profile = requests.get(scheme_link + '/profiles_csv')
with open('profiles.list','w') as f:
    i = 0
    for line in profile.iter_lines():

        # Last column is LINcode that should not be included
        if i == 0:
            # Firs line (header) has an additional element
            line = list(map(lambda x: x.decode('utf-8', errors='replace'), line.split()))[:-1]
        else:
            line = list(map(lambda x: x.decode('utf-8', errors='replace'), line.split()))[:-1]
        # replace "Ns" with 0
        line = ["0" if x == "N" else x for x in line]
        line = "\t".join(line)
        f.write(line + "\n")
        i = 1


