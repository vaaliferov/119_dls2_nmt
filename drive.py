'''
https://console.cloud.google.com/apis/dashboard
new project -> 'lfs-project'
https://console.cloud.google.com/apis/library
google drive api -> enable
https://console.cloud.google.com/apis/credentials/consent
external -> create, app name -> 'lfs'
user support email -> <your email>
email address -> <your email>
add or remove scopes -> 
'https://www.googleapis.com/auth/drive.file'
'https://www.googleapis.com/auth/drive.install'
test users -> add users -> <your email>
https://console.cloud.google.com/apis/credentials
create credentials -> oauth client id
application type -> web app -> name -> 'lfs-app'
authorized javascript origins -> http://localhost:8080
authorized redirect URIs -> http://localhost:8080/
client id, client secret -> settings.py
run this command to get your refresh_token:
get_refresh_token(client_id, client_secret)
refresh_token -> secret.py
'''

import os
import re
import json
import urllib
import socket
import requests

def get_refresh_token(client_id, client_secret):
    
    scopes = (
        'https://www.googleapis.com/auth/drive.file', 
        'https://www.googleapis.com/auth/drive.install'
    )
    params = {
        'client_id': client_id, 
        'scope': ' '.join(scopes), 
        'redirect_uri': 'http://localhost:8080/', 
        'access_type': 'offline', 'response_type': 'code', 
        'prompt': 'consent', 'include_granted_scopes': 'true', 
        'state': 'state_parameter_passthrough_value'
    }
    url = f'https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}'
    print(f'open this link in your browser and follow the instructions: {url}')
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('localhost', 8080)); sock.listen(1)
    conn, addr = sock.accept()
    data = conn.recv(2048).decode()
    code = re.findall('code=(.+)&', data)[0]
    conn.sendall(b'HTTP/1.0 200 OK\n\nOK')
    conn.close()
    
    data = {'client_id': client_id, 'client_secret': client_secret, 
            'code': code, 'grant_type': 'authorization_code', 
            'redirect_uri': 'http://localhost:8080/'}
    rsp = requests.post('https://oauth2.googleapis.com/token', data=data)
    if rsp.status_code == 200: return rsp.json()['refresh_token']


class GD:
    def __init__(self, client_id, client_secret, refresh_token):
        self.refresh_token = refresh_token
        self.client_secret = client_secret
        self.client_id = client_id

    def update_access_token(self):
        data = {'client_id': self.client_id, 'client_secret': self.client_secret, 
                'refresh_token': self.refresh_token, 'grant_type': 'refresh_token'}
        rsp = requests.post('https://oauth2.googleapis.com/token', data=data)
        if rsp.status_code == 200: self.access_token = rsp.json()['access_token']
    
    def upload_file(self, folder_id, file_path):
        with open(file_path, 'rb') as fd:
            headers = {'Authorization': f'Bearer {self.access_token}'}
            params = {'name': os.path.basename(file_path), 'parents': [folder_id]}
            data = ('metadata', json.dumps(params), 'application/json; charset=UTF-8')
            url = 'https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart'
            rsp = requests.post(url, headers=headers, files={'data': data, 'file': fd})
            if rsp.status_code == 200: return rsp.json()['id']