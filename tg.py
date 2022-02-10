import requests

class TG:
    def __init__(self, token, chat_id, local):
        self.local = local
        self.token = token
        self.chat_id = chat_id
        l = 'http://127.0.0.1:8081'
        r = 'https://api.telegram.org'
        self.base_url = l if local else r
    
    def send_message(self, text):
        url = f'{self.base_url}/bot{self.token}/sendMessage'
        rsp = requests.get(url, {'chat_id': self.chat_id, 'text': text})
        if rsp.status_code == 200 and rsp.json()['ok']:
            return rsp.json()['result']['message_id']
    
    def send_photo(self, path, caption=''):
        url = f'{self.base_url}/bot{self.token}/sendPhoto'
        data = {'chat_id': self.chat_id, 'caption': caption}
        if self.local == False:
            files = {'photo': open(path, 'rb')}
            rsp = requests.post(url, data, files=files)
        else:
            data['photo'] = f'file://{path}'
            rsp = requests.get(url, data)
        if rsp.status_code == 200 and rsp.json()['ok']:
            return rsp.json()['result']['photo'][-1]['file_id']
    
    def send_document(self, path, caption=''):
        url = f'{self.base_url}/bot{self.token}/sendDocument'
        data = {'chat_id': self.chat_id, 'caption': caption}
        if self.local == False:
            files = {'document': open(path, 'rb')}
            rsp = requests.post(url, data, files=files)
        else:
            data['document'] = f'file://{path}'
            rsp = requests.get(url, data)
        if rsp.status_code == 200 and rsp.json()['ok']:
            return rsp.json()['result']['document']['file_id']

    def get_file(self, file_id):
        url = f'{self.base_url}/bot{self.token}/getFile'
        rsp = requests.get(url, {'file_id': file_id})
        if rsp.status_code == 200 and rsp.json()['ok']:
            return rsp.json()['result']['file_path']

    '''
    def download_file(self, remote_file_path, local_file_path):
        url = f'{self.base_url}/file/bot{self.token}/{remote_file_path}'
        rsp = requests.get(url)
        if rsp.status_code == 200:
            with open(local_file_path, 'wb') as fd:
                fd.write(rsp.content)
    '''