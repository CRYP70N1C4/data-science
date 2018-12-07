import requests
import os


def load_header():
    with open('headers.txt', 'r', encoding='utf-8') as f:
        headers = {}
        for line in f.readlines():
            k, v = line.split('	', 2)
            headers[k] = v.strip()
        return headers


headers = load_header()

with open('data.txt', 'r', encoding='utf-8') as f:
    paths = [x.split('/zjyprc-hadoop')[-1].strip() for x in f.readlines() if '/zjyprc-hadoop' in x]


def download_folder(path ,down=False):
    url = 'https://cloud.d.xiaomi.net/api/service/v1/hdfs/cnbj1-fusion/webhdfs/v1' + path
    params = {'op': 'LISTSTATUS', 'namespace': 'zjyprc-hadoop', 'user': '4594ffad8d5200a4ec0606242545ea0f',
              'doas': 'u_jianglie', 'pagesize': 100, 'pagenum': 1}
    resp = requests.get(url, params, headers=headers, verify=False).json()

    if down:
        return resp
    files = [x['pathSuffix'] for x in resp['FileStatuses']['FileStatus']]

    for file in reversed(files):
        download_file("%s/%s" % (path, file),
                      os.path.join('D:/code/xiaomi/dmcontest/2018', os.path.basename(path), file))


def download_file(hdfspath, dst):
    print(hdfspath)
    print(dst)
    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
    url = 'https://cloud.d.xiaomi.net/api/service/v1/hdfs/cnbj1-fusion/webhdfs/v1' + hdfspath
    params = {'op': 'OPEN', 'namespace': 'zjyprc-hadoop', 'doas': 'u_jianglie',
              'user': '4594ffad8d5200a4ec0606242545ea0f'}
    r = requests.get(url, params, headers=headers, stream=True, verify=False, timeout=20)
    with open(dst, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


# for folder in paths:
#     download_folder(folder)

path = '/user/h_mifi/user/mifi_compete/app_usage_samples/';
resp = download_folder(path,down=True)
files = [path+x['pathSuffix'] for x in resp['FileStatuses']['FileStatus']]
for f in files:
    download_folder(f)