import requests
from bs4 import BeautifulSoup


def dict_generate():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.75 Safari/537.36"}
    url = 'https://www.babycenter.com/babyNamerSearch.htm?gender=UNKNOWN&batchSize=10000&includeExclude=ALL'
    resp = requests.get(url, headers=headers, verify=False)
    soup = BeautifulSoup(resp.content, 'lxml')
    with open('name_gender_info.txt', 'w', encoding='utf-8') as f:
        for tr in soup.select('tr'):
            name = tr.select('.nameCell')[0].text
            name = name.strip()
            gender = 'M'
            if 'bgGenderIconF' in str(tr.select('.genderCell')[0]):
                gender = 'F';

            f.write('%s,%s\n' % (name, gender))


def sql_gen():
    index = 3
    new_m = []
    new_f = []
    orig_m = []
    orig_f = []
    with open('name_gender_info.txt', 'r', encoding='utf-8') as f:
        for info in f.readlines():
            if info.strip().endswith('F'):
                new_f.append(info)
            else:
                new_m.append(info)

    new_m = new_m[index * 1000:(index + 1) * 1000]
    new_f = new_f[index * 1000:(index + 1) * 1000]

    with open('a%s.log' % index, 'r', encoding='utf-8') as f:
        for info in f.readlines():
            if info.strip().endswith('1'):
                orig_m.append(info)
            else:
                orig_f.append(info)

    sql_template = "mysql_sgp_xgame_" + str(index) + " -e \"update user_info set nickname = '%s' where user_id = %s and is_robot = 1\";"
    print("source ~/.bashrc")
    for i in range(len(orig_m)):
        nick_name = new_m[i].split(',')[0]
        user_id = orig_m[i].split('\t')[0]
        print(sql_template % (nick_name, user_id))

    for i in range(len(orig_f)):
        nick_name = new_f[i].split(',')[0]
        user_id = orig_f[i].split('\t')[0]
        print(sql_template % (nick_name, user_id))


sql_gen()
