# -- codeing:utf-8 --
import requests
import json
import smtplib
import datetime
import logging
from email.message import EmailMessage

headers = {"User-Agent": "okhttp/3.10.0",
           "Content-Type": "application/json; charset=UTF-8",
           "Accept-Encoding": "gzip"}

important = False

logger = logging.getLogger('aqj')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('info.log', encoding="UTF-8")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


def get__request_boby(planId):
    return "{\"body\":{\"pageIndex\":0,\"pageSize\":10,\"planId\":%s,\"type\":1},\"comm\":{\"pid\":\"866817039838610\",\"spanId\":1,\"systemVersion\":\"8.1.0\",\"traceId\":\"7237c965-9222-44ab-b755-868d0d48fb12\",\"type\":3,\"us\":2604,\"version\":\"7.1.0\"},\"token\":\"6B245DBFAF82509D19ADDAE9F58E28A6A969CB0A984AA9342D4DCE64E0521A9E8FB980AEAA961B3F75A3E4537D63ED1E0BFDF652ABB35A652939AA3045F87B06G6EC7DC9CD9417036C3043C0C4769B62C\"}" % planId


def get_details(planId):
    r = requests.post("https://v2.iqianjin.com/C3000/M3053", data=get__request_boby(planId), headers=headers,
                      verify=False)
    return r.json()


def monitor(planId):
    result = dict()
    macthingAmount = 0.
    amount = 0.
    global important
    try:
        resp = get_details(planId);
        macthingAmount = resp['body']['macthingAmount']
        macthingAmount = to_float(macthingAmount)
        amount = resp['body']['list'][0]['amount']
        amount = to_float(amount)
    except Exception as ex:
        logging.error("monitor fail planId = %s ,ex = %s", planId, ex)
        result['exception'] = ex

        important = True
    result['planId'] = planId
    result['macthingAmount'] = macthingAmount
    result['amount'] = amount
    return result


def to_float(obj):
    obj = str(obj)
    obj = obj.replace(',', '')
    return float(obj)


def need_send():
    now = datetime.datetime.now()
    return now.minute >= 50 and (now.hour % 12) == 7;


def back(planId, amount):
    try:
        param = {"body": {"amount": amount, "planId": planId},
                 "comm": {"pid": "866817039838610", "spanId": 1, "systemVersion": "8.1.0",
                          "traceId": "523c2de8-a3d9-41db-b2ec-0f10ad7c2d98", "type": 3, "us": 2604, "version": "7.1.0"},
                 "token": "79A5A0F9EE654787C158A0BEE69D73D97201F943300A8CDB5675E80A826E759A8FB980AEAA961B3F75A3E4537D63ED1E0BFDF652ABB35A652939AA3045F87B06G4CF7BC92928D08360617355D07C7C761"}
        r = requests.post("https://v2.iqianjin.com/C3000/M3056", data=json.dumps(param), headers=headers,
                          verify=False)
        return False
    except Exception  as ex:
        logging.error("monitor fail planId = %s ,amount = %s ,ex = %s", planId, amount, ex)
        return True


def monitor_all():
    plans = ["6393778", "6090912", "6368514"]
    msgs = []
    imp = False
    for planId in plans:
        tmp = monitor(planId)
        if tmp['macthingAmount'] > 0:
            back(planId, tmp['macthingAmount'])
            imp = True
        msgs.append(json.dumps(tmp, ensure_ascii=False))

    title = "aqj 无变化"
    if imp:
        title = "** aqj Important ** "

    content = '\n'.join(msgs)

    to_send = need_send()
    logger.info("imp = %s,to_seed = %s,title = %s,content = %s", imp, to_send, title, content)

    if imp or to_send:
        send_mail(title, content)


def send_mail(title, content):
    msg = EmailMessage()
    msg['Subject'] = title
    msg.set_content(content)
    msg['From'] = 'charlie9527l@sina.cn'
    msg['To'] = '254429775@qq.com'
    s = smtplib.SMTP()
    s.connect('smtp.sina.cn', 587)
    s.login('charlie9527l@sina.cn', '163tCrxyOzqtVHt')
    s.send_message(msg)
    s.close()


if __name__ == '__main__':
    monitor_all()