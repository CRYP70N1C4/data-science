import adb_helper, time


def check():
    path = "{}.png".format(int(time.time()))
    adb_helper.screenshots().save(path)


def go_back(n=1):
    for _ in range(n):
        adb_helper.click(1830, 40, msg='后退')


def do_usual():
    adb_helper.click(460, 540, msg='进入日常')
    do_usual_1()
    do_usual_2()
    do_usual_3()
    go_back()


def do_usual_1(n=5):
    adb_helper.click(260, 570, msg='进入一命通关')
    adb_helper.click(1670, 1040, msg='开战')
    for i in range(n):
        adb_helper.click(1540, 870, msg='扫荡-地狱')
        adb_helper.click(1000, 660, msg='tap')
        adb_helper.click(946, 984, msg='确认')
    adb_helper.click(1680, 170, msg='close')

    go_back()


def do_usual_2(n=2):
    adb_helper.click(720, 600, msg='进入活动模式')
    for x, y in [(137, 160), (137, 270), (137, 360)]:
        adb_helper.click(x, y, msg="挑战")
        adb_helper.click(1730, 1030, msg="开始准备")
        for _ in range(n):
            adb_helper.click(1510, 880, msg="扫荡-苦痛1")
            adb_helper.click(1000, 660, msg='tap')
            adb_helper.click(946, 984, msg='确认')
        adb_helper.click(1680, 170, msg='close')
    go_back()


def do_usual_3(n=1):
    adb_helper.click(1150, 570, msg='军械大师')

    for i in range(n):
        adb_helper.click(700, 800, msg='取消')
        adb_helper.click(1720, 1030, msg='开战')
        adb_helper.click(1615, 760, msg='更改难度')
        adb_helper.click(1510, 310, msg='一星')
        adb_helper.click(1635, 974, msg='开战')

        time.sleep(20)
        adb_helper.click(474, 62, msg='自动开火')
        time.sleep(250)
        adb_helper.click(1200, 920, msg='确认')
        adb_helper.click(1000, 660, msg='tap')
        time.sleep(10)
    go_back()


def do_challenge():
    adb_helper.click(1440, 500, msg='进入挑战')

    # adb_helper.click(410, 710, msg='进入无尽模式')
    # adb_helper.click(1424,990,msg="扫荡")
    # adb_helper.click(1340,525,msg="普通扫荡")
    #
    # go_back()

    do_challenge_2(7)

    go_back()


def do_challenge_2(n=7):
    adb_helper.click(993, 730, msg='进入突破加鲁加')
    for _ in range(n):
        adb_helper.click(1680, 1000, msg='继续挑战')
        adb_helper.click(1383, 1018, msg='扫荡')
        adb_helper.click(1614, 786, msg='选择武器')
        adb_helper.click(1504, 923, msg='扫荡')
        adb_helper.click(920, 943, msg='确认')
        adb_helper.click(932, 521, msg='领取奖励')
        adb_helper.click(1174, 906, msg='离开')
        go_back()
    go_back()


def start():
    # do_usual()
    do_challenge()


if __name__ == '__main__':
    start()