# coding=utf-8
from bs4 import BeautifulSoup
import copy
import time
import json
import LOGIN
import MENU


class PlannedCourseInfo:
    def __init__(self, main_num=None, name=None, code=None, margin=None, detail=None, url=None, course_dic=None):
        if course_dic is None:
            self.num = str(main_num)
            self.name = str(name)
            self.code = str(code)
            self.margin = str(margin)
            self.url = url
            self.detail = copy.deepcopy(detail)
        else:
            self.num = course_dic["num"]
            self.name = course_dic["name"]
            self.code = course_dic["code"]
            self.margin = course_dic["margin"]
            self.url = course_dic["url"]
            self.detail = course_dic["detail"]

    def show_course_summary(self):
        print("主编号:" + self.num
              + "\t名称:" + self.name
              + "\t代码:" + self.code)

    def show_course_info(self):
        for item in self.detail:
            print("       ∟____ 辅编号:" + item["secondary_num"] + "\t教师:" + item["teacher"]
                  + "\t时间:" + item["time"])
            # print(self.code)

    def to_json(self):
        """
        将本类的数据转换为一个json,并返回字符串
        """
        js = {"name": self.name, "num": self.num, "code": self.code, "margin": self.margin, "url": self.url,
              "detail": self.detail}
        return json.dumps(js)


class PlannedCourse:
    """
    思路：
        1：登录
        2：进入选课界面
        3：抓取课程信息并保存
        4：用户输入想要抢的一门或几门课程
        5：开始抢课
    """

    def __init__(self, account):
        """初始化登录"""
        self.account = account
        self.english_course = []
        self.professional_course = []
        self.target = ""

    def init_menu(self):
        """输出菜单，并输入想要抢的课程"""
        menu_dic = {
            "-1": "更新数据（需要等待一分半左右）",
            "1": "本专业课程",
            "2": "大学英语扩展课",
            "0": "退出",
        }
        menu = MENU.MENU(menu_dic=menu_dic)
        menu.print_list()
        while True:
            _key = input(">>>")
            if int(_key) == 1:
                # 设置本专业课程target
                self.get_professional_course()
                print("输入课程编号选择课程，0返回")
                for item in self.professional_course:
                    item.show_course_summary()
                length = len(self.professional_course)
                while True:
                    i_key = input("(主编号)>>>")
                    if 0 < int(i_key) <= length:
                        print("你选择了", self.professional_course[int(i_key) - 1].name)
                        self.professional_course[int(i_key) - 1].show_course_info()
                        item_length = len(self.professional_course[int(i_key) - 1].detail)
                        while True:
                            j_key = input("(辅编号)>>>")
                            if 1 <= int(j_key) <= item_length:
                                detail = self.professional_course[int(i_key) - 1].detail[int(j_key) - 1]
                                print("你选择了: 辅编号:", detail["secondary_num"], "\t教师:", detail["teacher"],
                                      "\t时间:", detail["time"])
                                tmp = i_key + ":" + j_key
                                self.target = tmp
                                self.attack_professional()
                                return
                            elif int(j_key) == 0:
                                break
                            else:
                                print("请输入正确的数字")

                    elif int(i_key) == 0:
                        break
                    elif int(i_key) == -1:
                        self.update_course()
                    else:
                        print("请输入正确的数字")
            elif int(_key) == 2:
                # 设置英语扩展课课程target
                self.get_english_course()
                print("输入课程编号选择课程，0返回")
                for item in self.english_course:
                    item.show_course_summary()
                length = len(self.english_course)
                while True:
                    i_key = input("(主编号)>>>")
                    if 0 < int(i_key) <= length:
                        print("你选择了", self.english_course[int(i_key) - 1].name)
                        self.english_course[int(i_key) - 1].show_course_info()
                        item_length = len(self.english_course[int(i_key) - 1].detail)
                        while True:
                            j_key = input("(辅编号)>>>")
                            if 1 <= int(j_key) <= item_length:
                                detail = self.english_course[int(i_key) - 1].detail[int(j_key) - 1]
                                print("你选择了: 辅编号:", detail["secondary_num"], "\t教师:", detail["teacher"],
                                      "\t时间:", detail["time"])
                                tmp = i_key + ":" + j_key
                                self.target = tmp
                                self.attack_english()
                                return
                            elif int(j_key) == 0:
                                break
                            else:
                                print("请输入正确的数字")

                    elif int(i_key) == 0:
                        break
                    elif int(i_key) == -1:
                        self.update_course()
                    else:
                        print("请输入正确的数字")
            # elif int(_key) == 3:
            #     pass
            elif int(_key) == -1:
                self.update_course()
            elif int(_key) == 0:
                return
            else:
                print("请输入正确的数字")

    def __catch_view_state(self):
        """抓取 HTML中的 VIEWSTATE"""
        url = LOGIN.ZUCC.PlanCourageURL + "?xh=" + self.account.account_data[
            "username"] + "&xm=" + self.account.name + "&gnmkdm=N121101"
        header = LOGIN.ZUCC.InitHeader
        header["Referer"] = LOGIN.ZUCC.PlanCourageURL + "?xh=" + self.account.account_data["username"]
        response = self.account.session.get(url=url, headers=header)
        while response.status_code == 302:
            response = self.account.session.get(url=url, headers=header)
            time.sleep(0.2)
        self.account.soup = BeautifulSoup(response.text, "lxml")
        # print(response.status_code)

    def __enter_english_page(self):
        """进入计划内选课--英语页面，为抓取数据做准备"""
        self.__catch_view_state()
        url = LOGIN.ZUCC.PlanCourageURL + "?xh=" + self.account.account_data["username"]
        post_data = {"__EVENTTARGET": "", "__EVENTARGUMENT": "", "__LASTFOCUS": "", "__VIEWSTATEGENERATOR": "4842AF95",
                     "zymc": "0121%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%A7%91%E5%AD%A6%E4%B8%8E%E6%8A%80%E6%9C%AF%E4%B8%BB%E4%BF%AE%E4%B8%93%E4%B8%9A%7C%7C2019",
                     "xx": "", "Button3": "大学英语拓展课",
                     "__VIEWSTATE": self.account.soup.find(name='input', id="__VIEWSTATE")["value"]}
        response = self.account.session.post(url=url, data=post_data)
        self.account.soup = BeautifulSoup(response.text, "lxml")
        links = self.account.soup.find_all(name="tr")
        return links

    def __enter_professional_course(self):
        """进入计划内选课--本专业页面，为抓取数据做准备"""
        self.__catch_view_state()
        url = LOGIN.ZUCC.PlanCourageURL + "?xh=" + self.account.account_data["username"]
        post_data = {"__EVENTTARGET": "", "__EVENTARGUMENT": "", "__LASTFOCUS": "", "__VIEWSTATEGENERATOR": "4842AF95",
                     "xx": "", "Button5": "本专业选课",
                     "__VIEWSTATE": self.account.soup.find(name='input', id="__VIEWSTATE")["value"]}
        response = self.account.session.post(url=url, data=post_data)
        # print(response.text)
        self.account.soup = BeautifulSoup(response.text, "lxml")
        links = self.account.soup.find_all(name="tr")
        return links

    def get_english_course(self):
        """从文件中取得课程数据"""
        js_file = open("english_information.json", "r", encoding='utf-8')
        js_list = json.load(js_file)
        js_file.close()

        for course in js_list:
            tmp = PlannedCourseInfo(course_dic=course)
            self.english_course.append(tmp)

    def get_professional_course(self):
        """从文件中取得课程数据"""
        js_file = open("professional_information.json", "r", encoding='utf-8')
        js_list = json.load(js_file)
        js_file.close()

        for course in js_list:
            tmp = PlannedCourseInfo(course_dic=course)
            self.professional_course.append(tmp)

    def update_course(self):
        """更新课程信息并保存到文件"""
        links = self.__enter_english_page()
        course_list = []
        i = 1
        #  遍历10种英语课程
        for link in links[1:-1]:
            tmp = link.find_all("td")
            detail = []
            url = "http://" + LOGIN.ZUCC.DOMAIN + tmp[0].find(name="a")["onclick"][21:-8]
            header = LOGIN.ZUCC.InitHeader
            header["Referer"] = "https://jwxt.buu.edu.cn/xs_main.aspx?xh="+self.account.account_data['username']
            time.sleep(4)
            item_response = self.account.session.get(url=url, headers=header)
            item_soup = BeautifulSoup(item_response.text, "lxml")
            item_trs = item_soup.find_all(name="tr")
            j = 1
            print('.', end='')
            #  遍历所以的教学班
            for item_tr in item_trs[1:-1]:
                tds = item_tr.find_all("td")
                detail_td = {"secondary_num": str(j), "code": tds[0].find(name="input")["value"],
                             "teacher": tds[2].find(name="a").text,
                             "time": tds[3].text,
                             "margin": str(int(tds[11].text) - int(tds[13].text)) + "/" + tds[11].text}
                #  将教学班信息打包成列表
                detail.append(detail_td)
                j += 1
            tmp = link.find_all("td")
            course_list.append(
                PlannedCourseInfo(main_num=i, name=tmp[1].find(name="a").text, code=tmp[0].find(name="a").text,
                                  margin=tmp[9].text, detail=detail, url=url))
            i += 1

        js_str = "["
        flag = True
        for course in course_list:
            if flag:
                js_str += course.to_json()
                flag = False
            else:
                js_str += "," + course.to_json()
        js_str += "]"
        # 缓存在文件
        english_file = open("english_information.json", "w", encoding='utf-8')
        english_file.write(js_str)
        english_file.close()

        links = self.__enter_professional_course()
        course_list = []
        i = 1
        #  遍历专业课程
        for link in links[1:-1]:
            tmp = link.find_all("td")
            detail = []
            url = "http://" + LOGIN.ZUCC.DOMAIN + "/clsPage/xsxjs.aspx?" + "xkkh=" + \
                  tmp[0].find(name="a")["onclick"].split("=")[1][0:-3] + "&xh=" + self.account.account_data["username"]
            header = LOGIN.ZUCC.InitHeader
            header["Referer"] = "https://jwxt.buu.edu.cn/xs_main.aspx?xh=31901040"
            time.sleep(4)
            # print(url)
            item_response = self.account.session.get(url=url, headers=header)
            # print(item_response.text)
            item_soup = BeautifulSoup(item_response.text, "lxml")
            item_trs = item_soup.find_all(name="tr")
            j = 1
            print('.', end='')
            #  遍历所以的教学班
            for item_tr in item_trs[1:-1]:
                tds = item_tr.find_all("td")
                detail_td = {"secondary_num": str(j), "code": tds[0].find(name="input")["value"],
                             "teacher": tds[2].find(name="a").text,
                             "time": tds[3].text,
                             "margin": str(int(tds[11].text) - int(tds[13].text)) + "/" + tds[11].text}
                #  将教学班信息打包成列表
                detail.append(detail_td)
                j += 1
            tmp = link.find_all("td")
            course_list.append(
                PlannedCourseInfo(main_num=i, name=tmp[1].find(name="a").text, code=tmp[0].find(name="a").text,
                                  margin=tmp[9].text, detail=detail, url=url))
            i += 1
        js_str = "["
        flag = True
        for course in course_list:
            if flag:
                js_str += course.to_json()
                flag = False
            else:
                js_str += "," + course.to_json()
        js_str += "]"
        # 缓存在文件
        professional_file = open("professional_information.json", "w", encoding='utf-8')
        professional_file.write(js_str)
        professional_file.close()
        print("\n更新完成！")

    def attack_english(self):
        self.get_english_course()
        self.__enter_english_page()
        course_xy = self.target.split(":")
        x = int(course_xy[0])
        y = int(course_xy[1])
        header = LOGIN.ZUCC.InitHeader
        header["Referer"] = "https://jwxt.buu.edu.cn/xs_main.aspx?xh=31901040"
        response = self.account.session.get(url=self.english_course[x - 1].url, headers=header)
        # print(self.english_course[x - 1].url)
        self.account.soup = BeautifulSoup(response.text, "lxml")
        post_data = {"__EVENTTARGET": "Button1",
                     "__VIEWSTATEGENERATOR": "55DF6E88",
                     "RadioButtonList1": "1",
                     "xkkh": self.english_course[x - 1].detail[y - 1]["code"],
                     "__VIEWSTATE": self.account.soup.find_all(name='input', id="__VIEWSTATE")[0]["value"]}
        while True:
            response = self.account.session.post(url=self.english_course[x - 1].url, data=post_data)
            soup = BeautifulSoup(response.text, "lxml")
            try:
                reply = soup.find(name="script").text.split("'")[1]
            except BaseException:
                reply = "未知错误"
            print(reply+"\t\t"+str(time.strftime('%m-%d-%H-%M-%S',time.localtime(time.time()))))
            if reply == "选课成功！":
                return

    def attack_professional(self):
        self.get_professional_course()
        self.__enter_professional_course()
        course_xy = self.target.split(":")
        x = int(course_xy[0])
        y = int(course_xy[1])
        header = LOGIN.ZUCC.InitHeader
        header["Referer"] = "https://jwxt.buu.edu.cn/xs_main.aspx?xh=31901040"
        response = self.account.session.get(url=self.professional_course[x - 1].url, headers=header)
        # print(self.professional_course[x - 1].url)
        # print(response.text)
        self.account.soup = BeautifulSoup(response.text, "lxml")
        post_data = {"__EVENTTARGET": "Button1",
                     "__VIEWSTATEGENERATOR": "55DF6E88",
                     "RadioButtonList1": "1",
                     "xkkh": self.professional_course[x - 1].detail[y - 1]["code"],
                     "__VIEWSTATE": self.account.soup.find_all(name='input', id="__VIEWSTATE")[0]["value"]}

        while True:
            response = self.account.session.post(url=self.professional_course[x - 1].url, data=post_data)
            soup = BeautifulSoup(response.text, "lxml")
            try:
                reply = soup.find(name="script").text.split("'")[1]
            except BaseException:
                reply = "未知错误"
            print(reply)
            if reply == "选课成功！":
                return


if __name__ == "__main__":
    account = LOGIN.Account()
    account.login()
    planned_course_spider = PlannedCourse(account)
    # planned_course_spider.update_course()
    planned_course_spider.init_menu()
    # planned_course_spider.catch_english_course()
    # planned_course_spider.update_course()
