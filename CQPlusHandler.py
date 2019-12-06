import cqplus
import os
from datetime import datetime
import re
import json
from urllib import parse
import pyttsx3
'''
用户信息：
age
nickname
sex
user_id

群组字段有：
from_group
from_qq
msg
msg_id

群用户的信息字段有
age
area
card
group_id
join_time
last_send_time
level
nickname
role
sex 1 女 255 男 0 未知
'''

def getImgUrl(HOME_PATH, msg):
    # 获取图片地址
    msg_list = msg.split('file=')
    if len(msg_list) == 2:
        end_index = msg_list[1].index(']')
        img_file = msg_list[1][:end_index]
        img_url = False
        if img_file.split('.')[1] in ['jpg', 'png', 'gif']:
            file_path = os.path.join(HOME_PATH, 'data', 'image', img_file + '.cqimg')
            img_url = '图片地址：'
            with open(file_path, 'r') as f:
                while True:
                    line_text = f.readline()
                    if line_text[:4] == 'url=':
                        img_url += line_text[5:]
                        break
                    if line_text == '' or line_text is None:
                        break
    else:
        img_url = False
    return img_url

def unixtimeToDatetime(utime):
    return datetime.fromtimestamp(utime).strftime("%Y-%m-%d %H:%M:%S")

def getVoiceUrl(msg):
    url = "https://openapi.data-baker.com/tts?access_token=b2e4e4ee-5171-454b-beeb-ff80b0867cf0&domain=1&speed=5&volume=5&language=zh&voice_name=%E6%99%BA%E8%83%BD%E5%AE%A2%E6%9C%8D_%E6%9E%9C%E6%9E%9C&text="
    return url + parse.quote(msg)

class MainHandler(cqplus.CQPlusHandler):
    def handle_event(self, event, params):
        HOME_PATH = params['env']['CQ_HOME']
        # # 群聊信息
        # if event == 'on_group_msg':
        #     user_info = self._api.get_group_member_info(params['from_group'], params['from_qq'], False)
        #     # 此处是获取用户信息的代码
        #     for key in user_info:
        #         self.logging.debug(key + "  " + str(user_info[key]))
        #     self.logging.debug(params['msg'])
        #     # 监听的特定qq群列表
        #     group_id_list = [633706028]
        #     # 需要转发的目标群
        #     target_group = 553018228
        #     # 如果接收到的群在监听群组中
        #     if params['from_group'] in group_id_list:
        #         # 获取群消息
        #         msg = params['msg']
        #         # 获取发送消息人的qq和昵称
        #         nickname = user_info['nickname']
        #         from_qq = params['from_qq']
        #         # 组织要转发的话
        #         send_msg = "{}({})：{}".format(nickname, from_qq, msg)
        #         self.logging.debug(send_msg)
        #         cqplus._api.send_group_msg(params["env"], target_group, send_msg)
        # 私聊信息
        if event == 'on_private_msg':
            # self.logging.debug(str(params["msg"]))
            # img_url = getImgUrl(HOME_PATH, params['msg'])

            if params['from_qq'] in [579332, 2387168191]:
                self.logging.debug(params['from_qq'])

                cqplus._api.send_private_msg(params["env"], params["from_qq"], getVoiceUrl(params['msg']))
                # engine = pyttsx3.init()
                # engine.setProperty('rate', 180)
                # engine.say(params['msg'])
                # engine.runAndWait()
            # if img_url:
            #     # 如果有图片
            #     if params['from_qq'] in [952718180]:
            #         mes_obj = cqplus._message.MessageBuilder()
            #         # mes_obj.append_record('D:\\b.mp3')
            #         # reback_msg = json.dumps(mes_obj.items)
            #         cqplus._api.send_private_msg(params["env"], params["from_qq"], params['msg'])
            # else:
            #     # 只是文本
            #     self.logging.debug(params['msg'])
            #     # 时间戳转时间 匹配出10位时间戳
            #     unix_obj = re.search(r'[\d]{10}', params['msg'])
            #     unix_time = unixtimeToDatetime(int(unix_obj[0])) if unix_obj else False
            #     for key in params:
            #         self.logging.debug(key, params[key])
            #     if params['from_qq'] in [952718180]:
            #         if unix_time:
            #             cqplus._api.send_private_msg(params["env"], params["from_qq"], unix_time)
            #         cqplus._api.send_private_msg(params["env"], params["from_qq"], params['msg'])
        # 群聊信息
        # elif event == 'on_group_msg':
        #     user_info = self._api.get_group_member_info(params['from_group'], params['from_qq'], False)
        #     for key in user_info:
        #         self.logging.debug(key + "  " + str(user_info[key]))
        #     self.logging.debug(params['msg'])
        #     # 特定群发图片 则转发图片链接
        #     group_id_list = [553018228]
        #     img_url = getImgUrl(HOME_PATH, params['msg'])
        #     self.logging.debug(img_url)
        #     if img_url:
        #         if params['from_group'] in group_id_list:
        #             cqplus._api.send_group_msg(params["env"], params['from_group'], img_url)


