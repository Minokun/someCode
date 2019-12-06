from pywinauto import Desktop, Application

app = Application().connect(path='C:\Program Files (x86)\Tencent\WeChat\WeChat.exe')
app.top_window()
