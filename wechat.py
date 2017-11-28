from drlchat.drlchatter import DRLChatter
from drlchat.msgregister import MsgRegister


chat = DRLChatter(["睡不醒", "唐春旭"], ["唐春旭"], './info.txt')
MsgRegister.handler = chat.handle_msg
chat.start()
