import datetime

x = datetime.datetime.now()
x_new = x.strptime("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S")

print(x_new) 