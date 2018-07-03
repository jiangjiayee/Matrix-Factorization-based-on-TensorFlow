user_set = set()
item_set = set()

max_user = 0
max_item = 0
with open('ratings.csv','r') as f:
	f.readline()
	for line in f.readlines():
		# print(line)
		# print(type(line))
		# break
		user,item,rating,timestamp = line.split(',')
		user_set.add(user)
		item_set.add(item)
		max_user = max(max_user,int(user))
		max_item = max(max_item,int(item))

print('User',len(user_set),'|',max_user)
print('item',len(item_set),'|',max_item)
# User 671 | 671
# item 9066 | 163949