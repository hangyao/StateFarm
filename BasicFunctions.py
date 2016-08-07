

friends = ['john', 'pat', 'gary', 'michael']
for i, name in enumerate(friends):
    print "name %i is %s" % (i, name)

# How many friends contain the letter 'a' ?
count_a = 0
for name in friends:
    if 'a' in name:
        count_a += 1

print "%.2f percent of the names contain an 'a'" % ( float(count_a) / len(friends) * 100 )


# Say hi to all friends
def print_hi(name, greeting='hello'):
    print "%s %s" % (greeting, name)

map(print_hi, friends)

# Print sorted names out
friends.sort()
print friends

'''
    Calculate the factorial N! = N * (N-1) * (N-2) * ...
'''

def factorial(x):
    """
    Calculate factorial of number
    :param N: Number to use
    :return: x!
    """
    if x==1: return 1
    return factorial(x-1) * x

print "The value of 5! is", factorial(5)
