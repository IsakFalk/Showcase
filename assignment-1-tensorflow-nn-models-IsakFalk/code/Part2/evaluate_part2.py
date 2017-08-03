# Evaluate the error of the saved models of part 2

import part2lib as p2l

data = p2l.Data(1)

print "=== Part 2 ==="

# model 2a

print "\nModel 2a (P2:b)\n"

rel_path = './../../save/np/model_2a'
model_2a = p2l.load_network(rel_path)
print "Train error: {:.5f}".format(1 - model_2a.accuracy('train', data))
print "Test error: {:.5f}".format(1 - model_2a.accuracy('test', data))

# model 2b

print "\nModel 2b (P2:c)\n"

rel_path = './../../save/np/model_2b'
model_2b = p2l.load_network(rel_path)
print "Train error: {:.5f}".format(1 - model_2b.accuracy('train', data))
print "Test error: {:.5f}".format(1 - model_2b.accuracy('test', data))

# model 2c

print "\nModel 2c (P2:d)\n"

rel_path = './../../save/np/model_2c'
model_2c = p2l.load_network(rel_path)
print "Train error: {:.5f}".format(1 - model_2c.accuracy('train', data))
print "Test error: {:.5f}".format(1 - model_2c.accuracy('test', data))

# model 2d

print "\nModel 2d (P2:e)\n Not implemented"

# rel_path = './../../save/np/model_2d'
# model_2d = p2l.load_network(rel_path)
# print model_2a.accuracy('train')
# print model_2a.accuracy('test')
