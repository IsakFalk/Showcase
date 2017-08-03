#A3
# EPOCHS=500
# BATCHES=16
# python A3.py -n A3linear -l 0.5 -es linear -b $BATCHES -e $EPOCHS
# python A3.py -n A3linear -l 0.1 -es linear -b $BATCHES -e $EPOCHS
# python A3.py -n A3linear -l 0.01 -es linear -b $BATCHES -e $EPOCHS
# python A3.py -n A3linear -l 0.001 -es linear -b $BATCHES -e $EPOCHS
# python A3.py -n A3linear -l 0.0001 -es linear -b $BATCHES -e $EPOCHS
# python A3.py -n A3linear -l 0.00001 -es linear -b $BATCHES -e $EPOCHS
# python A3.py -n A3NNet -l 0.5 -es NNet -b $BATCHES -e $EPOCHS
# python A3.py -n A3NNet -l 0.1 -es NNet -b $BATCHES -e $EPOCHS
# python A3.py -n A3NNet -l 0.01 -es NNet -b $BATCHES -e $EPOCHS
# python A3.py -n A3NNet -l 0.001 -es NNet -b $BATCHES -e $EPOCHS
# python A3.py -n A3NNet -l 0.0001 -es NNet -b $BATCHES -e $EPOCHS
# python A3.py -n A3NNet -l 0.00001 -es NNet -b $BATCHES -e $EPOCHS

# A4
#python A45.py -n A4 -l 0.001 -u 100 -e 2000 -i 20 -r 100

# A5
#python A45.py -n A5a -l 0.00001 -u 30 -e 2000 -i 20 -r 1
#python A45.py -n A5b -l 0.00001 -u 1000 -e 2000 -i 20 -r 1

#BATCH_SIZE=128
#BUFFER_SIZE=100000

# A6
#python A6.py -n A6 -l 0.00001 -u 100 -b $BUFFER_SIZE -bs $BATCH_SIZE -r 1

# A7
#python A7.py -n A6 -l 0.00001 -u 100 -b $BUFFER_SIZE -bs $BATCH_SIZE -r 1

# A8
#python A8.py -n A8 -l 0.00001 -u 100 -e 2000 -i 20 -r 1

# B1
#python B1.py -g pong -e 100
#python B1.py -g pacman -e 100
#python B1.py -g boxing -e 100

# B2
#python B2.py -g pong -e 100
#python B2.py -g pacman -e 100
#python B2.py -g boxing -e 100

# B3
#python B3.py -g pong -n pong -l 0.001 -v 50000 -e 1000000
#python B3.py -g pacman -n pacman -l 0.001 -v 50000 -e 1000000
#python B3.py -g boxing -n boxing -l 0.001 -v 50000 -e 1000000

# B4
python B4.py -g pong -n pong -e 100
python B4.py -g pacman -n pacman -e 100
python B4.py -g boxing -n boxing -e 100
