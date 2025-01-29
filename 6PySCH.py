#Program:   TSCH Node Synchronization
#Inputs:    DATA_STORE_LOCATION/Realdata/_realdata_.txt
#Outputs:   DATA_STORE_LOCATION/Statistics/*.txt
#Author:    Surja Sanyal
#Email:     hi.surja06@gmail.com
#Date:      03 DEC 2024
#Comments:  1. Please create a folder named "Statistics" in the location DATA_STORE_LOCATION. Outputs will be saved there.




##   Start of Code   ##


#   Imports    #

import os
# import re
import sys
import glob
import math
import copy
import time
# import json
# import psutil
# import shutil
import random
import datetime
import platform
import traceback
# import itertools
import numpy as np
import multiprocessing
# from textwrap import wrap
# from functools import partial
# import matplotlib.pyplot as plt
# from scipy.stats import truncnorm




##  Global environment   ##

#   Customize here  #
SEED                    = 2                                             # Random seed
SIMS                    = 5                                             # Simulations per setup
EB_PROB                 = 0.1                                           # EB probability limit
COMM_RANGE              = 50                                            # Signal distance
SLOTFRAME_SIZES         = [23, 41, 61, 83, 101]                         # Slots
EB_INTERVALS            = [1, 2, 3, 4]                                  # Times slotframe
NODES                   = list(range(200, 600 + 1, 100))                # Total nodes
AREAS                   = list(range(100, 300 + 1, COMM_RANGE))         # Area range
CHANNELS                = list(range(4, 16 + 1, 4))                     # Channels
ATP_LEVELS              = list(range(2, 8 + 1, 2))                      # ATP level
MECHANISMS              = ['P', 'E', 'R', 'D', 'A']                     # Mechanisms
TIMEOUT                 = 3600                                          # Seconds (= 1 hour)



#   Do not change   #
REALDATA                = ["Realdata", "_realdata_.txt"]
DATA_LOAD_LOCATION      = os.path.dirname(sys.argv[0])    #Local data load location
DATA_STORE_LOCATION     = os.path.dirname(sys.argv[0])    #Local data store location




##  Function definitions    ##


#   Print with lock    #
def print_locked(*content, sep=" ", end="\n"):

    store = os.path.dirname(sys.argv[0])

    #print (*content, sep = sep, end = end)

    print (*content, sep = sep, end = end, file=open(os.path.join(store, "Log_Files", "_Log_" + str(sys.argv[0].split("\\")[-1].split('.')[0]) + ".txt"), 'a'))



# Node struct #
class node:
    def __init__(self, area, realdata = [], node_id = 0, height = 0, start_channel = -1, start_asn = -1, chs = []):
        self.iden = node_id
        self.parent = -1
        self.height = height
        self.x_axis = int(random.choice(realdata) % area) if (node_id > 0) else area // 2
        self.y_axis = int(random.choice(realdata) % area) if (node_id > 0) else area // 2
        self.start_channel = start_channel if (node_id > 0) else 1
        self.start_asn = start_asn if (node_id > 0) else 0
        self.nbr_available_asn = -1
        self.joined_asn = -1
        self.joining_dur = -1
        self.joined = False if (node_id > 0) else True
        self.collisions = 0
        self.no_eb = 0
        self.last_eb = -1
        self.last_eb_freq = []
        self.chs = chs
        self.nbrs = []
        self.children = []

    def join(self, parent, height, asn, ch, slot, chs):
        self.joined = True
        self.parent = parent
        self.height = height
        self.start_channel = ch
        self.start_asn = slot
        self.joined_asn = asn
        self.joining_dur = asn - self.nbr_available_asn + 1 if (self.joined_asn < 0) else self.joined_asn
        self.chs = chs

    def set_last_eb_details(self, sf_number, freq):
        self.last_eb = sf_number
        self.last_eb_freq = freq

    def info(self):
        return self.parent, self.height

    def hops(self):
        return self.height

    def identity(self):
        return self.iden

    def x_coord(self):
        return self.x_axis

    def y_coord(self):
        return self.y_axis

    def details(self):
        #return self.id, self.joined, self.parent, self.height, self.start_channel, self.start_asn, self.x_axis, self.y_axis, self.chs
        print_locked("Id:", self.iden, ", DODAG:", self.joined, ", Parent Id:", self.parent, ", Height:", self.height, \
                     ", Start channel:", self.start_channel, \
                     ", Start time:", self.start_asn, ", Nbr avl:", self.nbr_available_asn, ", Joining time:", self.joined_asn, \
                     ", X-coord:", self.x_axis, ", Y-coord:", self.y_axis, ", Nbrs:", len(self.nbrs), ", CHS:", True if (len(self.chs) > 0) else False)




# Distance between nodes #
def get_distance(pledge, joined_node):

    return math.sqrt((pledge.x_coord() - joined_node.x_coord()) ** 2 + (pledge.y_coord() - joined_node.y_coord()) ** 2)




# Generate beacons #
def generate_beacons(j_nodes, sf_size, eb_interval, asn, chs, r_chs, n_channels, scheme, eb_prob_limit, atp_level):
    
    eb_prob = eb_prob_limit
    sf_number = asn // sf_size

    if (scheme == 'P'):
        eb_prob = eb_prob_limit if (math.exp(-1 * (sf_number) / n_channels) < eb_prob_limit) else math.exp(-1 * (sf_number) / n_channels)
        #print_locked(asn / sf_size, math.exp(-1 * (sf_number) / n_channels))
        #ch_dict = {joined_node : chs[(asn + joined_node.start_channel) % n_channels] for joined_node in j_nodes if (random.random() <= (eb_prob_limit if (math.exp(-1 * (len(joined_node.children) ** 2) / n_channels) < eb_prob_limit) else math.exp(-1 * (len(joined_node.children) ** 2) / n_channels)) and (joined_node.last_eb < 0 or sf_number >= joined_node.last_eb + eb_interval))}
        ch_dict = {joined_node : chs[(asn + joined_node.start_channel) % n_channels] for joined_node in j_nodes if (random.random() <= eb_prob and (joined_node.last_eb < 0 or sf_number >= joined_node.last_eb + eb_interval))}
        ch = list(ch_dict.values())
        jn = list(ch_dict.keys())
        [j.set_last_eb_details(sf_number, [c]) for c, j in zip(ch, jn)]
    elif (scheme == 'E'):
        ch_dict = {joined_node : chs[asn % n_channels] for joined_node in j_nodes if (random.random() <= eb_prob and (joined_node.last_eb < 0 or sf_number >= joined_node.last_eb + eb_interval))}
        ch = list(ch_dict.values())
        jn = list(ch_dict.keys())
        [j.set_last_eb_details(sf_number, [c]) for c, j in zip(ch, jn)]
    elif (scheme == 'R'):
        ch_dict = {joined_node : random.choice(r_chs) for joined_node in j_nodes if (joined_node.last_eb < 0 or sf_number >= joined_node.last_eb + eb_interval)}
        ch = list(ch_dict.values())
        jn = list(ch_dict.keys())
        [j.set_last_eb_details(sf_number, [c]) for c, j in zip(ch, jn)]
    elif (scheme == 'D'):
        ch_dict = {joined_node : chs[(asn + joined_node.start_channel) % n_channels] for joined_node in j_nodes if (joined_node.last_eb < 0 or sf_number >= joined_node.last_eb + eb_interval)}
        ch = list(ch_dict.values())
        jn = list(ch_dict.keys())
        [j.set_last_eb_details(sf_number, [c]) for c, j in zip(ch, jn)]
    elif (scheme == 'A'):
        ch_dict = {joined_node : [chs[(asn + (channel_offset + 1)) % n_channels] for channel_offset in range(atp_level)] for joined_node in j_nodes if (random.random() <= eb_prob and (joined_node.last_eb < 0 or sf_number >= joined_node.last_eb + eb_interval))}
        ch = [c for ch_j in list(ch_dict.values()) for c in ch_j]
        jn = list(ch_dict.keys())
        [j.set_last_eb_details(sf_number, c) for c, j in zip(ch_dict.values(), jn)]
    
    return ch_dict



# Join one pledge #
def join_node(pledge, j_nodes, sf_size, asn, chs, r_chs, n_channels, scheme, assigned, collisions, ch_dict):
    
    sf_number = asn // sf_size
    r_ch = random.choice(r_chs)
    
    ch, jn = [c for j, c in ch_dict.items() if (j in j_nodes)], [j for j in ch_dict.keys() if (j in j_nodes)]
    
    if (scheme == 'A'):
        ch = [c for ch_j in ch for c in ch_j]
    
    #print_locked(sf_number, [(joined_node.identity(), joined_node.last_eb, freq) for joined_node, freq in ch_dict.items()], r_ch, pledge.identity())

    if (r_ch in ch and sum([1 for c in ch if c == r_ch]) == 1):

        if (scheme == 'P'):
            joined_node = [j for j, c in ch_dict.items() if c == r_ch][0]
            if (joined_node.hops() + 1 < pledge.hops()):
                pledge.join(joined_node.identity(), joined_node.hops() + 1, asn, r_ch, asn % n_channels, chs)
                joined_node.children += [pledge.identity()]
                return [collisions, assigned, True]
        elif (scheme == 'E'):
            joined_node = [j for j, c in ch_dict.items() if c == r_ch][0]
            if (joined_node.hops() + 1 < pledge.hops()):
                pledge.join(joined_node.identity(), joined_node.hops() + 1, asn, r_ch, asn % n_channels, chs)
                joined_node.children += [pledge.identity()]
                return [collisions, assigned, True]
        elif (scheme == 'R'):
            joined_node = [j for j, c in ch_dict.items() if c == r_ch][0]
            if (joined_node.hops() + 1 < pledge.hops()):
                pledge.join(joined_node.identity(), joined_node.hops() + 1, asn, r_ch, asn % n_channels, chs)
                joined_node.children += [pledge.identity()]
                return [collisions, assigned, True]
        elif (scheme == 'D'):
            joined_node = [j for j, c in ch_dict.items() if c == r_ch][0]
            if (joined_node.hops() + 1 < pledge.hops()):
                assigned += 1
                pledge.join(joined_node.identity(), joined_node.hops() + 1, asn, assigned % n_channels, asn % n_channels, chs)
                joined_node.children += [pledge.identity()]
                return [collisions, assigned, True]
        elif (scheme == 'A'):
            joined_node = [j for j, c in ch_dict.items() if r_ch in c][0]
            if (joined_node.hops() + 1 < pledge.hops()):
                pledge.join(joined_node.identity(), joined_node.hops() + 1, asn, r_ch, asn % n_channels, chs)
                joined_node.children += [pledge.identity()]
                return [collisions, assigned, True]
                
    elif (r_ch in ch and sum([1 for c in ch if c == r_ch]) > 1):
        pledge.collisions += 1
        collisions += 1
        
    elif (r_ch not in ch):
        pledge.no_eb += 1
        #print_locked("No EB.", [joined_node.hops() for joined_node in j_nodes])
        
    else:
        print_locked("Unknown.")
    
    return [collisions, assigned, False]




# Build DODAG #
def DODAG_joining(joined_nodes, nodes, n_nodes, sf_size, eb_interval, chs, r_chs, n_channels, eb_prob, asn, scheme, timeout, atp_level, d_assigned = 0):

    d_assigned = 0
    collisions = 0

    while(len(joined_nodes) <= n_nodes):

        if ((asn - sf_size) / 100 > timeout):
            #print_locked("\n{} has very poor performance. Exiting joining process.".format(scheme.upper()))
            break
            
        start_dodag_size = len(joined_nodes)
        
        ch_dict = generate_beacons(joined_nodes, sf_size, eb_interval, asn, chs, r_chs, n_channels, scheme, eb_prob, atp_level)

        for pledge in [node for node in nodes if node.joined == False]:

            nbrs = [joined for joined in joined_nodes if (joined.identity() in pledge.nbrs and joined.joined_asn != asn)]
            #print_locked(len(nbrs), len(pledge.nbrs))

            if (len(nbrs) > 0):
                                
                if (pledge.nbr_available_asn < 0):
                   pledge.nbr_available_asn = asn

                pledge_status = pledge.joined
                
                joining_details = join_node(pledge, nbrs, sf_size, asn, chs, r_chs, n_channels, scheme, d_assigned, collisions, ch_dict)
                
                if (pledge_status == False and joining_details[-1] == True):
                    #print_locked(joining_details)
                    joined_nodes += [pledge]
                
                collisions = joining_details[0]
                d_assigned = joining_details[1]

        end_dodag_size = len(joined_nodes)
        
        #if (asn > 0 and (asn // sf_size) % 1000 == 0):
            #print_locked("\n{} has collisions: {}, at SF: {}, DODAG size: {}.".format(scheme.upper(), collisions, asn // sf_size, end_dodag_size - 1))
        
        asn += sf_size
    
    #print_locked("\nOut: {} has collisions: {}, at SF: {}, DODAG size: {} / {}.".format(scheme.upper(), collisions, asn // sf_size, end_dodag_size - 1, n_nodes))

    return joined_nodes, asn - sf_size, collisions




# Get neighbours #
def get_nbrs(lbr, nodes, comm_range):

    for each_node in nodes:
            
        for other_node in [lbr] + nodes:
                
            if (each_node.identity() != other_node.identity() and get_distance(each_node, other_node) <= comm_range):
                    
                each_node.nbrs += [other_node.identity()]

    return nodes




# Get realdata #
def get_realdata(load):

    realdata_loc = os.path.join(load, *REALDATA)

    realdata = np.loadtxt(realdata_loc, dtype='object', delimiter=",")
    print_locked("\nRealdata shape:", realdata.shape)

    return [int(val) for val in realdata[:, 0].reshape(1, -1)[0]]



# Execute method #
def execute_method(store, n_channels, sf_size, eb_interval, n_nodes, area, sims, lbr, nodes, chs, r_chs, eb_prob, mechanism, timeout, atp_level):

    total_time, avg_time, avg_nbr_time, num_joined, num_collisions = 0, 0, 0, 0, 0
    
    for sim in range(sims):

        joined_nodes = [copy.deepcopy(lbr)]
        
        joined_nodes, asn, collisions = DODAG_joining(joined_nodes, copy.deepcopy(nodes), n_nodes, sf_size, eb_interval, chs, r_chs, n_channels, eb_prob, asn = 0, scheme = mechanism, timeout = timeout, atp_level = atp_level)

        # Print results #
        print_locked("\nPledges: {}, Area: {}, Slotframe: {}, EB-Interval: {}, Channels: {}.\n{} mechanism{}, sim: {}, Timestamp: {}, DODAG size: {}, Time: {}s, Collisions: {}, at SF: {}.".format(n_nodes, area, sf_size, eb_interval, n_channels, mechanism, (" level: " + str(atp_level)) if (mechanism == 'A') else "", sim + 1, datetime.datetime.now(), len(joined_nodes) - 1, asn / 100, collisions, asn // sf_size))
        #[joined.details() for joined in joined_nodes[1:]]

        # Compute stats #
        dodag_size = len(joined_nodes) - 1 if (len(joined_nodes) > 1) else 1
        total_time += asn
        num_collisions += collisions
        num_joined += len(joined_nodes) - 1
        avg_time += round(asn / dodag_size, 2)
        avg_nbr_time += round(sum([each_node.joining_dur for each_node in joined_nodes]) / dodag_size, 2)

    # Save stats #
    fp = open(os.path.join(store, "Statistics", "_" + str(mechanism).lower() + ("_" + str(atp_level) if (mechanism == 'A') else "") + "_stats_.txt"), 'a')
    np.savetxt(fp, [n_nodes, area, n_channels, sf_size, eb_interval, round(num_joined / sims, 0), round(total_time / sims, 2), round(avg_time / sims, 2), round(avg_nbr_time / sims, 2), round(num_collisions / sims, 2)], fmt="%s", delimiter=",", newline=",")
    fp.write("\n")
    fp.close()




##  The main function   ##

#   Main    #
def main():


    # Global settings #
    seed = SEED
    num_nodes, areas, channels, comm_range, atp_levels, eb_prob, sf_sizes, eb_intervals, sims, mechanisms, timeout = \
        NODES, AREAS, CHANNELS, COMM_RANGE, ATP_LEVELS, EB_PROB, SLOTFRAME_SIZES, EB_INTERVALS, SIMS, MECHANISMS, TIMEOUT
    load, store = DATA_LOAD_LOCATION, DATA_STORE_LOCATION
    n_CPU = multiprocessing.cpu_count() - 4

    random.seed(seed)

    # Delete previous stats #
    old_stat_files = glob.glob(os.path.join(load, "Statistics", '*.*'))
    
    try:
        [os.remove(filename) for filename in old_stat_files]
    except OSError:
        pass

    realdata = get_realdata(load)
    
    n_processes = len(num_nodes) * len(areas) * len(channels) * len(sf_sizes) * len(eb_intervals) * (len(mechanisms) - 1 + len(atp_levels))
    print("\nThe total number of child processes to execute are: {}.".format(n_processes), end="\n\n")
    print_locked("\nThe total number of child processes to execute are: {}.".format(n_processes))

    processes = []

    # Iterate over different combinations of channels, slotframe sizes, EB intervals, nodes, areas, mechanisms #
    for n_channels in reversed(channels):

        chs, r_chs = [i + 1 for i in range(n_channels)], [i + 1 for i in range(n_channels)]
        
        for sf_size in sf_sizes:
            
            for eb_interval in eb_intervals:
        
                for n_nodes in reversed(num_nodes):

                    for area in reversed(areas):

                        random.shuffle(chs)
                        #print_locked("\n\n\n\nPledges: {}, Area: {}, Slotframe: {}, EB-Interval: {}, Channels: {}, CHS: {}.".format(n_nodes, area, sf_size, eb_interval, n_channels, chs), end = "\n\n")

                        # Create nodes #
                        lbr = node(area, chs = copy.deepcopy(chs))
                        nodes = [node(area, realdata = realdata, node_id = i + 1, height = n_nodes) for i in range(n_nodes)]
                        nodes = get_nbrs(lbr, nodes, comm_range)

                        # Display details of all nodes #
                        #print_locked("\nJoin coordinator details:")
                        #lbr.details()
                        #print_locked("\nPledge details:")
                        #[pledge.details() for pledge in nodes[:20]]


                        for mechanism in reversed(mechanisms):

                            if (mechanism == 'A'):
                                
                                # ATP levels #
                                for atp_level in atp_levels:

                                    while (sum(1 for process in processes if (process.is_alive())) == n_CPU):

                                        time.sleep(1)
                                    
                                    processes += [multiprocessing.Process(target=execute_method, args=(store, n_channels, sf_size, eb_interval, n_nodes, area, sims, copy.deepcopy(lbr), copy.deepcopy(nodes), copy.deepcopy(chs), r_chs, eb_prob, mechanism, timeout, atp_level, ), daemon=True)]
                                    processes[-1].start()

                            else:

                                while (sum(1 for process in processes if (process.is_alive())) == n_CPU):

                                    time.sleep(1)
                                
                                processes += [multiprocessing.Process(target=execute_method, args=(store, n_channels, sf_size, eb_interval, n_nodes, area, sims, copy.deepcopy(lbr), copy.deepcopy(nodes), copy.deepcopy(chs), r_chs, eb_prob, mechanism, timeout, -1, ), daemon=True)]
                                processes[-1].start()
                            
                            print("\rPercent jobs completed / submitted: {:6.2f} % ({}) / {:6.2f} % ({}).".format(100 * (len(processes) - sum(1 for process in processes if (process.is_alive()))) / n_processes, (len(processes) - sum(1 for process in processes if (process.is_alive()))), 100 * len(processes) / n_processes, len(processes)), end = '')


    [process.join() for process in processes]
    
    print("\n\nThe total number of child processes executed are: {}.".format(len(processes)))
    print_locked("\n\nThe total number of child processes executed are: {}.".format(len(processes)))
    




##  Call the main function  ##

#   Initiation  #
if __name__=="__main__":

    try:

        open(os.path.join(os.path.dirname(sys.argv[0]), "Log_Files", "_Log_" + str(sys.argv[0].split("\\")[-1].split('.')[0]) + ".txt"), 'w').close()

        #   Start logging to file     #        
        print_locked('\n\n\n\n{:.{align}{width}}'.format("Execution Start at: " 
            + str(datetime.datetime.now()), align='<', width=150), end="\n\n")

        print_locked("\n\nPython Version:\n\n" + str(platform.python_version()))
        
        print_locked("\n\nProgram Name:\n\n" + str(sys.argv[0].split("\\")[-1]))
        
        print_locked("\n\nProgram Path:\n\n" + os.path.dirname(os.path.abspath(sys.argv[0])))
        
        print_locked("\n\nProgram Name With Path:\n\n" + os.path.abspath(sys.argv[0]))

        print_locked("\n\nProgram arguments aubmitted:\n\n" + str(sys.argv[1:]))

        print_locked('\n\n\n\n{:.{align}{width}}'.format("Program Body Start:", align='<', width=50), end="\n\n\n")

        
        #   Clear the terminal  #
        os.system("clear")
        
        #   Call the main program   #
        start = datetime.datetime.now()
        print("\n\nExecution Start at:", datetime.datetime.now())

        #    main()
        main()

        print_locked('\n\n{:.{align}{width}}'.format("Program Body End:", align='<', width=50), end="\n\n\n\n")

        print_locked("\nProgram execution time:\t\t", datetime.datetime.now() - start, "hours\n")

        print("\nProgram execution time:", datetime.datetime.now() - start, "hours\n")


        #    Wait for manual exit
        #input('\n\n\n\nPress ENTER to exit: ')

    except Exception:
    
        print_locked(traceback.format_exc())


##   End of Code   ##

