import threading
import time
import csv

exitFlag = 0

class myThread (threading.Thread):
   def __init__(self, threadID, name, counter):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
   def run(self):
      print ("Starting " + self.name)
      print_time(self.name, self.counter, 5)
      print_to_csv(1,2, self.name + ".csv")
      print ("Exiting " + self.name)

def print_time(threadName, delay, counter):
   while counter:
      if exitFlag:
         threadName.exit()
      time.sleep(delay)
      print ("%s: %s" % (threadName, time.ctime(time.time())))
      counter -= 1

def print_to_csv(value1, value2, filename):
    # Note: Should probably use a lock when writing to the same file
    # http://effbot.org/zone/thread-synchronization.htm
    # https://stackoverflow.com/questions/33107019/multiple-threads-writing-to-the-same-csv-in-python
    with open(filename, "wb") as outfile:
        writer = csv.writer(outfile)
        writer.writerow([value1, value2])
        outfile.close()

thread_list = []

for i in range(5):
    # Create new threads
    cur_thread = myThread(i, "Thread-"+str(i), i)
    cur_thread.start()
    thread_list += [cur_thread]

for thread in thread_list:
    thread.join()

print ("Exiting Main Thread")
