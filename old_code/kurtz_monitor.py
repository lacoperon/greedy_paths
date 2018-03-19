import subprocess

from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

sched = BlockingScheduler()

@sched.scheduled_job('interval', seconds=1)
def appendTop():
    ## use the following where appropriate within your loop
    with open("ss.txt", "a") as outfile:
        outfile.write("\n--- {} ---\n".format(str(datetime.now())))
        subprocess.call("top -n1", shell=True, stdout=outfile)

sched.start()
