from utils.train import train
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='Distributed training of Image Captioning Network')
parser.add_argument('job_name', metavar='type of machine', type=str,
                    help='Type of Machine, either Parameter Server or Worker Server.')
parser.add_argument('task_id', metavar='specific task', type=int,
                    help='ID of the task that this particular machine needs to carry out.')

args = parser.parse_args()
job_name = args.job_name
task_id = args.task_id

cluster = tf.train.ClusterSpec({
    "ps" : ["sc2tf01:8000"],
    "worker" : ["sc2tf02:8000", "sc2tf03:8000", "sc2tf4:8000", "sc2tf5:8000]
})

server = tf.train.Server(cluster, job_name=job_name, task_index=task_id)

if job_name == "ps":
    server.join()

else:
    device_function = tf.train.replica_device_setter(worker_device="/job:worker/task:" + str(task_id))
    with tf.device(device_function):
        try:
            train(.001,False,False, server=server, is_chief=(task_id == 0), device_function=device_function) #train from scratch
            #train(.001,True,True)    #continue training from pretrained weights @epoch500
            #train(.001,True,False)  #train from previously saved weights
        except KeyboardInterrupt:
            print('Exiting Training')
