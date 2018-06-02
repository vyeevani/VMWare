import tensorflow as tf

tasks = ["localhost:2222", "localhost:2223"]

jobs = {"local": tasks}

spec = tf.train.ClusterSpec(jobs)

server1 = tf.train.Server(spec, "local", task_index=0)
server2 = tf.train.Server(spec, "local", task_index=1)

tf.reset_default_graph()

var = tf.Variable(initial_value=0.0)

sess1 = tf.Session(server1.target)
sess2 = tf.Session(server2.target)

sess1.run(tf.global_variables_initializer())
sess2.run(tf.global_variables_initializer())

print("Initial Value of var in sess1: ", sess1.run(var))
print("Initial Value of var in sess2: ", sess2.run(var))

sess1.run(var.assign_add(1.0))

print("Value of var in sess1: ", sess1.run(var))
print("Value of var in sess2: ", sess2.run(var))


