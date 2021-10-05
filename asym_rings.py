
import numpy as np
import matplotlib.pyplot as plt 
import scipy
import math
import random


class Node:
	def __init__(self, counter, output, posx, posy, external, EI):
		self.weight = [] #each neuron will have a weight for each input. 
		self.input = [] #each neuron will have a list of inputs that spans the entire network. 
		self.output = output #each neuron has one output that can feed into other inputs
		self.out_nodes = [] #each neuron has a list of further connections to other neuronsself.out_nodes = [] #each neuron has a list of further connections to other neurons
		self.counter = counter #this allows for simple tracking of each neuron
		self.external = external #this is the external input on a neuron, outside of the network.
		self.posx = posx #the cartesian position of the neuron. 
		self.posy = posy
		self.next = None #allows for a connection based on a linked list. could also consider this to be a graph. 
		self.EI = EI #boolean value 0 for inhib 1 for excitation
		self.spike_times = [] #stores spike times for each neuron.

class neuralNetwork:
	def __init__(self):
		self.head = None

#insert function. This will insert a neuron into the neural network, and update the position of the neuron.
#at the end of every function call, there will be a network of neurons the size of counter+1 that has 
#counter+1 input and weight slots. 
#N represents the total number of neurons in the network. 
	def insert_E(self, counter, N):

			newNode = Node(counter, 0, 0, 0, 0, 1) #make a node that is initialized to 0, with counter value (counter)
			head = self.head

			if(head):
				current = self.head
				while(current):
					current.input.append(0)
					current.weight.append(0)
					current.out_nodes.append(1) #all to all connections
					current = current.next
			
				current = self.head

				while(current.next):
					current = current.next

				current.next = newNode
				for i in range(counter):
					newNode.input.append(0)
					newNode.weight.append(0)
					newNode.out_nodes.append(1)
				newNode.weight.append(0)
				newNode.input.append(0)
				newNode.out_nodes.append(1)
				newNode.posx = math.cos(counter * 2 * np.pi / N)
				newNode.posy = math.sin(counter * 2 * np.pi / N)
				 
			else:
				newNode.posx = math.cos(counter * 2 * np.pi / N)
				newNode.posy = math.sin(counter * 2 * np.pi / N)
				newNode.input.append(0)
				newNode.weight.append(0)
				self.head = newNode
	def insert_I(self, counter, N): #for the sake of experiment, position == (0,0)

			newNode = Node(counter, 0, 0, 0, 0, 0) #make a node that is initialized to 0, with counter value (counter)
			head = self.head

			if(head):
				current = self.head
				while(current):
					current.input.append(0)
					current.weight.append(0)
					current.out_nodes.append(1) #all to all connections
					current = current.next
			
				current = self.head

				while(current.next):
					current = current.next

				current.next = newNode
				for i in range(counter+1):
					newNode.input.append(0)
					newNode.weight.append(0)
					newNode.out_nodes.append(1)
				
				newNode.posx = 0
				newNode.posy = 0
				 
			else:
				newNode.posx = 0
				newNode.posy = 0
				newNode.input.append(0)
				newNode.weight.append(0)
				self.head = newNode

	def external_inputs(self, input_posx, input_posy):
			head = self.head
			current = head
			for i in range(N):
				for x in range(len(input_posx)): #based on location of external input, each neuron will fire in a different way.
					distance = math.sqrt((current.posx - input_posx[x])**2 + (current.posy - input_posy[x])**2)
					current.external = 1/distance
					if(current.external < 0.25):
						current.external = 0;
				current = current.next
	def weights(self):
		head = self.head
		current = head
		
		for i in range(N):
			other_node = current.next
			for x in range(N-1):

				distance = math.sqrt((current.posx - other_node.posx)**2 + (current.posy - other_node.posy)**2)
				if(current.EI == 1):
					current.weight[other_node.counter] = 1/distance
				else:
					current.weight[other_node.counter] = -1/distance
				current.weight[other_node.counter] += 0.1 * random.randint(-2, 2) #add noise to maintain an asymmetric network. remove to show symmetric ring
				other_node = other_node.next
			current = current.next

	
#step function, can also be a sigmoid. 
def reLU(curr, X):
	if(curr.EI == 0): 
		return min(0,X)
	else:
		return max(0, X)


def get_Queue(curr, Queue): 
	for x in range(len(curr.out_nodes)):
		#print("x == ", x) #update the inputs of the other neurons
		dum = curr
		if (curr.out_nodes[x] == 1):
			for y in range(x):
				dum = dum.next
				#print(dum.counter)
			Queue = enqueue_PQ(Queue, dum, curr)
			#check queue
			curr_queue = Queue
			while(curr_queue != None):
				#print("curr_queue val == ", curr_queue.counter)
				curr_queue = curr_queue.next

			dum.input[x] = curr.output
	return curr, Queue


def update_outputs(curr, outputs):
	out_val = curr.external + (np.dot(curr.input, curr.weight))
	curr.output = reLU(curr, out_val)
	if(curr.output > 0):
		curr.spike_times.append(time)
	node_excited.append(curr.counter)
	outputs.append(curr.output)
	
	return curr
def update_inputs(curr):
	for x in range(len(curr.out_nodes)): #update the inputs of the other neurons
		dummy = curr
		if curr.out_nodes[x] == 1:
			for y in range(x):
				dummy = dummy.next
		dummy.input[x] = curr.output
	return

def copy_neuron_info(curr):
	newNode = Node(curr.counter, 0, 0, 0, 0, 0)
	newNode.EI = curr.EI
	newNode.input = curr.input.copy()
	newNode.weight = curr.weight.copy()
	newNode.out_nodes = curr.out_nodes.copy()
	newNode.next = None
	return newNode

def enqueue_PQ(Queue_head, dummy, curr): #problem is here
	newNode = copy_neuron_info(dummy)
	if(Queue_head == None):
		#print("null head runs")
		Queue_head = newNode
		return Queue_head
	elif(Queue_head != None):
		#print("non null head runs")
		curr_queue = Queue_head
		if(curr.weight[newNode.counter] > curr.weight[curr_queue.counter]): #if weight curr->newNode is > than weight of curr->queue_head,
			newNode.next = curr_queue									  #then newNode is the new head of the queue.
			curr_queue = newNode
			#print("new curr_counter == ", curr_queue.counter)
			return curr_queue
		else: #if this is not the case, check every other node and add node in the correct position
			if(curr_queue.next == None): 
				curr_queue.next = newNode
				return Queue_head
			while(curr_queue.next != None):
				curr_holder = curr_queue
				curr_queue = curr_queue.next
				if(curr.weight[dummy.counter] > curr.weight[curr_queue.counter]):
					newNode.next = curr_queue
					curr_holder.next = newNode
					return Queue_head
			while(curr_queue.next != None):
				curr_queue = curr_queue.next
			curr_queue.next = newNode
	return Queue_head




network = neuralNetwork()
N = 6

for i in range(N-1):
	network.insert_E(i, N-1)

network.insert_I(N-1, N) #1 inhibitory neuron in the system

current = network.head
while(current.next):
	#print("current EI == ", current.EI)
	current = current.next
print(current.counter)
current.next = network.head

network.weights()
#show neural network locations
x = []
y = []
for i in range(N):
	x.append(current.posx)
	y.append(current.posy)
	print(current.posx, current.posy)
	current = current.next
plt.scatter(x,y)
plt.show()
head = network.head
curr = head
input_posx = [1,0,-1]
input_posy = [2, 2, 2]

network.external_inputs(input_posx, input_posy)

head = network.head
curr = head
dummy = curr
largest_ext = curr.external
for x in range(N):
	if(dummy.external > largest_ext):
		largest_ext = dummy.external
	dummy = dummy.next

time = 0
t_final = 10 # run for 10ms 
spikes = []
node_excited = []
outputs = []
for i in range(N):
	spikes.append(current.spike_times)
	current = current.next
#run the simulation. this simulation is a circular ring network with no external input.
#constant weights, one spike every 0.1 ms. 
while(curr.external != largest_ext):
	curr = curr.next
Queue = None #the nodes affected by the current node

while time <= t_final:
	
	out_val = curr.external + (np.dot(curr.input, curr.weight))
	curr.output = reLU(curr, out_val)
	if(curr.output != 0):
		curr.spike_times.append(time)
	node_excited.append(curr.counter)
	outputs.append(curr.output)
	for x in range(len(curr.out_nodes)): #update the inputs of the other neurons
		dummy = curr
		if curr.out_nodes[x] == 1:
			for y in range(x):
				dummy = dummy.next
		dummy.input[x] = curr.output

	curr, Queue = get_Queue(curr, Queue)
	time = time + 0.1
	old_Queue = Queue
	while(Queue != None):
		dummy = head
		while (Queue.counter != dummy.counter):
			dummy = dummy.next

		dummy = update_outputs(dummy, outputs)

		#update those outputs for each dummy
		Queue = Queue.next
	#go to next excited neuron
	dummy = curr
	for x in range(len(curr.out_nodes)):
		update_inputs(dummy)
		dummy = dummy.next
	node_excited.append('\n')
	outputs.append('\n')
	while(old_Queue.counter != curr.counter):
		curr = curr.next
	Queue = None


curr = head
weights = []
for i in range(N):
	weights.append(curr.weight)
	curr = curr.next
plt.imshow(weights, cmap=plt.cm.Blues)
plt.colorbar()
plt.show()
for i, spike_times in enumerate(spikes):
	plt.vlines(spike_times, i+0.5, i+1)

plt.show()
print(node_excited)
#print("\n")
print(outputs)


input_posx = [1,0,-1]
input_posy = [-2, -2, -2]

network.external_inputs(input_posx, input_posy)
head = network.head
curr = head
dummy = curr
largest_ext = curr.external
for x in range(N):
	if(dummy.external > largest_ext):
		largest_ext = dummy.external
	dummy = dummy.next
	
while(curr.external != largest_ext):
	curr = curr.next
Queue = None 

time = 0
while time <= t_final:
	
	out_val = curr.external + (np.dot(curr.input, curr.weight))
	curr.output = reLU(curr, out_val)
	if(curr.output != 0):
		curr.spike_times.append(time)
	node_excited.append(curr.counter)
	outputs.append(curr.output)
	for x in range(len(curr.out_nodes)): #update the inputs of the other neurons
		dummy = curr
		if curr.out_nodes[x] == 1:
			for y in range(x):
				dummy = dummy.next
		dummy.input[x] = curr.output

	curr, Queue = get_Queue(curr, Queue)
	time = time + 0.1
	old_Queue = Queue
	while(Queue != None):
		dummy = head
		while (Queue.counter != dummy.counter):
			dummy = dummy.next

		dummy = update_outputs(dummy, outputs)

		#update those outputs for each dummy
		Queue = Queue.next
	#go to next excited neuron
	dummy = curr
	for x in range(len(curr.out_nodes)):
		update_inputs(dummy)
		dummy = dummy.next
	node_excited.append('\n')
	outputs.append('\n')
	while(old_Queue.counter != curr.counter):
		curr = curr.next
	Queue = None


curr = head
weights = []
for i in range(N):
	weights.append(curr.weight)
	curr = curr.next
plt.imshow(weights, cmap=plt.cm.Blues)
plt.colorbar()
plt.show()
for i, spike_times in enumerate(spikes):
	plt.vlines(spike_times, i+0.5, i+1)

plt.show()
print(node_excited)
#print("\n")
print(outputs)



