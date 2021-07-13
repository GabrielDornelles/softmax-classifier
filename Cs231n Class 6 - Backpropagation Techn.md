# Cs231n Class 6 - Backpropagation Techniques
SGD is pretty good but its really slow, momentum will give it acceleration to get faster in the minima of the loss function.

## Momentum update

### Original class:
In particular, the loss can be interpreted as the **height** of a hilly terrain (and therefore also to the potential energy since U=mgh and therefore U∝h ). **Initializing the parameters with random numbers is equivalent to setting a particle with zero initial velocity at some location**. The **optimization process** can then be seen as **equivalent to the process of simulating the parameter vector (i.e. a particle) as rolling on the landscape**.
- Note: **Potential energy** is the energy held by an object because of its relative position, ready to be used later.

### Notes:
Adding momentum to the vanilla SGD means that we're adding acceleration to the gradient (analogous to physics, where f=ma, so the (negative) gradient is in this view proportional to the acceleration of the particle.)
It has a physical interpretation like a ball rolling down the loss function landscape + friction (mu coefficient).

```py
# vanilla SGD
x += learning_rate * dx
v = 0 # initialize velocity at 0
# momentum update
# mu usually something between 0.5 and 0.99
v = mu * v - learning_rate * dx # integrate velocity
x += v # integrate position
```

- Note: kinetic energy is the energy of a system due to its movement.

Here we see the variable mu, which controls velocity (positive or negative acceleration) reducing the kinetic energy of the system, otherwise the particle (our gradient) would never come to stop at the bottom of the loss function hill (global minima of the loss function).
![gif](https://user-images.githubusercontent.com/56324869/125319344-0ef7c480-e311-11eb-8e1b-19acef0bfd4b.gif)


Usually, momentum is increased in later stages of learning. A typical setting is to start with momentum of about 0.5 and anneal it to 0.99 or so over multiple epochs.

## AdaGrad
Per parameter **Ada**ptive learning rate method for the **grad**ients. Using that, every single parameter in your net now has kind of its own learning rate that scaled dynamically based of what kinds of gradient we saw (we are adding the square of the parameters to cache).

```py
# AdaGrad update
cache += dx**2
x +=  - learning_rate * dx  / (np.sqrt(cache) + 1e-7)
```
If you are training for too much time, your learning rate decays towards zero, because you are caching the squared gradient and summing all of them, eventually you are divinding by large values and the update is almost 0.
- Note: 1e-7 is there just to prevent a division by zero, when we get too close to it.

## RMSProp
![image](https://user-images.githubusercontent.com/56324869/125327737-d27c9680-e319-11eb-9ec9-b371a43d82db.png)

We add a variable **decay_rate** that usually is set to 0.9, and leak a bit of the cache, so we don't end up getting close to 0 as fast as AdaGrad, and also that's optimized to mini-batches.
```py
# RMSProp
cache = decay_rate * cache + (1 - decay_rate) * dx**2
x +=  - learning_rate * dx  / (np.sqrt(cache) + 1e-7)
```

## Adam
**Ada**ptative **M**oment estimation. Also computes adaptative learning rates for each parameter in the neural net.
Looks like RMSProp with momentum. We do RMSProp, so it looks like Adagrad but leaking a bit of its cache, and also add momentum to it.
- beta1: usually = 0.9
- beta2: usually = 0.995

### Incomplete version, but close:
```py
m = beta1*m + (1-beta1)*dx # update first moment (momentum like). beta1 is like the friction we had in Momentum SGD.
v = beta2*v + (1-beta2)*(dx**2) # update second moment (RMSProp like) where our cache was: 
# cache += decay_rate * cache + (1 - decay_rate)*(dx**2)

# Adam update
x += - learning_rate * m / (np.sqrt(v) + 1e-7)
```

### Complete version
```py
m,v = #initialize chaces to zeros
for t in xrange(0,big_number):
    dx = #evaluate gradient
    m = beta1*m + (1-beta1)*dx # momentum
    v = beta2*v + (1-beta2)*(dx**2) #rmsprop
    m /= 1-beta1**t # correct bias
    v /= 1-beta2**t # correct bias
    x +=  - learning_rate * m / (np.sqrt(v) + 1e-7) 
```
Where the bias correction compensates the fact that m,v are initialized at zero and need some time to "warm up".

## Visualization
![opt1](https://user-images.githubusercontent.com/56324869/125450065-957493d6-85e4-4f20-be85-41457cc6a066.gif)

A visualization of a saddle point in the optimization landscape, where the curvature along different dimension has different signs (one dimension curves up and another down). Notice that SGD has a very hard time breaking symmetry and gets stuck on the top. Conversely, algorithms such as RMSprop will see very low gradients in the saddle direction. Due to the denominator term in the RMSprop update, this will increase the effective learning rate along this direction, helping RMSProp proceed. 

![opt2](https://user-images.githubusercontent.com/56324869/125450069-4a40296d-e7d7-4c6a-b50a-0fa88bbf011f.gif)

Images credit: Alec Radford.



Tags:
  Backpropagation