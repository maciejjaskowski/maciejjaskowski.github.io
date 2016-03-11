---
title: Deep Q-Learning (Space Invaders)
---


Ever since I learned about neural networks playing Atari games I wanted to reimplemnted it and learn how it works. Below you can see an AI playing Space Invaders. 
I trained it during my batch at <a href="http://recurse.com">Recurse Center</a> on little over 50M frames. 

<center>
<video width="480" height="350" controls autoplay loop>
  <source src="{{site_url}}/assets/space_invaders/space_invaders.webm" type="video/webm">
  <source src="{{site_url}}/assets/space_invaders/space_invaders.mp4" type="video/mp4">
  I'm sorry; your browser doesn't support HTML5 video in WebM with VP8 or MP4 with H.264.
</video>
</center>


It is more awesome if you realize that the AI was trained in a similar way a human would learn: the only inputs are screen and number of gained (or lost) points after each action taken by the AI.

DQN does much better then a best-action strategy (do nothing but shoot) and random strategy. 

| algorithm | points |
| --------- | ------ |
| DQN       | *550*    |
| Best action  | 240 |
| Random | 150 |

You can play with my implementation here: <a href="https://github.com/maciejjaskowski/deep-q-learning">Deep Q-Learning</a>. This is first post on the topic, stay tuned for the next ones!

<script src="http://d3js.org/d3.v3.min.js"></script>
<svg id="example" width='600' height='300'></svg>

<script>
"use strict";
console.log("ABC")

var w = 600
var h = 300
var pad = 30

var xScale = d3.scale.linear().domain([0,90000]).range([pad, w - pad])
var yScale = d3.scale.linear().domain([650, 0]).range([pad, h - pad])

var lineFunction = d3.svg.line()
                          .x(function(d) { return xScale(d.game); })
                          .y(function(d) { return yScale(d.reward); })
                          .interpolate("linear"); 


var svg = d3.select("#example")

var xAxis = d3.svg.axis();
xAxis.scale(xScale);
xAxis.orient("bottom");

var styleAxis = function(axis) {
  return axis
    .style("fill", "none")
    .style("stroke", "black")
    .style("shape-rendering", "crispEdges")
    .style("font-family", "sans-serif")
    .style("font-size", "11px")
}
styleAxis(svg.append("g")
    .attr("transform", "translate(0," + (h - pad) + ")")
    .call(xAxis))

var yAxis = d3.svg.axis()
yAxis.scale(yScale)
yAxis.orient("left")

styleAxis(svg.append("g")
   .attr("transform", "translate(" + pad + ",0)")
   .call(yAxis))
   .append("text")
   .text("Average game reward")
   .attr("x", 20)
   .attr("y", 20)
   .attr("transform", "rotate(-90)")
   

svg.selectAll("random")
   .data([[{'game': 0, 'reward': 150}, {'game': 90000, 'reward': 150}]])
   .enter()
   .append("path")
   .attr("class", "random")
   .attr("stroke", "blue")
   .attr("fill", "none")
   .attr("d", lineFunction)

svg.selectAll("best-action")
   .data([[{'game': 0, 'reward': 240}, {'game': 90000, 'reward': 240}]])
   .enter()
   .append("path")
   .attr("class", "best-action")
   .attr("stroke", "red")
   .attr("fill", "none")
   .attr("d", lineFunction)

svg.selectAll("max")
   .data([[{'game': 0, 'reward': 550}, {'game': 90000, 'reward': 550}]])
   .enter()
   .append("path")
   .attr("class", "max")
   .attr("stroke", "#ccc")
   .attr("stroke-dasharray", "10,5")
   .attr("fill", "none")
   .attr("d", lineFunction)



d3.csv("{{site_url}}/assets/space_invaders/avg_reward_per_game.csv")
  .get(function(err, rows) {
     svg.selectAll("chart").data([rows]).enter().append("path")
        .attr("class", "chart")
        .style("stroke", "black")
        .style("stroke-width", "1")
        .style("fill", "none")
        .attr("d", lineFunction)
        
  })


</script>

Average game reward (600 games) after N games played. Blue line is random strategy baseline, red line is best-action strategy baseline.

#Algorithm
So what is Deep Q-Learning (DQN)? Below you will find a gentle introduction. I omit certain details for the sake of simplicity
and I encourage you to read the <a href="http://deeplearning.net/software/theano/extending/unittest.html">original paper</a>. 

The task for Neural Network in DQN is to learn which action to take based on the screen and previous experience. Obviously the nueral network should choose the best action but how to learn which one is best?

Turns out your neural network can be pretty simple: the input is game screen and hidden layers consists of 3 convolutional layers and a single fully connected layer. The number of neurons in last layer corresponds to number of actions that can be taken.
In the case of Space Invaders there were 4 actions (do nothing, shoot, go left, go right), therefore there were 4 neurons in the output layer. 

The tricky and crucial part is the loss function.
In 'normal' neural networks the loss function is straigtforward as for each training example $$X$$ there is a known expected outcome $$y$$. 
Nothing like that is available in our case but we can deal with it thanks to some insights from <a href="https://en.wikipedia.org/wiki/Q-learning">Q-Learning</a>!

# Loss function
The best measure of how good an action is <em>accumulated future reward</em>

$$ \sum_{i=t_0} r_i  $$

where sum is taken over time from $$t_0$$ until the end of the game and $$r_i$$ is reward gained at time $$i$$. There is a couple of problems with that simplified definition and we'll deal with them one by one. 

For one there is no way to calculate that sum as we don't know the future. With this, we'll deal at the end though.

Intuitively speaking the immediate reward $$r_{t_0}$$ should be more valuable then a very distant one. After all, future is uncertain and we might never get this distant reward at all.  
Let's introduce <em>discounted accumulated future reward</em>.

$$ \sum_{i=t_0} \gamma^i r_i  $$

Where $$\gamma$$ is between 0 and 1. We'll set $$\gamma$$ to $$0.99$$, though, as the distant rewards are very important.

Finally our game is stochastic (we don't know when an enemy shoots a laser beam) therefore we should rather think in terms of expected value.
Ideally, what we want the neural network to learn is function Q defined as:

$$ Q(s)(a) = \mathbb{E}\left(\sum_{i=t_0} \gamma^i r_i \right) \quad \quad  \text{Expected discounted accumulated future reward} $$

where $$s$$ is the input game screen at time $$t_0$$, $$a$$ indicates the neuron corresponding with action $$a$$, $$r_{i}$$ is reward obtained after action taken at time $$i$$.
That is what we want each neuron of the output layer to learn.

To do that efficiently we need to realise that $$Q(s)(a) = r + \gamma \max_{a'}Q(s')(a')$$ where $$s'$$ is game screen experienced after taking action $$a$$ after seeing game screen $$s$$.

Now if $$Q^{*}$$ is our neural network we can treat $$Q^{*}(s)(a) - \left(r + \gamma \max_{a'}Q^{*}(s')(a')\right)$$ as a measure of surprise and therefore a loss function (after squaring).

Note that the loss depends on the neural network itself in an untypical way.


# Correlation
Since we play the game online it is tempting to simply update the network after each taken action or in mini-batches of, say, 32 actions taken. Such mini-batches would be highly correlated and any stochastic update algorithm would fail on that.

To deal with that issue we keep previous experiences in memory and after each action taken we draw a mini-batch of experiences from that memory to perform the update step.

# Other technicalities
* provide only every 4th frame to the neural network
* stack 4 frames one on top of the other to make the neural network aware of time. This could be avoided if you used LSTM.
* keep a stale network at hand and calculate loss with regards to that stale network
* gradient clipping (to avoid blowing up gradients)
* DeepMind Rmsprop (instead of normal one) - improved performance by 40% in my case.
* I used <a href="https://github.com/maciejjaskowski/deep-q-learning/blob/master/arcadelearningenvironment.org">Arcade Learning Environment</a> to play space invaders.


# Lessons Learned
That was my first exposure to training non trivial neural networks so there is plenty of things that I learned.

1. Nan's as weights is no good. Seems obvious but it does not mean that it's easy to track down such problems. Theano <a href="http://deeplearning.net/software/theano/tutorial/nan_tutorial.html">provides means</a> of doing that efficiently. Of course an NaN usually means that you divide $$\infty$$ by $$0$$.

2. Test your Theano code. It's not so hard! And here is relevant <a href="http://deeplearning.net/software/theano/extending/unittest.html">documentation</a>.

3. If your network converges or diverges to $$\infty$$ very quickly it's probably caused by suboptimal learning rates applied in your update function. Little is known about how to correctly choose network's hyperparameters so trial, error and verification is what's left.

4. Update method might play a gigantic role in performance of your neural network. In my case, learning curve of my DQN implementation flattened after 10M frames around 400 points for traditional RMSProp. "DeepMind" RmsProp was learning slower but boosted the performance to over 550 points on average after 50M frames and one can clearly see that it kept learning all the time.
