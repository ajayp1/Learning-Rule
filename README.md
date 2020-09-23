# Learning Rule Comparison Project

In this project, I trained three different convolutional neural networks (CNNs) using three different learning rules. I then compared the representations of each network with neural data to determine which learning rule is the most biologically plausible. Finally, I compared the representation of each network over the course of training to evidence of semantic development in infants.

# Background

Normative models which are optimized for task performance have been shown to produce representations that correspond well to patterns of neural activity in the visual cortex [1](https://www.pnas.org/content/111/23/8619). Such models - known as deep convolutional neural networks (CNNs) - are loosely based on the hierarchical architecture of the visual system and are trained by minimizing an error function using a technique known as gradient descent [2](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networ). However, it is difficult to imagine that biological neural networks rely on (1) a supervised learning scheme and (2) gradient descent for optimization [3](https://www.biorxiv.org/content/10.1101/058545v3). As such, numerous “biologically-plausible” learning rules have been proposed for credit assignment in brains. In this project, I will compare two well-known alternatives to supervised learning and gradient descent: Hebbian learning and burst-dependent synaptic plasticity. First, I will compare representations generated by each learning rule using representational analysis to neural data. Next, I will consider whether the evolution of these representations during training corresponds well to semantic development observed in infants. 

# Methods

## Network

Within the brain, object recognition occurs within the ventral stream - a series of layers (denoted as V1 through V4 and the inferior temporal (IT) cortex) that encode increasingly complex features of visual stimuli. In this project, I used the convolutional network employed by Yamins and DiCarlo where artificial layers serve as a model for the layers of the ventral stream [1](https://www.pnas.org/content/111/23/8619), [4](https://www.biorxiv.org/content/10.1101/408385v1). In this architecture, each “brain area” consists of two convolution, rectification, and normalization layers followed by a max-pooling operation. The number of filters progressively increased in each brain area. To meet computational constraints, the number of filters and layers were adjusted for each learning rule. For object recognition, a linear readout was used to predict a given object-class. 

## Learning Rules

*Gradient Descent*

Interestingly enough, gradient descent was first formulated by Cauchy [5](https://dl.acm.org/doi/10.1007/BF01386306). The learning rule works by estimating the discrepancy between a model’s predicted value and the actual recorded value. This discrepancy is quantified by a loss function which is then minimized. Instead of solving for an analytical minimization, gradient descent estimates a local derivative of the loss function to determine an optimal adjustment to be made to the model. This process is iteratively repeated until convergence.

*Hebbian Learning*

Hebbian Learning was proposed by Donald Hebb and can be summarized as “neurons that fire together wire together [6](http://s-f-walker.org.uk/pubsebooks/pdfs/The_Organization_of_Behavior-Donald_O._Hebb.pdf).” That is, if the activity of a given neuron coincides with a directly connected downstream neuron, then the connection between the two neurons is strengthened. Mathematically, this rule is implemented as a weight update in a neural network where the update is proportional to the product of the activation of each neuron with a proportionality constant as a hyperparameter. Hebbian plasticity has been discovered in numerous neurophysiological experiments [7](https://www.sciencedirect.com/science/article/pii/S016501739600015X?casa_token=pzd66OaM7bcAAAAA:JHijw11Ht_z5Ryk586tHepMewSkz0_3Ji-pIq2iTeY0mWoIa-pj9n97ftcjff5Ujk2skDj35scI). In this project, I used an implementation for Hebbian learning for a convolutional neural network developed by Gabriele Lagani [8](https://github.com/GabrieleLagani/HebbianLearningThesis).  

*Burst-dependent synaptic plasticity*

Burst-dependent synaptic plasticity - henceforth “Burstprop” - was developed by Blake Richards’ group and motivated by previous observations that bursting in apical dendrites can serve as feedback signals from downstream neurons [9](https://www.biorxiv.org/content/10.1101/2020.03.30.015511v1.full). In Burstprop, these feedback  connections allow for global error signals to be transmitted throughout the entire network. As such, Burstprop provides an implementation for credit assignment that is supported by a variety of physiological evidence [10](https://www.jneurosci.org/content/13/6/2391.short). 

## Dataset

I compared the representations of each CNN to fMRI data collected by Horikawa & Kamitani and made available on [openneuro.org](https://openneuro.org/datasets/ds001246/versions/1.0.1). In their experiment, subjects were presented with stimuli from the ImageNet hierarchical database [11](http://www.image-net.org/papers/imagenet_cvpr09.pdf). Each CNN was trained with all available examples for each ImageNet category of stimuli presented to an experimental test subject. 

## Representational Similarity Analysis

In order to compare representations of CNNs to neural data, I used Representational Similarity Analysis (RSA) to generate representational dissimilarity matrices (RDMs) [12](https://www.frontiersin.org/articles/10.3389/neuro.06.004.2008/full). The elements in these matrices indicate distances between pairs of response patterns of a neural network or brain area. A similarity metric (dot product) was then used to compare the unrolled matrices. 

# Results

## Gradient Descent confers the greatest task performance

| ![GD_learningcurve](https://imgur.com/a/We5GLs8)  | ![Burstprop_learningcurve](https://imgur.com/a/EvPEPR6) | ![Hebbian_learningcurve](https://imgur.com/a/mNVAedJ) |
| ------------- | ------------- | ------------- |

Unsurprisingly, gradient descent achieved the highest performance with a test accuracy of 48.55%. Burst-dependent synaptic plasticity - nicknamed “Burstprop” - performed similarly as in its original publication with a test accuracy lower than gradient descent (19.81%) but still well above chance-level (0.67%). As expected, Hebbian learning performed poorly. It appears that its highly simplified weight update fails to promote representations that capture complex image statistics. An additional limitation of specific the Hebbian learning implementation I used was its high computational cost, which required a lowered input image resolution as well as decreased filter number for every convolutional layer. These limitations very likely affected classification accuracy.  

## Burstprop rivals Gradient Descent in its similarity to neural activity

Representational Similarity Analysis (RSA) was used to compare the representations of the deep nets to fMRI brain activity. The representational dissimilarity matrices (RDMs) generated by RSA were used to compute the dot-product similarity between layers of each CNN and activity in corresponding brain areas. 

![RSA_bar_chart](https://imgur.com/a/p83mMnB)

Interestingly, RSA revealed that Burstprop has a high similarity to brain activity despite achieving a far lower accuracy than gradient descent. This suggests that training a model for the best possible performance may not be necessary for it to mirror neural activity. An additional finding is that the CNN trained on Hebbian learning steadily decreased in similarity to higher regions along the ventral stream. This suggests lower areas in the visual system may operate under a simple and perhaps unsupervised learning rule. 
    
## Dissimilarity matrices indicate that none of the learning rules achieve brain-like stimulus organization

In addition to computing similarity between different brain regions and CNN layers, I also plotted the RDMs of evoked responses in area V4 of the visual cortex as well as the final layer of each model which corresponds to V4 [1](https://www.pnas.org/content/111/23/8619). 

| ![fMRI_RDM](https://imgur.com/CV3kG2o)  | ![GD_RDM](https://imgur.com/a/X6IaW5w)  | ![burst_RDM](https://imgur.com/a/zGaQLaU) | ![hebb_RDM](https://imgur.com/a/OcNPysU) |
| ------------- | ------------- | ------------- | ------------- |

First, the RDM of fMRI activity shows that V4 does not organize stimuli according to an easily identifiable hierarchy. That is, stimuli are not grouped by any obvious category distinction. While past studies have shown some amount of semantic organization along the ventral stream [13](https://www.nature.com/articles/nn.4247), [14](https://www.jneurosci.org/content/30/39/12978), most studies have found that structured semantic organization is most prominent in IT - the final stage of the ventral stream [15](https://www.sciencedirect.com/science/article/pii/S0896627308009434). However, there appears to be two distinct regions of high similarity in the fMRI RDM, suggesting that some stimuli may be organized in V4 according to some unknown heuristic. Further study on the statistics of the clustered images might reveal the underlying rule driving such representations. 
    
Second, the discrepancy in representations between CNN and brain RDMs suggests that the three learning rules may not rely on semantic organization to produce representations that are helpful for classification. A possible future direction is the development of new learning rules which prioritize semantically organized representations. On the other hand, the poor qualitative similarity between the CNN RDMs and the brain RDM suggests that RSA may not be a reliable indicator of brain-like computation in artificial neural networks. Furthermore, RSA might overemphasize the importance of the similarity-based organization in the comparison between different networks. In the future, consideration should also be given to the similarity metric used to compare RDMs in order to avoid conflict between similarity score and qualitative inspection of RDMs. 


## Evolution of RDMs during training reveals that ANNs don’t learn like babies

The final question I asked in this project was whether CNNs develop in a similar fashion to infants during the course of training. To test this hypothesis, I generated RDMs of the final convolutional layer of each network for each epoch of training and considered the results with respect to established literature on semantic development in infants. I chose to analyze the final convolutional layer because it has been shown to bear the closest resemblance to higher brain areas like V4 and IT. 
    
| ![GD_gif](https://giphy.com/gifs/fHS2no35pufd6430Ub)  | ![Burstprop_gif](https://giphy.com/gifs/LwxNPIdpAZk4hZ907d) | ![Hebbian_gif](https://giphy.com/gifs/S1WKdQ9oiyg5jEWWne) |
| ------------- | ------------- | ------------- |

In each animation, one can observe how each learning rule gives rise to different organizations of similar stimuli (henceforth “similarity structure”) over the course of training. Interestingly, it appears that none of the learning rules promotes a discernible trend in similarity structure during training. These seemingly random fluctuations are not consistent with established literature on semantic development in infants, which suggests that infants develop general distinctions between objects before discovering finer-grained idiosyncrasies. Specifically, the Mandler et al. and Pauen have shown that infants can discriminate animals from furniture at 7-9 months of age, and only later make finer-grained distinctions [16](https://www.sciencedirect.com/science/article/abs/pii/001002859190011C), [17](https://srcd.onlinelibrary.wiley.com/doi/10.1111/1467-8624.00454). Additionally, Keil observed that elementary school children gain the ability to more precisely match predicates with nouns as they age [18](https://www.hup.harvard.edu/catalog.php?isbn=9780674181816). 
    
The surprisingly random trajectory of the CNN representations under each learning rule calls to question the utility of artificial neural networks (ANNs) as models for their biological counterparts. Furthermore, it is possible that optimizing ANNs for predictive performance may stray them away from biological similarity. However, that’s a bit hard to believe given Yamins’ thorough analysis demonstrating that predictive performance strongly correlates with similarity to neural data [1](https://www.pnas.org/content/111/23/8619). Additionally, McLelland and Rogers performed the same experiment in their 2004 book, and showed that ANNs do indeed develop general category distinctions and subsequently finer-grained representations [19](http://nwkpsych.rutgers.edu/~jose/courses/578_mem_learn/2012/readings/Rogers_McClelland_2003.pdf). Khaligh-Razavi and Kriegeskorte also demonstrated that a supervised CNN can develop a semantically organized representation for solid color images [20](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003915#s3).   
    
Perhaps rather than the present results being a limitation of ANNs, it is possible that ImageNet wasn’t a suitable dataset for revealing semantically organized, hierarchical representations. First, manual inspections of ImageNet images show that categories are not all well organized. For example, both computer “mice” and animal mice are both present in the “mouse” category. Additionally, the categories selected as stimuli for the fMRI dataset used were not hierarchically organized. Instead, the only clear category distinction existed between animals and inanimate objects, but this was “confounded” by the presence of animals in inanimate images (e.g. a human holding a screwdriver). 

# If I had unlimited budget and time, I would…

## Use a different dataset

On that note, I think a new hierarchical image dataset is worth considering for deep learning-driven neuroscience, and I’m not the first to raise this issue [21](http://www.cnbc.cmu.edu/~tai/microns_papers/Barlow-SensoryCommunication-1961.pdf). I also would have liked a dataset that included activity from the inferior temporal cortex, which is well known to be the brain area that most-explicitly encodes a semantically-organized representation of objects. 

## Try to model lower regions of the visual cortex

Next, I would develop an unsupervised learning rule to generate statistically independent representations. This is motivated by two observations: (1) unsupervised Hebbian learning trained a CNN whose layers were most similar to early layers of the visual cortex, and (2) there is plenty of evidence demonstrating that the lower regions of the visual system (from retina to V2/3) decorrelate visual features through the means of “efficient-coding [21]”. That is, representations in the early visual cortex appear to maximize the information within each receptive field. While previous studies have shown that unsupervised ANNs that optimize for statistical independence can match well to neural data, I haven’t found any that compare unsupervised “ICA-like” CNNs to vanilla supervised CNNs in their similarity to brain activity. Currently, Ganguli’s group has provided a theoretical understanding of how efficient-coding emerges in early layers of supervised CNNs [22](https://arxiv.org/abs/1901.00945), and Wiskott’s group is in the process of developing an “ICA” network that can handle high-dimensional data [23](https://arxiv.org/abs/1904.09858?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29). Higgins at DeepMind has also used a variational autoencoder for unsupervised maximization of statistical independence [24](https://arxiv.org/abs/1606.05579), and Pehlevan and Chklovskii have developed a fully-connected “ICA” network [25](https://arxiv.org/abs/1908.01867). Perhaps future work can explore whether multiple learning rules are present in different areas of the ventral stream (and I don’t think this would surprise anybody). 

## Monitor the brain during learning
    
As far as V4 and IT go, perhaps an interesting future experiment would be to infer learning rules from human neural activity over the course of training. For example, as an adult human learns to discriminate between colors that are only discerned by a trained eye [26](https://languagelog.ldc.upenn.edu/nll/?p=17970). In this sort of psychophysics experiment - where expert level object recognition is developed - I would be curious to observe how the dynamics of the visual system change over time. (Surprisingly, I couldn’t find any such experiments, but will have to ask some folks that are more well read in this area.) A concurrent exercise could be to discover backpropagation and gradient descent within an ANN during training. Perhaps the tool used to solve such a problem can also extract a learning rule from neural data.

## Consider other tools for comparing representations

Finally, I am now curious as to whether representational similarity analysis is the best game in town. What if the observed semantic organization of stimuli in the brain is an epiphenomenon? What if measuring the relative dissimilarity of different outputs of a network overlooks other important attributes? Perhaps future analytical tools that seek to capture brain (or ANN) representations should observe the effect of perturbing inferred representations on brain function and/or behavior. 
 
# Conclusions

Representational similarity analysis (RSA) revealed that burst-dependent synaptic plasticity managed to train a CNN that was pretty close to a CNN trained by gradient descent in similarity to neural data. Hebbian learning trained a network that was more similar to early layers of the visual cortex but progressively decreased in similarity to higher regions. Qualitative inspection of representational dissimilarity matrices generated by RSA showed that none of the learning rules appeared to semantically organize objects in a brain-like fashion. Additionally, the evolution of network representations during training revealed that none of the learning rules promote brain-like semantic development. 