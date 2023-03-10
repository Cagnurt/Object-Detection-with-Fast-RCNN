# ML Challenge

# Problem Definition

Let’s make the problem **SMART**
**S**pecific: The model should be able to accurately track different types of nuts and bolts, even under varying lighting conditions, and keep count of them.
**M**easurable: Defining end-2-end pipeline with decreasing loss values over iterations for train and validation. (Because it is my first attempt to solve this problem, this success definition is meaningful. For further iteration, this part should be modified like that: maximizing accuracy subject to <0.5sec CPU inference time)
**A**ctionable: Using simple and familiar methods for you. (since it is an initial attempt)
**R**elevant: High accuracy for customer 
**T**ime-bound: it should be implemented in ~10h

## Constraints

Time. Other constraints such as hardware specs are not important for this first attempt as long as it can be implementable and fast enough for compilation. 

# Pipeline

Although my pipeline is visualized in Figure-1, I could not fully implement this.  For example, for preparing the dataset part,  I forgot to extract images without target values.

![Figure - 1: Stroma Challange Pipeline
](readme_imgs/pipeline.png)

Figure - 1: Stroma Challange Pipeline

# Result

In end2end.ipynb, I tried to convey my idea with my programming overall. Although it is not working properly now, with creating issues on GitHub, it can improve systematically.