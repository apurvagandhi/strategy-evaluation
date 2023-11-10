import random
import numpy as np  

class BagLearner(object):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    This is a Bag Regression Learner. It is implemented correctly.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    def __init__(self, learner, kwargs = {}, bags = 20, boost = False, verbose = False):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	
        # The learner points to the learning class that will be used in the BagLearner   	
        self.learner = learner
        #  keyword arguments that are passed on to the learner’s constructor and they can vary according to the learner
        self.kwargs = kwargs
        # number of learners you should train using Bootstrap Aggregation. 
        self.bags = bags
        self.boost = boost
        # generate output if true
        self.vrbose = verbose
        # creating an instance of a learner object using the keyword arguments provided in kwargs 
        # and then appending that learner object to the learners list.
        self.learners = []
        for i in range(bags):
            self.learners.append(self.learner(**self.kwargs))	   		
          		 		  		  		    	 		 		   		 		  
    def author(self):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		  		 		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        return "agandhi301"  		  	   		  		 		  		  		    	 		 		   		 		  

    def add_evidence(self, data_x, data_y):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		  		 		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		  		 		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
        """
        for learner in self.learners:
            # Generate a list of random numbers between 0 and the number of columns in a 2D array
            random_numbers = [random.randint(0, data_x.shape[0] - 1) for _ in range(data_x.shape[0])]
            # Get the bag x data as per the randomized numbers
            self.bag_x = data_x[random_numbers]
            self.bag_y = data_y[random_numbers]
            # call learner with random x bag and random y bag data
            learner.add_evidence(self.bag_x, self.bag_y)
       	 		  		  		    	 		 		   		 		       	  		 		  		  		    	 		 		   		 		  
    def query(self, points):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		  		 		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		  		 		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		 
        y_pred_results = []
        # For each learner in the self.learners collection, it calls the query method of that learner with the provided points.
        # The results of these queries are collected into a list and mean of that list is returned.
        for learner in self.learners:
            y_pred_results.append(learner.query(points))
            
        return np.mean(y_pred_results, axis = 0)
    	    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    print("Running Bag Learner")  		  	   		  		 		  		  		    	 		 		   		 		  
