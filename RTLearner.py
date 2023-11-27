import numpy as np  
from scipy import stats

class RTLearner(object):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    This is a Random Tree Regression Learner. It is implemented correctly.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size = 1, verbose=False):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   	
        self.leaf_size = leaf_size
        self.vrbose = verbose
           		  	   		  		 		  		  		    	 		 		   		 		  
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
        data_y = np.array([data_y])
        data = np.append(data_x, data_y.T, axis=1)
        self.tree = self.build_tree(data)

    def build_tree(self, data):
        data_y = data[:, -1]
        if data.shape[0] == 1 or data.shape[0] <= self.leaf_size:
            return np.array([['leaf', stats.mode(data_y)[0][0], None, None]])
        elif np.all(data_y == data_y[0]):
            return np.array([['leaf', data_y[0], None, None]])
        else: 
            random_feature = np.random.randint(data.shape[1]-1)
            splitVal = np.median(data[:, random_feature])
            maximum_value = max(data[:, random_feature])
            if maximum_value == splitVal:
                return np.array([['leaf', stats.mode(data[:, -1])[0][0], None, None]])
            
            left_tree = self.build_tree(data[data[:, random_feature] <= splitVal])
            right_tree = self.build_tree(data[data[:, random_feature] > splitVal])
            root = np.array([[random_feature, splitVal, 1, left_tree.shape[0] + 1]])
            return np.append(root, np.append(left_tree, right_tree, axis=0), axis=0)      	  		 		  		  		    	 		 		   		 		       
  		  	   		  		 		  		  		    	 		 		   		 		  
    def query(self, features):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		  		 		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		  		 		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		 
        predicted_value = []
        for feature in features:
            row = 0
            node = self.tree[row,0]
            while (node != "leaf"): # if it is not a leaf node, enter loop
                splitVal = self.tree[row, 1]
                left = int(self.tree[row, 2])
                right = int(self.tree[row, 3])
                if(feature[int(node)] <= splitVal):
                    row = row + left
                else:
                    row = row + right
                node = self.tree[row,0]
            predicted_value.append(self.tree[row,1])
        return predicted_value
    	    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    print("Running RT Learner")  	