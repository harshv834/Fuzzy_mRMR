from skfeature.utility.entropy_estimators import *
#Modified LCSI code from skfeature to output K-best feature sets with a greedy strategy.
import numpy as np
from scipy.io import loadmat
import heapq


def lcsi(X, y, **kwargs):
	"""
This function implements the basic scoring criteria for linear combination of shannon information term.
The scoring criteria is calculated based on the formula j_cmi=I(f;y)-beta*sum_j(I(fj;f))+gamma*sum(I(fj;f|y))

Input
-----
X: {numpy array}, shape (n_samples, n_features)
input data, guaranteed to be a discrete data matrix
y: {numpy array}, shape (n_samples,)
input class labels
kwargs: {dictionary}
Parameters for different feature selection algorithms.
beta: {float}
beta is the parameter in j_cmi=I(f;y)-beta*sum(I(fj;f))+gamma*sum(I(fj;f|y))
gamma: {float}
gamma is the parameter in j_cmi=I(f;y)-beta*sum(I(fj;f))+gamma*sum(I(fj;f|y))
function_name: {string}
name of the feature selection function
n_selected_features: {int}
number of features to select

Output
------
F: {numpy array}, shape: (n_features,)
index of selected features, F[0] is the most important feature
J_CMI: {numpy array}, shape: (n_features,)
corresponding objective function value of selected features
MIfy: {numpy array}, shape: (n_features,)
corresponding mutual information between selected features and response

Reference
---------
Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
	"""

	n_samples, n_features = X.shape
	beta = 0.0
	gamma = 0.0		
	
	# index of selected features, initialized to be empty
	# Objective function value for selected features
	# Mutual information between feature and respons
	# indicate whether the user specifies the number of features
	is_n_selected_features_specified = False
	# initialize the parameters
	if 'beta' in kwargs.keys():
		beta = kwargs['beta']
	if 'gamma' in kwargs.keys():
		gamma = kwargs['gamma']
	if 'n_selected_features' in kwargs.keys():
		n_selected_features = kwargs['n_selected_features']
	       	is_n_selected_features_specified = True
		n_done=0
	if 't_best' in kwargs.keys():
		t_best = kwargs['t_best']
	J = np.zeros((t_best,1))
	F = np.zeros((t_best,n_features))
    # select the feature whose j_cmi is the largest
    # t1 stores I(f;y) for each feature f
	t1 = np.zeros((n_features))
    # t2 stores sum_j(I(fj;f)) for each feature f
	t2 = np.zeros((n_features,n_features))
    # t3 stores sum_j(I(fj;f|y)) for each feature f
	t3 = np.zeros((n_features,n_features))
	for i in range(n_features):
		f = X[:, i]
		t1[i] = midd(f, y)
		for j in range(n_features):
			t2[i][j] = midd(f,X[:, j])
			if kwargs['function_name']=='JMI': 
				t3[i][j] = cmidd(f,X[:, j],y)		   	

    # make sure that j_cmi is positive at the very beginning
	j_cmi = 1

	while True:
		if n_done== 0:
            # select the feature whose mutual information is the largest
			idx = heapq.nlargest(t_best,range(len(t1)),t1.__getitem__)
			F[np.arange(len(F)),idx] = 1;
			J = t1[idx];
			n_done = n_done +1	
		if is_n_selected_features_specified:
			print F.sum()
			print n_done
			if n_done == n_selected_features:
				break
			else:
				if j_cmi < 0:
					break

        # we assign an extreme small value to j_cmi to ensure it is smaller than all possible values of j_cmi
		t = np.ones((n_features,n_features));
		t = -1E30*t
		index = np.zeros((t_best,t_best));
		J_temp = np.zeros((t_best*t_best,1));
		if 'function_name' in kwargs.keys():
			if kwargs['function_name'] == 'MRMR':
				beta = t_best / float(F.sum())
			elif kwargs['function_name'] == 'JMI':
				beta = t_best / float(F.sum())
				gamma = t_best / float(F.sum())
		for j in range(t_best):
			for i in range(n_features):
                # calculate j_cmi for feature i (not in F)
				if (F[j][i] == 0):
					if gamma==0:
						t[j][i] = t1[i] - beta*(t2[i][F[j]==1].sum())
					else:
						t[j][i] = t1[i] - beta*(t[2][F[j]==1].sum()) + gamma*(t3[i][F[j]==1].sum())
                # record the largest j_cmi and the corresponding feature index
			index[j] = heapq.nlargest(t_best,range(len(t[j])),t[j].__getitem__)
			J_temp[j*t_best:(j+1)*t_best] = (J[j] + t[j][(index[j]).astype(int)]).reshape(t_best,1)
	    	
		new_ind = heapq.nlargest(t_best,range(len(J_temp)),J_temp.__getitem__)
		temp_F = np.zeros((t_best,n_features))
		ind1,ind2 = np.unravel_index(new_ind,(t_best,t_best))
		temp_F[range(t_best)] = F[ind1]
		temp_F[range(t_best),(index[ind1,ind2]).astype(int)] = 1
		F = temp_F
		J = J_temp[new_ind]
		n_done = n_done+1		

	return F, J












