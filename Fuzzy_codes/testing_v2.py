from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from LCSI_new import lcsi
from scipy.special import expit
from sklearn import svm
import sys
import scipy.io
import math
from sklearn.model_selection import StratifiedKFold
from skfeature.utility.entropy_estimators import *
from skfeature.function.information_theoretical_based import MRMR


#This script contains class-weighted SVM, stratified K-fold, k-best compatible code
#Also, this includes model selection
def select(Xtr, ytr, c_labels,n_clus,k):
	Score_temp = np.zeros((Xtr.shape[1],n_clus));
	idx = np.zeros((Xtr.shape[1],n_clus));	
	for i in range(n_clus):
		arr1 = (c_labels == i);
		X1 = Xtr[arr1];
		y1 = ytr[arr1];
		idx1,Score1 = lcsi(X1,y1,n_selected_features = k,t_best = 10,function_name='MRMR');
		Score_temp[:,i] = (idx1.T).dot(Score1).reshape(Score_temp[:,i].shape)
		idx[:,i] = Score1.sum(axis=0)
	features = idx.sum(axis=-1) > 0;
	num_sel = features.sum();
	Score = Score_temp[features];
	return Score,features;
			
	

def main():
	max_iter = int(sys.argv[3]);
	clus_max = int(sys.argv[1]);
	dataset = str(sys.argv[2]);
	var = float(sys.argv[4]);
	k = int(sys.argv[5])
	j = 0;	
	data = scipy.io.loadmat(dataset);
	X = data['X'];
	X = X.astype(float);
	y = data['Y'];
	y = y[:,0];
	n ,d = X.shape;
	classes = np.unique(y);
	n_class = len(classes);
	s_scores = np.zeros((clus_max,1));
	for i in range(clus_max):
		kmeans = KMeans(init ='k-means++',n_clusters = i+2, n_init = 10);
		labels = kmeans.fit_predict(X);
		s_scores[i] = metrics.silhouette_score(X, labels);

	n_clus = np.argmax(s_scores) + 2;
	ss = StratifiedKFold(n_splits = max_iter, shuffle =True)
	clf = svm.SVC()
	parameters  ={'kernel':('linear','rbf'),'C':[0.01,0.1,1,10,100]}
	grid_search=GridSearchCV(clf,param_grid=parameters)

	acc =0 
	acc1 = 0
	acc2 = 0
	acc_ts =0 
	acc1_ts = 0
	acc2_ts = 0
	avg_feature = 0
	redundant_features = 0
	un,inv,cnts = np.unique(y,return_inverse=True,return_counts=True)
	un = 1./cnts
	un = un/min(un)
	cl_wts = un[inv]





	j=0
	for train, test in ss.split(X,y):
		print("%f - th iteration" %(j+1));
		Xtr,Xts,ytr,yts,cl_tr,cl_ts = X[train],X[test],y[train],y[test],cl_wts[train],cl_wts[test];		
		ntr = Xtr.shape[0];
		nts = Xts.shape[0];

		kmeans = KMeans(init = 'k-means++',n_clusters = n_clus, n_init =10).fit(Xtr);
		distr = kmeans.transform(Xtr);
		dists = kmeans.transform(Xts);
		distr = distr + 0.0001;
		dists = dists + 0.0001;
		one = np.ones(distr.shape);
		one1 = np.ones(dists.shape);
		distr = np.divide(one,distr);
		dists = np.divide(one1,dists);
		c_labels = kmeans.predict(Xtr);
		c_labelst = kmeans.predict(Xts);
		inv_dis_max = max(distr.max(),dists.max());
		distr = distr/inv_dis_max;
		dists = dists/inv_dis_max;
		
		Scores, features = select(Xtr, ytr, c_labels,n_clus,k);
		Xtrnew = Xtr[:,features];
		Xtsnew = Xts[:,features];
		for l in range(Xtrnew.shape[1]):
			for m in range(l):
				if(midd(Xtrnew[:,l],Xtrnew[:,m])>1):
					redundant_features=redundant_features+1;

		fwtr = expit(var*distr.dot(Scores.transpose()));
		fwts = expit(var*dists.dot(Scores.transpose()));
		
		Xtrnew = np.multiply(Xtrnew,fwtr);
		Xtsnew = np.multiply(Xtsnew,fwts);
	
		ynew = np.zeros((ytr.shape));
		ysnew = np.zeros((yts.shape));
		for i in range(n_clus):
			arr1 = (c_labels == i);
			arr2 = (c_labelst == i);
			Xtrnewj = Xtrnew[arr1];
			Xtsnewj = Xtsnew[arr2];
			ytrj = ytr[arr1];			
			grid_search.fit(Xtrnewj,ytrj);
			ynew[arr1] = grid_search.predict(Xtrnewj);
			ysnew[arr2] = grid_search.predict(Xtsnewj);
		

		grid_search.fit(Xtr,ytr);
		y1 = grid_search.predict(Xtr);
		y1s = grid_search.predict(Xts);	


		num_fea = Xtrnew.shape[1]; 
		avg_feature[k] = avg_feature[k] + num_fea;
		id1,_,_ = MRMR.mrmr(Xtr,ytr, n_selected_features = num_fea);
		Xtr2 = Xtr[:,id1[0:num_fea]];
		Xts2 = Xts[:,id1[0:num_fea]];
		grid_search.fit(Xtr2,ytr);
		y2 = grid_search.predict(Xtr2);
		y2s = grid_search.predict(Xts2);


		acc= acc + metrics.accuracy_score(ytr,ynew);
		acc1 = acc1 + metrics.accuracy_score(ytr,y1);
		acc2 = acc2 + metrics.accuracy_score(ytr,y2);
		acc_ts = acc_ts + metrics.accuracy_score(ysnew,yts);
		acc1_ts = acc1_ts + metrics.accuracy_score(y1s,yts);
		acc2_ts = acc2_ts + metrics.accuracy_score(y2s,yts);
		j = j +1;	

	avg_feature = avg_feature*1.0/max_iter;
	redundant_features = redundant_features*1.0/max_iter;
	acc = acc*100.0/max_iter;
	acc1 = acc1*100.0/max_iter;
	acc2 = acc2*100.0/max_iter;
	acc_ts = acc_ts*100.0/max_iter;
	acc1_ts = acc1_ts*100.0/max_iter;
	acc2_ts = acc2_ts*100.0/max_iter;
	

	
	print ("Original Dataset Size %f %f" %(X.shape))
	print ("Average number of features after selection : " );
	print avg_feature;
	print ("Number of clusters :");
	print n_clus;
	print ("Redundant_feature_pairs :");
	print avg_feature;
	print ("Accuracy after Selection :Train Set:");
	print acc;
	print("Test Set :");
	print(acc_ts);
	print ("Accuracy with all features :Train Set ");
	print (acc1);
	print ("Test Set: ");
	print (acc1_ts);
	print ("Accuracy by mRMR for same number of features: Train Set");
	print (acc2);
	print("test Seet :");
	print (acc2_ts);

	scipy.io.savemat(dataset+'.mat',{'acc' :acc, 'acc1': acc1, 'acc2':acc2,'acc_ts':acc_ts,'acc1_ts':acc1_ts,'acc2_ts':acc2_ts,'avg_feature':avg_feature,'redundant_features':redundant_features});






if __name__ == '__main__':
	main()	










