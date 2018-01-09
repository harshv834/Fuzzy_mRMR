import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from skfeature.function.information_theoretical_based import MRMR
from scipy.special import expit
from sklearn.cross_validation import cross_val_predict
from sklearn import svm
import sys
import scipy.io
import math
from sklearn.cross_validation import train_test_split


def select(Xtr, ytr, c_labels,n_clus,k):
	Score_temp = np.zeros((Xtr.shape[1],n_clus));
	idx = np.zeros((Xtr.shape[1],n_clus));

	
	for i in range(n_clus):
		arr1 = (c_labels == i);
		X1 = Xtr[arr1];
		y1 = ytr[arr1];
		idx1 ,Score1,_  = MRMR.mrmr(X1,y1,n_selected_features = k);
		Score_temp[idx1,i] = Score1;
		idx[idx1,i]=1;	
	features = idx.sum(axis=-1) > 0;
	num_sel = features.sum();
	Score = Score_temp[features];
	return Score,features;
			
	

def main():
	
	max_iter = int(sys.argv[4]);
	acc = 0;
	acc_ts =0;
	acc1_ts =0 ;
	acc2_ts =0;
	acc1 = 0;
	acc2 = 0;
	avg_feature = 0;
	n_clus = int(sys.argv[1]);
	k = int(sys.argv[2]);
	dataset = str(sys.argv[3]);
	
	data = scipy.io.loadmat(dataset);
	X = data['X'];
	X = X.astype(float);
	y = data['Y'];
	y = y[:,0];

	n ,d = X.shape;
	classes = np.unique(y);
	n_class = len(classes);

	for i in range(max_iter):
		print("%f - th iteration" %(i+1));
		Xtr,Xts,ytr,yts = train_test_split(X,y,test_size = 0.2, random_state = 42);	
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
		

		Scores, features = select(Xtr, ytr, c_labels,n_clus,k);
		Xtrnew = Xtr[:,features];
		Xtsnew = Xts[:,features];

		fwtr = expit(distr.dot(Scores.transpose()));
		fwts = expit(dists.dot(Scores.transpose()));
		print(np.where(fwts>0.6,1,0).sum());
		print(np.where(fwtr>0.6,1,0).sum());
		Xtrnew = np.multiply(Xtrnew,fwtr);
		Xtsnew = np.multiply(Xtsnew,fwts);
	

	
		clf = svm.SVC(kernel = 'linear', C=0.5);
		ynew = cross_val_predict(clf,Xtrnew,ytr, cv = 4);
		clf1 = svm.SVC(kernel = 'linear', C=0.5);
		y1 = cross_val_predict(clf1,Xtr,ytr, cv = 4);	


		num_fea = Xtrnew.shape[1];
		avg_feature = avg_feature + num_fea;
		id1,_,_ = MRMR.mrmr(Xtr,ytr, n_selected_features = num_fea);
		Xtr2 = Xtr[:,id1[0:num_fea]];
		Xts2 = Xts[:,id1[0:num_fea]];
		clf2 = svm.SVC(kernel ='linear', C=0.5);
		y2 = cross_val_predict(clf2,Xtr2,ytr, cv = 4);
		acc = acc + metrics.accuracy_score(ytr,ynew);
		acc1 = acc1 + metrics.accuracy_score(ytr,y1);
		acc2 = acc2 + metrics.accuracy_score(ytr,y2);
		clf.fit(Xtrnew,ytr);
		clf1.fit(Xtr,ytr);
		clf2.fit(Xtr2,ytr);
		acc_ts = acc_ts + clf.score(Xtsnew,yts);
		acc1_ts = acc1_ts + clf1.score(Xts,yts);
		acc2_ts = acc2_ts + clf2.score(Xts2,yts);

	print ("Original Dataset Size %f %f" %(X.shape))
	print ("Average number of features after selection : %f" % (avg_feature*1.0/max_iter));
	print ("Accuracy after Selection :Train Set: %f"%(acc*100.0/max_iter));
	print("Test Set : %f"% (acc_ts*100.0/max_iter) );
	print ("Accuracy with all features :Train Set : %f "%(acc1*100.0/max_iter));
	print ("Test Set: %f" %(acc1_ts*100.0/max_iter));
	print ("Accuracy by mRMR for same number of features: Train Set: %f" %(acc2*100.0/max_iter))
	print("test Seet :%f"%(acc2_ts*100.0/max_iter));








if __name__ == '__main__':
	main()	










