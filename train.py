from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer,HashingVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import csv
from rawJob import rawJob
import numpy as np
from time import time,sleep
import os
from itertools import izip
import json
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, vstack, csr_matrix, lil_matrix
import json

def LoadData(filePath):
	reader = csv.reader(open(filePath))
	jobList = []
	for i,row in enumerate(reader):
		if i > 0:
			jobList.append(rawJob(row))
	return jobList

def getTargets(data,start,end):
	targets = []
	for i in xrange(start,end):
		job = jobData[i].data
		targets.append(np.log(float(job['salaryNormalized'])))
	return np.array(targets)

def getTests(data,start,end,vect,vect2):
	targets = []
	desTests = []
	titleTests = []
	for i in xrange(start,end):
		job = data[i].data
		desTests.append(job['description'] + ' ' + job['company'] + ' ' + job['normalizedLocation'])
		titleTests.append(job['title'])
	desTests = vect.transform(desTests)
	titleTests = vect2.transform(titleTests)
	tests = hstack((desTests,titleTests))
	return tests

def loadJobs(filePath,trainNum,testNum):
	jobs = []
	jobFile = open(filePath)
	for i,job in enumerate(jobFile):
		if i < trainNum+testNum:
			jobs.append(json.loads(job))
	return jobs

def fit(des,titles,sals,clf,alpha):
	tRidge = time()
	vect = TfidfVectorizer(min_df=1,ngram_range=(1,3),max_features=24000000)
	vect2 = TfidfVectorizer(min_df=1,ngram_range=(1,3),max_features=24000000)
	des = vect.fit_transform(des)
	titles = vect2.fit_transform(titles)
	merged = hstack((des,titles))
	print 'Vectorized job data in',time()-tRidge,'seconds'
	if clf == 'ridge':
		rr = linear_model.Ridge(alpha= alpha)
		rr.fit(merged,sals)
	print 'fitting done in',time()-tRidge,'seconds'
	return vect,vect2,rr

def getFittingData(jobs):
	sals = []
	des = []
	titles = []
	for i in xrange(trainNum):
		job = jobs[i].data
		sals.append(np.log(float(job['salaryNormalized'])))
		des.append(job['description'] + ' ' + job['company'] + ' ' + job['normalizedLocation'])
		titles.append(job['title'])
	return sals,des,titles

paths = json.load(open('SETTINGS.json','rb'))
t0 = time()
print 'loading data from',paths['TRAIN_DATA_PATH']
jobs = LoadData(paths['TRAIN_DATA_PATH'])
trainNum = len(jobs)
print 'formatting data'
sals,des,titles = getFittingData(jobs)
print 'fitting vectorizers and model'
vect,vect2,rr = fit(des,titles,sals,clf='ridge',alpha=0.035)
print 'serializing model as',paths['MODEL_PATH']
joblib.dump(rr,paths['MODEL_PATH'],compress=3)
joblib.dump(vect,paths['VECTORIZER_1_PATH'],compress=3)
joblib.dump(vect2,paths['VECTORIZER_2_PATH'],compress=3)

