// writen by WangJin 2018/1/10
// 目标：实现支持向量机
// 支持向量机的头文件

/*
用支持向量机求解二分类问题
分离超平面为：w'·x+b=0
分类决策函数:f(x)=sign(w'·x+b)
*/

#include "svm.h"
#include <ctime>

SVM::SVM(vector<vector<float>> dataSet, vector<float> labels, float C, float toler){
	m = dataSet.size();
	n = dataSet[0].size();
	dataMatIn.resize(m, n);
	classLabels.resize(m, 1);
	for (int i = 0; i < m; i++){
		classLabels(i, 0) = labels[i];
		for (int j = 0; j < n; j++){
			dataMatIn(i, j) = dataSet[i][j];
		}
	}
	this->C = C;
	this->toler = toler;
	b = 0;
	alpha.resize(m, 1);
	alpha.setZero();
	eCache.resize(m, 2);
	eCache.setZero();
	srand((unsigned)time(NULL)); //为了后续产生随机数
}

float SVM::calcEk(int k){
	//float fXk = (float)((alpha.array()*classLabels.array()).transpose()*(dataMatIn*dataMatIn.row(k).transpose())) + b;
	MatrixXf middleX = (alpha.array()*classLabels.array()).transpose();
	MatrixXf middleY = dataMatIn*dataMatIn.row(k).transpose();
	float fXk = (float)((middleX*middleY).sum()) + b;
	float Ek = fXk - (float)classLabels(k, 0);
	return Ek;
}

int SVM::selectJrand(int i, int m){
	int j = i;
	//srand((unsigned)time(NULL));
	while (j == i)
		j = (int)rand() % m;
	return j;
}

float SVM::clipAlpha(float aj, float H, float L){
	if (aj > H)
		aj = H;
	if (aj < L)
		aj = L;
	return aj;
}

pair<int, float> SVM::selectJ(int i, float Ei){
	pair<int, float> retResult;
	int maxK = -1;
	float maxDeltaE = 0, Ej = 0;
	//查找出缓存eCache中不为0的索引
	vector<int> noZeroIndex; //不为0的下标索引值
	eCache(i, 0) = 1;
	eCache(i, 1) = Ei;
	for (int in = 0; in < eCache.rows();in++)
		if (eCache(in, 0) != 0)
			noZeroIndex.push_back(in);
	if (noZeroIndex.size()>1){
		for (int k = 0; k < noZeroIndex.size(); k++){
			if (noZeroIndex[k] == i)
				continue;
			float Ek = calcEk(noZeroIndex[k]);
			float deltaE = abs(Ei - Ek);
			if (deltaE>maxDeltaE){
				maxK = noZeroIndex[k];
				maxDeltaE = deltaE;
				Ej = Ek;
			}
		}
		retResult.first = maxK;
		retResult.second = Ej;
		return retResult;
	}
	else{
		int j = selectJrand(i, m);
		Ej = calcEk(j);
		retResult.first = j;
		retResult.second = Ej;
	}
	return retResult;
}

void SVM::updateEk(int k){
	float Ek = calcEk(k);
	eCache(k, 0) = 1;
	eCache(k, 1) = Ek;
}

int SVM::innerL(int i){
	float Ei = calcEk(i);
	if ((classLabels(i, 0)*Ei<-toler&&alpha(i, 0)<C) || (classLabels(i, 0)*Ei>toler&&alpha(i, 0)>0)){
		pair<int, float> decesion = selectJ(i,Ei);
		int j = decesion.first;
		float Ej = decesion.second;
		float alphaIold = alpha(i,0), alphaJold = alpha(j,0);
		float L, H;
		if (classLabels(i, 0) != classLabels(j, 0)){
			L = max((float)0, alpha(j,0) - alpha(i, 0));
			H = min(C, C + alpha(j,0) - alpha(i, 0));
		}
		else{
			L = max((float)0, alpha(j,0) + alpha(i,0) - C);
			H = min(C, alpha(j,0) + alpha(i,0));
		}
		if (L == H){
			cout << "L is equal to H" << endl;
			return 0;
		}
		float eta = (float)(2.0*dataMatIn.row(i)*dataMatIn.row(j).transpose()) - \
			dataMatIn.row(i)*dataMatIn.row(i).transpose() - dataMatIn.row(j)*dataMatIn.row(j).transpose();
		if (eta >= (float)0){
			cout << "the eta is more than zero" << endl;
			return 0;
		}
		alpha(j,0) = alpha(j,0) - classLabels(j,0) * (Ei - Ej) / eta;
		alpha(j,0) = clipAlpha(alpha(j,0), H, L);
		updateEk(j);
		if (abs(alpha(j,0) - alphaJold) < 0.00001){
			cout << "j is not moving enough" << endl;
			return 0;
		}
		alpha(i,0) = alpha(i,0) + classLabels(j,0) * classLabels(i,0) * (alphaJold - classLabels(j,0));
		updateEk(i);
		float b1 = b - Ei - classLabels(i, 0) * (alpha(i, 0) - alphaIold)*(float)(dataMatIn.row(i)*dataMatIn.row(i).transpose())\
			- classLabels(j, 0) * (alpha(j, 0) - alphaJold)*(float)(dataMatIn.row(i)*dataMatIn.row(j).transpose());
		float b2 = b - Ej - classLabels(i, 0) * (alpha(i, 0) - alphaIold)*(float)(dataMatIn.row(i)*dataMatIn.row(j).transpose())\
			- classLabels(j, 0)* (alpha(j, 0) - alphaJold)*(float)(dataMatIn.row(j)*dataMatIn.row(j).transpose());
		if ((alpha(i, 0)>0) && (alpha(i, 0) < C))
			b = b1;
		else if ((alpha(j, 0)>0) && (alpha(j, 0) < C))
			b = b2;
		else
			b = (b1 + b2) / (float)2.0;
		return 1;
	}
	else
		return 0;
}

void SVM::smoP(int maxIter,pair<string,int> kTup){
	int iter = 0;
	bool entireSet = true;
	int alphaPairsChanged = 0;
	while (((iter<maxIter) && (alphaPairsChanged>0)) || (entireSet)){
		alphaPairsChanged = 0;
		if (entireSet){
			for (int i = 0; i < m; i++){
				alphaPairsChanged = alphaPairsChanged + innerL(i);
				cout << "fullset" << " " << "is:" << iter << " ";
				cout << alphaPairsChanged << " pairs changed" << endl;
			}
			iter += 1;
		}
		else{
			//找出alpha中数值在（0，C）之间元素的下标
			vector<int> noZeroIndx;
			for (int i = 0; i < alpha.rows();i++)
				if ((alpha(i, 0)>0) && (alpha(i, 0) < C))
					noZeroIndx.push_back(i);
				for (int j = 0; j < noZeroIndx.size(); j++){
					alphaPairsChanged = alphaPairsChanged + innerL(noZeroIndx[j]);
					cout << alphaPairsChanged << " pairs changed" << endl;
				}
		}
		if (entireSet)
			entireSet = false;
		else if (alphaPairsChanged == 0)
			entireSet = true;
		cout << "iteration numbers is:" << iter << endl;
	}
}

void SVM::showResult(){
	cout << "Aalpha is:" << endl;
	cout << alpha << endl;
	cout << "b is: " << b << endl;
}

MatrixXf SVM::calcWs(){
	MatrixXf w;
	w.resize(n, 1);
	w.setZero();
	for (int i = 0; i < m; i++){
		w = w + alpha(i, 0)*classLabels(i, 0)*dataMatIn.row(i).transpose();
	}
	cout << w << endl;
	return w;
}

int SVM::testWhichClass(MatrixXf testData){
	MatrixXf ws = calcWs();
	float finalRe = (testData*ws).sum() + b;
	if (finalRe>0)
		return 1;
	else
		return -1;
}