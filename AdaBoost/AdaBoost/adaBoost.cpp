//writen by WangJin  2018/1/16

#include "adaBoost.h"

AdaBoost::AdaBoost(vector<vector<float>> dataSet, vector<float> labels){
	m = dataSet.size();
	n = dataSet[0].size();
	dataMatIn.resize(m, n);
	classLabels.resize(m, 1);
	for (int i = 0; i < m; i++){
		classLabels(i, 0) = labels[i];
		for (int j = 0; j < n; j++)
			dataMatIn(i, j) = dataSet[i][j];
	}
}

MatrixXf AdaBoost::stumpClassify(int dimen, float threshVal, string threshIneq){
	MatrixXf retArray;
	retArray.resize(m, 1);
	retArray.setOnes();
	//数组过滤，利用阈值对数据进行分类
	if (threshIneq == "lt"){
		//过滤出第dimen列中数据大于等于threshVal的数据
		for (int i = 0; i < m; i++){
			if (dataMatIn(i, dimen) <= threshVal)
				retArray(i, 0) = -1.0;
		}
	}
	else{
		//过滤出第dimen列中数据小于threshVal的数据
		for (int i = 0; i < m; i++){
			if (dataMatIn(i, dimen) > threshVal)
				retArray(i, 0) = -1.0;
		}
	}
	return retArray;
}

MatrixXf AdaBoost::stumpClassify(MatrixXf testData, int dimen, float threshVal, string threshIneq){
	MatrixXf retArray;
	int testDataM = testData.rows();
	retArray.resize(testDataM, 1);
	retArray.setOnes();
	//数组过滤，利用阈值对数据进行分类
	if (threshIneq == "lt"){
		//过滤出第dimen列中数据大于等于threshVal的数据
		for (int i = 0; i < testDataM; i++){
			if (testData(i, dimen) <= threshVal)
				retArray(i, 0) = -1.0;
		}
	}
	else{
		//过滤出第dimen列中数据小于threshVal的数据
		for (int i = 0; i < testDataM; i++){
			if (testData(i, dimen) > threshVal)
				retArray(i, 0) = -1.0;
		}
	}
	return retArray;
}

float AdaBoost::buildStump(MatrixXf D, MatrixXf &predictedVals, bestStump &retBestStump){
	float numSteps = 10.0;
	float minError = 100000.0;
	MatrixXf bestPreVals;
	for (int i = 0; i < n; i++){
		float rangeMin = dataMatIn.col(i).minCoeff();
		float rangeMax = dataMatIn.col(i).maxCoeff();
		float stepSize = (rangeMax - rangeMin) / numSteps;
		for (int j = -1; j < (int)numSteps + 1; j++){
			for (int k = 0; k < 2; k++){
				string inequal;
				if (k == 0)
					inequal = "lt";
				else
					inequal = "gt";
				float threshVal = rangeMin + ((float)j)*stepSize;
				predictedVals = stumpClassify(i, threshVal, inequal);
				MatrixXf errArr;
				errArr.resize(m, 1);
				errArr.setOnes();

				//判断预测的类型predictedVals和标签对应的类型classLabels是否一致：如果一致将errArr对应位置的元素置0
				for (int index = 0; index < m;index++)
					if (predictedVals(index, 0) == classLabels(index, 0))
						errArr(index, 0) = 0;
				float weightedError = (D.transpose()*errArr).sum();

				//输出调试的信息
				cout << "split: dim " << i << ", thresh " << threshVal << ", thresh inequal" << inequal\
					<< ", the weighted error is:" << weightedError << endl;

				if (weightedError < minError){
					bestPreVals = predictedVals;
					minError = weightedError;
					retBestStump.dim = i;
					retBestStump.inequal = inequal;
					retBestStump.thresh = threshVal;
				}
			}
		}
	}
	predictedVals = bestPreVals;
	return minError;
}

void AdaBoost::adaBoostTrainDS(int numIter){
	MatrixXf D;
	D.resize(m, 1);
	D.setOnes();
	D = D / m;
	MatrixXf aggClassEst;
	aggClassEst.resize(m, 1);
	aggClassEst.setZero();
	for (int i = 0; i < numIter; i++){
		bestStump retBestStump = bestStump();
		MatrixXf predictedVals;
		float error;
		error = buildStump(D, predictedVals, retBestStump);
		cout << "D:" << D.transpose() << endl;
		float alpha = (float)(0.5*log((1.0 - error) / max(error, powf(10,-16))));
		retBestStump.alpha = alpha;
		weakClassArr.push_back(retBestStump);
		cout << "predictedVals:" << predictedVals.transpose() << endl;
		
		//更新D的值
		for (int j = 0; j < m; j++){
			//样本被正确分类时
			if (predictedVals(j, 0) == classLabels(j, 0))
				D(j, 0) = (D(j, 0)*exp(-alpha)) / D.sum();
			else
				D(j, 0) = (D(j, 0)*exp(alpha)) / D.sum();
		}

		//错误率累加计算
		aggClassEst = aggClassEst + alpha*predictedVals;
		cout << "aggClassEst:" << aggClassEst.transpose() << endl;
		int errorCout = 0; //分类错误的样本数目
		for (int j = 0; j < m;j++)
			if (sign(aggClassEst(j, 0)) != classLabels(j, 0))
				errorCout++;
		if (errorCout == 0)
			break;
	}
}

float AdaBoost::sign(float aggClassEs){
	if (aggClassEs > 0)
		return (float)1.0;
	else if (aggClassEs == 0)
		return (float)0.0;
	else
		return (float)-1.0;
}

MatrixXf AdaBoost::adaClassify(vector<vector<float>> dataTest){
	MatrixXf dataTestMat;
	int x = dataTest.size(), y = dataTest[0].size();
	dataTestMat.resize(x, y);
	for (int i = 0; i < x;i++)
		for (int j = 0; j < y; j++)
			dataTestMat(i, j) = dataTest[i][j];
	int dataM = dataTestMat.rows();
	MatrixXf aggClassEst;
	aggClassEst.resize(dataM, 1);
	aggClassEst.setZero();
	for (int i = 0; i < weakClassArr.size(); i++){
		MatrixXf classEst = stumpClassify(dataTestMat,weakClassArr[i].dim,weakClassArr[i].thresh,weakClassArr[i].inequal);
		cout << "classEst:" << classEst.transpose() << endl;
		aggClassEst = aggClassEst + weakClassArr[i].alpha*classEst;
		cout << "aggClassEst:" << aggClassEst.transpose() << endl;
	}
	for (int i = 0; i < dataM; i++)
		aggClassEst(i, 0) = sign(aggClassEst(i, 0));
	cout << aggClassEst << endl;
	return aggClassEst;
}