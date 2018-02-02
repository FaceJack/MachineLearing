//writen by WangJin 2018/1/6 
//k近邻算法
//小目标：开始学习使用Eign库处理矩阵和向量

#include <iostream>
#include <vector>
#include <map>
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

template <typename T>
class kNN{
private:
	MatrixXf group; //使用动态矩阵表示数据集
	vector<T> label; //使用动态向量表示数据集标签

public:
	//构造函数
	kNN(vector<vector<float>> dataSet, vector<T> labels){
		int row = dataSet.size(), col=dataSet[0].size();
		group.resize(row, col);
		label.resize(labels.size());
		//赋值操作
		if (row != labels.size())
			cout << "the input data is error, please check the size of the dataSet and the labels" << endl;
		for (int i = 0; i < row;i++){
			label[i] = labels[i];
			for (int j = 0; j < col; j++)
				group(i, j) = dataSet[i][j];
		}
	}
	
	//k-近邻算法的实现对输入数据的分类
	T classify(vector<float> inX, int k){
		int dataSetSize = group.rows(); //数据集的行数
		//将输入数据inX重复dataSetSize
		MatrixXf diffMat;
		diffMat.resize(group.rows(), group.cols());
		for (int i = 0; i < dataSetSize; i++)
			for (int j = 0; j < inX.size(); j++)
				diffMat(i, j) = inX[j];
		
		//获得输入数据和数据集中元素之间的差值
		diffMat = diffMat - group;
		cout << diffMat << endl;
		//对diffMat矩阵中的每一个元素平方
		diffMat = diffMat.array().abs2();
		cout << diffMat << endl;
		//对diffMat矩阵的所有行求和
		MatrixXf sqDiffMat = diffMat.rowwise().sum();
		cout << sqDiffMat << endl;
		//对sqDiffMat矩阵中的每一个元素开平方
		sqDiffMat = sqDiffMat.array().sqrt();
		cout << sqDiffMat << endl;
		//对sqDiffMat矩阵中的元素从小到大进行排序，找出其中的前k个，并返回其前k个元素的索引
		vector<int> sortedIndex = argsort(sqDiffMat,k);
		for (int i = 0; i < k; i++)
			cout << sortedIndex[i] << " ";
		cout << endl;
		//对索引对应的k个标签进行计算，选择标签出现次数最多的返回
		return sortLabels(sortedIndex);
	}

private:
	//对矩阵进行从小到大的排序，找出前k个元素，并返回他们的索引
	//这里引入multimap处理
	vector<int> argsort(MatrixXf sqDiffMat, int k){
		vector<int> index,indexAll;
		multimap<float, int> mmp;
		for (int i = 0; i < sqDiffMat.size(); i++){
			mmp.insert(pair<float, int>(sqDiffMat(i), i));
		}
		for (multimap<float, int>::iterator itr = mmp.begin(); itr != mmp.end(); ++itr)
			indexAll.push_back((*itr).second);
		for (int i = 0; i < k; i++)
			index.push_back(indexAll[i]);
		return index;
	}

	//返回出现次数最多的标签
	T sortLabels(vector<int> sortedIndex){
		//将label标签存入到map中:如果键值key存在，对应的value加1；如果键值不存在，对应的value值置1
		map<T, int> mp;
		map<T, int>::iterator itr;
		for (int i = 0; i < sortedIndex.size(); i++){
			itr = mp.find(label[sortedIndex[i]]);
			//如果键值key存在，对应的value加1
			if (itr != mp.end()){
				itr->second += 1;
			}
			//如果键值不存在，对应的value值置1
			else{
				mp.insert(pair<T, int>(label[sortedIndex[i]], 1));
			}
		}

		//遍历一遍map，返回其中最大的value值
		map<T, int>::iterator it; //用于记录最大value的迭代器
		int maxValue = 1;
		for (itr = mp.begin(); itr != mp.end(); itr++){
			if (itr->second > maxValue){
				maxValue = itr->second;
				it = itr;
			}
		}
		return it->first;
	}
};