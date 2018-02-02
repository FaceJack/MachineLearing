//writen by WangJin 2018/1/6 
//k�����㷨
//СĿ�꣺��ʼѧϰʹ��Eign�⴦����������

#include <iostream>
#include <vector>
#include <map>
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

template <typename T>
class kNN{
private:
	MatrixXf group; //ʹ�ö�̬�����ʾ���ݼ�
	vector<T> label; //ʹ�ö�̬������ʾ���ݼ���ǩ

public:
	//���캯��
	kNN(vector<vector<float>> dataSet, vector<T> labels){
		int row = dataSet.size(), col=dataSet[0].size();
		group.resize(row, col);
		label.resize(labels.size());
		//��ֵ����
		if (row != labels.size())
			cout << "the input data is error, please check the size of the dataSet and the labels" << endl;
		for (int i = 0; i < row;i++){
			label[i] = labels[i];
			for (int j = 0; j < col; j++)
				group(i, j) = dataSet[i][j];
		}
	}
	
	//k-�����㷨��ʵ�ֶ��������ݵķ���
	T classify(vector<float> inX, int k){
		int dataSetSize = group.rows(); //���ݼ�������
		//����������inX�ظ�dataSetSize
		MatrixXf diffMat;
		diffMat.resize(group.rows(), group.cols());
		for (int i = 0; i < dataSetSize; i++)
			for (int j = 0; j < inX.size(); j++)
				diffMat(i, j) = inX[j];
		
		//����������ݺ����ݼ���Ԫ��֮��Ĳ�ֵ
		diffMat = diffMat - group;
		cout << diffMat << endl;
		//��diffMat�����е�ÿһ��Ԫ��ƽ��
		diffMat = diffMat.array().abs2();
		cout << diffMat << endl;
		//��diffMat��������������
		MatrixXf sqDiffMat = diffMat.rowwise().sum();
		cout << sqDiffMat << endl;
		//��sqDiffMat�����е�ÿһ��Ԫ�ؿ�ƽ��
		sqDiffMat = sqDiffMat.array().sqrt();
		cout << sqDiffMat << endl;
		//��sqDiffMat�����е�Ԫ�ش�С������������ҳ����е�ǰk������������ǰk��Ԫ�ص�����
		vector<int> sortedIndex = argsort(sqDiffMat,k);
		for (int i = 0; i < k; i++)
			cout << sortedIndex[i] << " ";
		cout << endl;
		//��������Ӧ��k����ǩ���м��㣬ѡ���ǩ���ִ������ķ���
		return sortLabels(sortedIndex);
	}

private:
	//�Ծ�����д�С����������ҳ�ǰk��Ԫ�أ����������ǵ�����
	//��������multimap����
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

	//���س��ִ������ı�ǩ
	T sortLabels(vector<int> sortedIndex){
		//��label��ǩ���뵽map��:�����ֵkey���ڣ���Ӧ��value��1�������ֵ�����ڣ���Ӧ��valueֵ��1
		map<T, int> mp;
		map<T, int>::iterator itr;
		for (int i = 0; i < sortedIndex.size(); i++){
			itr = mp.find(label[sortedIndex[i]]);
			//�����ֵkey���ڣ���Ӧ��value��1
			if (itr != mp.end()){
				itr->second += 1;
			}
			//�����ֵ�����ڣ���Ӧ��valueֵ��1
			else{
				mp.insert(pair<T, int>(label[sortedIndex[i]], 1));
			}
		}

		//����һ��map��������������valueֵ
		map<T, int>::iterator it; //���ڼ�¼���value�ĵ�����
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