//writen by WangJin  2018/1/8 
//逐行读取文件中的数据并进行保存

#include <vector>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

class ReadFile{
private:
	vector<vector<float>> dataSet;
	vector<float> labels;

public:
	//构造函数
	ReadFile(string fileName){
		string line;
		//每次读入一整行，直到到达文件末尾
		ifstream file(fileName);
		int dataSetSize = 0;
		while (getline(file, line)){
			dataSet.push_back(vector<float>());
			vector<string> str = splitString(line, "\t");
			int strSize = str.size();
			for (int i = 0; i < strSize - 1; i++)
				dataSet[dataSetSize].push_back(stof(str[i]));
			labels.push_back(stof(str[strSize - 1]));
			dataSetSize++;
		}
	}

	//获取数据集
	vector<vector<float>> getDataSet(){
		return dataSet;
	}

	//获取数据集的标签
	vector<float> getLabels(){
		return labels;
	}

	//输出数据集的标签
	void showLabels(){
		for (int i = 0; i < labels.size(); i++)
			cout << labels[i] << " ";
		cout << endl;
		cout << dataSet.size() << " " << dataSet[0].size() << " " << labels.size() << endl;
	}

private:
	//分割字符串
	vector<string> splitString(const string& src, string separate_character)
	{
		vector<string> strs;

		int separate_characterLen = separate_character.size(); //分割字符串的长度,这样就可以支持如“,,”多字符串的分隔符
		int lastPosition = 0, index = -1;
		while (-1 != (index = src.find(separate_character, lastPosition)))
		{
			strs.push_back(src.substr(lastPosition, index - lastPosition));
			lastPosition = index + separate_characterLen;
		}
		string lastString = src.substr(lastPosition); //截取最后一个分隔符后的内容
		if (!lastString.empty())
			strs.push_back(lastString); //如果最后一个分隔符后还有内容就入队
		return strs;
	}
};