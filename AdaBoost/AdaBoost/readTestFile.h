//writen by WangJin  2018/1/8 
//���ж�ȡ�ļ��е����ݲ����б���

#include <vector>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

class ReadTestFile{
private:
	vector<vector<float>> dataSet;

public:
	//���캯��
	ReadTestFile(string fileName){
		string line;
		//ÿ�ζ���һ���У�ֱ�������ļ�ĩβ
		ifstream file(fileName);
		int dataSetSize = 0;
		while (getline(file, line)){
			dataSet.push_back(vector<float>());
			vector<string> str = splitString(line, "\t");
			int strSize = str.size();
			for (int i = 0; i < strSize; i++)
				dataSet[dataSetSize].push_back(stof(str[i]));
			dataSetSize++;
		}
	}

	//��ȡ���ݼ�
	vector<vector<float>> getDataSet(){
		return dataSet;
	}

private:
	//�ָ��ַ���
	vector<string> splitString(const string& src, string separate_character)
	{
		vector<string> strs;

		int separate_characterLen = separate_character.size(); //�ָ��ַ����ĳ���,�����Ϳ���֧���硰,,�����ַ����ķָ���
		int lastPosition = 0, index = -1;
		while (-1 != (index = src.find(separate_character, lastPosition)))
		{
			strs.push_back(src.substr(lastPosition, index - lastPosition));
			lastPosition = index + separate_characterLen;
		}
		string lastString = src.substr(lastPosition); //��ȡ���һ���ָ����������
		if (!lastString.empty())
			strs.push_back(lastString); //������һ���ָ����������ݾ����
		return strs;
	}
};