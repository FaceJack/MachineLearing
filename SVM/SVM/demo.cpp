#include "svm.h"
#include "readFile.h"

int main(int argc,char* argv[]){
	ReadFile file = ReadFile(argv[1]);
	SVM svm = SVM(file.getDataSet(), file.getLabels(), 0.6, 0.001);
	pair<string, int> ktup = make_pair("line", 0);
	svm.smoP(40, ktup);
	svm.showResult();

	return 0;
}