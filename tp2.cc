#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include<string>
using namespace std;

int main(){

std::ifstream  data("\\home\\cmsusr\\CMSSW_11_0_1\\src\\Trial2\\X_data.csv");
    std::string line;
    std::vector<std::vector<float> > parsedCsv;
    int i=0;
    while(std::getline(data,line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<float> parsedRow;
        while(std::getline(lineStream,cell,','))
        {
            parsedRow.push_back(std::stof(cell));
        }

        parsedCsv.push_back(parsedRow);
        cout<<++i<<" "<<parsedRow[0]<<" to "<<parsedRow[31]<<endl;
    }
}
