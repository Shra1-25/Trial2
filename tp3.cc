#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/default_device.h"
using namespace tensorflow;
using namespace std;

int main(int argc, char* argv[]) {
    std::cout<<"Hello"<<endl;
    
    std::ifstream  data("/home/cmsusr/CMSSW_11_0_1/src/Trial2/X_data.csv");
    std::string line;
    std::vector<std::vector<float> > X_vec;
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

        X_vec.push_back(parsedRow);
        cout<<++i<<" "<<parsedRow[0]<<" to "<<parsedRow[31]<<endl;
    }
    
    
    std::string graph_definition = "/home/cmsusr/CMSSW_11_0_1/src/Trial2/graph3.pb";
    Session* session;
    GraphDef graph_def;
    SessionOptions opts;
    std::vector<Tensor> outputs; // Store outputs
    TF_CHECK_OK(ReadBinaryProto(Env::Default(), graph_definition, &graph_def));

    // Set GPU options
    //graph::SetDefaultDevice("/gpu:0", &graph_def);
    //opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
    //opts.config.mutable_gpu_options()->set_allow_growth(true);

    // create a new session
    TF_CHECK_OK(NewSession(opts, &session));

    // Load graph into session
    TF_CHECK_OK(session->Create(graph_def));

    // Initialize our variables
    TF_CHECK_OK(session->Run({}, {}, {"init_all_vars_op"}, nullptr));

    Tensor x(DT_FLOAT, TensorShape({100, 32}));
    Tensor y(DT_FLOAT, TensorShape({100, 8}));
    auto _XTensor = x.matrix<float>();
    auto _YTensor = y.matrix<float>();
    std::copy_n(X_vec.begin(), X_vec.size(), _XTensor.flat<float>().data())
    //_XTensor.setRandom();
    _YTensor.setRandom();

    for (int i = 0; i < 10; ++i) {
        
        TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {"cost"}, {}, &outputs)); // Get cost
        float cost = outputs[0].scalar<float>()(0);
        std::cout << "Cost: " <<  cost << std::endl;
        TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {}, {"train"}, nullptr)); // Train
        outputs.clear();
    }

    session->Close();
    delete session;
    std::cout<<"All done"<<endl;
    return 0;
}