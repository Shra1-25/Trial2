#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/default_device.h"
using namespace tensorflow;

int main(int argc, char* argv[]) {

    std::cout<<"Hello"<<endl;
	std::string graph_definition = "/home/cmsusr/CMSSW_11_0_1/src/Trial2/graph3.pb";
    Session* session;
    GraphDef graph_def;
    SessionOptions opts;
    std::vector<Tensor> outputs; // Store outputs
    TF_CHECK_OK(ReadBinaryProto(Env::Default(), graph_definition, &graph_def));
	std::cout<<"Done1"<<endl;
    // Set GPU options
    //graph::SetDefaultDevice("/gpu:0", &graph_def);
    //opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
    //opts.config.mutable_gpu_options()->set_allow_growth(true);
	std::cout<<"Done2"<<endl;
    // create a new session
    TF_CHECK_OK(NewSession(opts, &session));
	std::cout<<"Done3"<<endl;
    // Load graph into session
    TF_CHECK_OK(session->Create(graph_def));
	std::cout<<"Done4"<<endl;
    // Initialize our variables
    TF_CHECK_OK(session->Run({}, {}, {"init_all_vars_op"}, nullptr));

    Tensor x(DT_FLOAT, TensorShape({100, 32}));
    Tensor y(DT_FLOAT, TensorShape({100, 8}));
    auto _XTensor = x.matrix<float>();
    auto _YTensor = y.matrix<float>();
	std::cout<<"Done";
    _XTensor.setRandom();
    _YTensor.setRandom();
	std::cout<<"Done5"<<endl;
    for (int i = 0; i < 10; ++i) {
        
        TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {"cost"}, {}, &outputs)); // Get cost
        float cost = outputs[0].scalar<float>()(0);
        std::cout << "Cost: " <<  cost << std::endl;
        TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {}, {"train"}, nullptr)); // Train
        outputs.clear();
    }
	std::cout<<"Done6"<<endl;
    session->Close();
    delete session;
	std::cout<<"Done7"<<endl;
    return 0;
}