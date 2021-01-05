/***
Copyright (c) UChicago Argonne, LLC. All rights reserved.
See LICENSE file.
***/
// Created by Barbara Frosik

#ifndef state_hpp
#define state_hpp

#include "vector"
#include "map"
#include "common.h"

class Params;
class Reconstruction;

namespace af {
    class array;
}

// This class maintain the state of the reconstruction process.
class State
{
private:
    // a reference to params object
    Params *params;
    
    // current iteration
    int current_iter;
    // number of configured iterations for reconstruction
    int total_iter_num;
    
    // The vector of errors indexed by iteration
    std::vector<d_type>  errors;
    
    // current algorithm id
    int current_alg;
    // current index of index switches vector
    int alg_switch_index;

public:
    // Constructor. Takes pointer to the Param object. Uses the Param object to set the initial values.
    State(Params *params);
    
    // Needs destructor to free allocated memory.
    ~State();
    
    // This initializes the object. It does the following:
    // - sets current algorithm
    // - sets the total number of iterations
    void Init();
    
    // This method determines the attributes of the current state (i.e. iteration).
    // It returns false if the program reached the end of iterations, and true otherwise.
    // It finds which algorithm will be run in this state .
    // It sets the algorithm to run convolution based on state. (convolution is run at the algorithm switch)
    // It calculates the averaging iteration (current iteration - start of averaging)
    // It updates support at the correct iterations.
    int Next();
        
    // Returns current iteration number
    int GetCurrentIteration();

    // Returns an algorithm that should be run in a current state (i.e. iteration).
//    Algorithm * GetCurrentAlg();
    int GetCurrentAlg();

    // Stores the error
    void RecordError(d_type error);
    
    // Returns vector containing errors
    std::vector<d_type> GetErrors();
};


#endif /* state_hpp */
