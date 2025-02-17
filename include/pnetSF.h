// #include <map>
// #include <string>
// #include <vector>
// #include <tuple>
// #include <iostream>
// taken from https://github.com/LLRCMS/KLUBAnalysis/blob/10821fdc58f6bdb68b0142740f05b8c9f6e23bf9/src/pnetSF.cc#L5
// 0 = signal, 1 = ttbar , 2 = dy , 3 = other
std::map<std::string, std::vector<float>> getSFmap(float pT_, const std::string& period_, const int whichType) {
std::map<std::string, std::vector<float>> scaleFactors ;
    if (whichType == 0){
        if(period_=="Run2_2016_HIPM"){
            if (pT_ < 500) {
            scaleFactors["Tight"] = {1.054, 0.080, -0.077};
            scaleFactors["Medium"] = {1.052, 0.087, -0.081};
            scaleFactors["Loose"] = {1.032, 0.096, -0.090};
            } else if (pT_ >= 500 && pT_ < 600) {
            scaleFactors["Tight"] = {1.139, 0.083, -0.081};
            scaleFactors["Medium"] = {1.068, 0.078, -0.073};
            scaleFactors["Loose"] = {1.062, 0.092, -0.082};
            } else if (pT_ >= 600) {
            scaleFactors["Tight"] = {1.049, 0.133, -0.130};
            scaleFactors["Medium"] = {0.996, 0.101, -0.097};
            scaleFactors["Loose"] = {1.002, 0.106, -0.101};
            }
        }
        else if (period_=="Run2_2016") {
            if (pT_ < 500) {
            scaleFactors["Tight"] = {1.031, 0.050, -0.046};
            scaleFactors["Medium"] = {1.029, 0.051, -0.045};
            scaleFactors["Loose"] = {1.031, 0.058, -0.050};
            } else if (pT_ >= 500 && pT_ < 600) {
            scaleFactors["Tight"] = {1.055, 0.069, -0.067};
            scaleFactors["Medium"] = {1.070, 0.066, -0.062};
            scaleFactors["Loose"] = {1.089, 0.076, -0.068};
            } else if (pT_ >= 600) {
            scaleFactors["Tight"] = {1.088, 0.076, -0.072};
            scaleFactors["Medium"] = {1.077, 0.067, -0.059};
            scaleFactors["Loose"] = {1.057, 0.077, -0.056};
            }
        }
        else if (period_=="Run2_2017"){
            if (pT_ < 500) {
            scaleFactors["Tight"] = {1.055, 0.057, -0.054};
            scaleFactors["Medium"] = {1.006, 0.052, -0.052};
            scaleFactors["Loose"] = {0.966, 0.055, -0.057};
            } else if (pT_ >= 500 && pT_ < 600) {
            scaleFactors["Tight"] = {1.067 , 0.057, -0.055};
            scaleFactors["Medium"] = {1.051 , 0.056, -0.055};
            scaleFactors["Loose"] = {1.021 , 0.053, -0.052};
            } else if (pT_ >= 600) {
            scaleFactors["Tight"] = {1.045 , 0.045, -0.046};
            scaleFactors["Medium"] = {0.991 , 0.038, -0.043};
            scaleFactors["Loose"] = {0.979 , 0.035, -0.038};
            }
        }
        else if (period_=="Run2_2018"){
            if (pT_ < 500) {
            scaleFactors["Tight"] = {0.994 , 0.064, -0.064};
            scaleFactors["Medium"] = {0.966 , 0.056, -0.057};
            scaleFactors["Loose"] = {0.921 , 0.071, -0.077};
            } else if (pT_ >= 500 && pT_ < 600) {
            scaleFactors["Tight"] = {1.072 , 0.041, -0.036};
            scaleFactors["Medium"] = {1.033 , 0.030, -0.025};
            scaleFactors["Loose"] = {1.006 , 0.024, -0.026};
            } else if (pT_ >= 600) {
            scaleFactors["Tight"] = {1.046 , 0.038, -0.038};
            scaleFactors["Medium"] = {1.010 , 0.030, -0.035};
            scaleFactors["Loose"] = {1.001 , 0.035, -0.037};
            }
        }
    }

    else if (whichType == 1){
        // HP and MP SFs- defaultto 1
        // TTLikeSFs
        if(period_=="Run2_2016_HIPM"){
            if (pT_ < 300) {
                scaleFactors["Tight"] = {1.,0.,0.} ;
                scaleFactors["Medium"] = {1.,0.,0.} ;
                scaleFactors["Loose"] = {0.4927,   0.3802, -0.3802};
                // scaleFactors["Loose"] = {0.7092995,   0.24862634, -0.24862634};

            }
            else if (pT_ >= 300 && pT_< 400 ) {
                scaleFactors["Tight"] = {1.,0.,0.} ;
                scaleFactors["Medium"] = {1.,0.,0.} ;
                scaleFactors["Loose"] = {1.0216, 0.6653, -0.6653};
                // scaleFactors["Loose"] = {0.85037322,   0.34022010, -0.34022010};
            }
            else if (pT_ >= 400  && pT_< 700  ) {
                scaleFactors["Tight"] = {1.,0.,0.} ;
                scaleFactors["Medium"] = {1.,0.,0.} ;
                scaleFactors["Loose"] = {0.3795, 1.5452,-1.5452 };
                // scaleFactors["Loose"] = {0.66767709,   0.36953115, -0.36953115};

            }
            else{
                scaleFactors["Tight"] = {1.,0.,0.} ;
                scaleFactors["Medium"] = {1.,0.,0.} ;
                scaleFactors["Loose"] = {1.,0.,0.} ;
            }
        }
        else if(period_=="Run2_2016"){
            if (pT_ < 300) {
                scaleFactors["Tight"] = {1.,0.,0.} ;
                scaleFactors["Medium"] = {1.,0.,0.} ;
                scaleFactors["Loose"] = {0.8377,0.5155,-0.5155};
                // scaleFactors["Loose"] = {0.8279069,   0.28734629, -0.28734629};
            }
            else if (pT_ >= 300 && pT_< 400 ) {
                scaleFactors["Tight"] = {1.,0.,0.} ;
                scaleFactors["Medium"] = {1.,0.,0.} ;
                scaleFactors["Loose"] = {0.571,0.7731,-0.7731};
                // scaleFactors["Loose"] = {0.84752986,   0.39659080, -0.39659080};
            }
            if (pT_ >= 400 && pT_< 700 ) {
                scaleFactors["Tight"] = {1.,0.,0.} ;
                scaleFactors["Medium"] = {1.,0.,0.} ;
                scaleFactors["Loose"] = {1.2948,3.1934,-3.1934};
                // scaleFactors["Loose"] = {0.95545670,   0.64319324, -0.64319324};

            }
            else{
                scaleFactors["Tight"] = {1.,0.,0.} ;
                scaleFactors["Medium"] = {1.,0.,0.} ;
                scaleFactors["Loose"] = {1.,0.,0.} ;
            }
        }
        else if(period_=="Run2_2017"){
            if (pT_ < 300) {
                scaleFactors["Tight"] = {1.,0.,0.} ;
                scaleFactors["Medium"] = {1.,0.,0.} ;
                scaleFactors["Loose"] = {1.6759,0.4351,-0.4351};
                // scaleFactors["Loose"] = {1.1069126,   0.24672529, -0.24672529};
            }
            else if (pT_ >= 300 && pT_< 400 ) {
                scaleFactors["Tight"] = {1.,0.,0.} ;
                scaleFactors["Medium"] = {1.,0.,0.} ;
                scaleFactors["Loose"] = {0.5925,0.3974,-0.3974};
                // scaleFactors["Loose"] = {0.66183375,   0.22188845, -0.22188845};
            }
            if (pT_ >= 400 && pT_ < 700 ) {
                scaleFactors["Tight"] = {1.,0.,0.} ;
                scaleFactors["Medium"] = {1.,0.,0.} ;
                scaleFactors["Loose"] = {0.8383,1.1402,-1.1402};
                // scaleFactors["Loose"] = {0.21924106,   0.13346345, -0.13346345};
            }
            else{
                scaleFactors["Tight"] = {1.,0.,0.} ;
                scaleFactors["Medium"] = {1.,0.,0.} ;
                scaleFactors["Loose"] = {1.,0.,0.} ;
            }
        }
        else if(period_=="Run2_2018"){
            if (pT_ < 300) {
                scaleFactors["Tight"] = {1.,0.,0.} ;
                scaleFactors["Medium"] = {1.,0.,0.} ;
                scaleFactors["Loose"] = {0.8782,0.3081,-0.3081};
                // scaleFactors["Loose"] = {0.9755087,   0.20477083, -0.20477083};
            }
            else if (pT_ >= 300 && pT_< 400 ) {
                scaleFactors["Tight"] = {1.,0.,0.} ;
                scaleFactors["Medium"] = {1.,0.,0.} ;
                scaleFactors["Loose"] = {0.6941,0.3251,-0.3251};
                // scaleFactors["Loose"] = {0.80578453,   0.20842184, -0.20842184};
            }
            if (pT_ >= 400 && pT_<700 ) {
                scaleFactors["Tight"] = {1.,0.,0.} ;
                scaleFactors["Medium"] = {1.,0.,0.} ;
                scaleFactors["Loose"] = {0.8692, 0.6044, -0.6044};
                // scaleFactors["Loose"] = {0.73371453,   0.27071888, -0.27071888};

            }
            else{
                scaleFactors["Tight"] = {1.,0.,0.} ;
                scaleFactors["Medium"] = {1.,0.,0.} ;
                scaleFactors["Loose"] = {1.,0.,0.} ;
            }
        }
    }
    else if (whichType == 2){
        // HP and MP SFs- defaultto 1
        // DYLikeSFs
        if (period_=="Run2_2016_HIPM"){
            if (pT_ < 300){
                scaleFactors["Tight"] = {1.,0.,0.};
                scaleFactors["Medium"] = {1.,0.,0.};
                 scaleFactors["Loose"] =  {1.0743, 0.2764, -0.2764};
                // scaleFactors["Loose"] = {0.99217367, 0.1945583, -0.1945583};
            } else if (pT_ >= 300 && pT_ < 400) {
                scaleFactors["Tight"] = {1.,0.,0.};
                scaleFactors["Medium"] = {1.,0.,0.};
                scaleFactors["Loose"] = {1.2709,0.3487,-0.3487};
                // scaleFactors["Loose"] = {1.34811143, 0.31670744, -0.31670744};
            }
            else if (pT_ >= 400 && pT_ < 700){
                scaleFactors["Tight"] = {1.,0.,0.};
                scaleFactors["Medium"] = {1.,0.,0.};
                scaleFactors["Loose"] = {1.5753,0.6695,-0.6695};
                // scaleFactors["Loose"] = {1.35099449, 0.42826490, -0.42826490};
            }else{
                scaleFactors["Tight"] = {1.,0.,0.} ;
                scaleFactors["Medium"] = {1.,0.,0.} ;
                scaleFactors["Loose"] = {1.,0.,0.} ;
            }
        }
        else if (period_=="Run2_2016"){
            if (pT_ < 300) {
                scaleFactors["Tight"] = {1.,0.,0.};
                scaleFactors["Medium"] = {1.,0.,0.};
                scaleFactors["Loose"] = {1.6456,0.4316,-0.4316};
                // scaleFactors["Loose"] = {1.46031728, 0.2959850, -0.2959850};
            }
            else if (pT_ >= 300 && pT_< 400 ) {
                scaleFactors["Tight"] = {1.,0.,0.};
                scaleFactors["Medium"] = {1.,0.,0.};
                scaleFactors["Loose"] = {0.8931,0.4125,-0.4125};
                // scaleFactors["Loose"] = {1.10929824, 0.31057888, -0.31057888};
            }
            else if (pT_ >= 400 && pT_ < 700){
                scaleFactors["Tight"] = {1.,0.,0.};
                scaleFactors["Medium"] = {1.,0.,0.};
                scaleFactors["Loose"] = {0.6539,0.763,-0.763};
                // scaleFactors["Loose"] = {0.65913237, 0.27456928, -0.27456928};
            }else{
                scaleFactors["Tight"] = {1.,0.,0.} ;
                scaleFactors["Medium"] = {1.,0.,0.} ;
                scaleFactors["Loose"] = {1.,0.,0.} ;
            }
        }
        else if (period_=="Run2_2017"){
            if (pT_ < 300)
            {
                scaleFactors["Tight"] = {1.,0.,0.};
                scaleFactors["Medium"] = {1.,0.,0.};
                scaleFactors["Loose"] = {1.0085,0.1877,-0.1877};
                // scaleFactors["Loose"] = {1.00916880, 0.1431627, -0.1431627}; // LP down
            }
            else if (pT_ >= 300 && pT_< 400 ) {
                scaleFactors["Tight"] = {1.,0.,0.};
                scaleFactors["Medium"] = {1.,0.,0.};
                scaleFactors["Loose"] = {1.4333,0.3412,-0.3412};
                // scaleFactors["Loose"] = {1.38009424, 0.24822319, -0.24822319};
            }
            else if (pT_ >= 400 && pT_ < 700){
                scaleFactors["Tight"] = {1.,0.,0.};
                scaleFactors["Medium"] = {1.,0.,0.};
                scaleFactors["Loose"] = {1.4411,0.5479,-0.5479};
                // scaleFactors["Loose"] = {1.08742263, 0.27418802, -0.27418802};
            }else{
                scaleFactors["Tight"] = {1.,0.,0.} ;
                scaleFactors["Medium"] = {1.,0.,0.} ;
                scaleFactors["Loose"] = {1.,0.,0.} ;
            }
        }
        else if (period_=="Run2_2018"){
            if (pT_ < 300) {
            scaleFactors["Tight"] = {1.,0.,0.};
            scaleFactors["Medium"] = {1.,0.,0.};
            scaleFactors["Loose"] = {1.1462,0.2,-0.2};
            // scaleFactors["Loose"] = {1.178, 0.144, -0.144}; // LP down
            } else if (pT_ >= 300 && pT_ < 400) {
            scaleFactors["Tight"] = {1.,0.,0.};
            scaleFactors["Medium"] = {1.,0.,0.};
            scaleFactors["Loose"] = {1.2728,0.2966,-0.2966};
            // scaleFactors["Loose"] = {1.09308159, 0.18257319, -0.18257319};
            } else if (pT_ >= 400 && pT_ < 700) {
            scaleFactors["Tight"] = {1.,0.,0.};
            scaleFactors["Medium"] = {1.,0.,0.};
            scaleFactors["Loose"] = {0.952,0.3111,-0.3111};
            // scaleFactors["Loose"] = {1.24201816, 0.27224163, -0.27224163};
            }else{
                scaleFactors["Tight"] = {1.,0.,0.} ;
                scaleFactors["Medium"] = {1.,0.,0.} ;
                scaleFactors["Loose"] = {1.,0.,0.} ;
            }

        }
    }
    else if (whichType == 3){
        // other samples -> SFto 1 (?)
        scaleFactors["Tight"] = {1.,0.,0.} ;
        scaleFactors["Medium"] = {1.,0.,0.} ;
        scaleFactors["Loose"] = {1.,0.,0.} ;
    }
    return scaleFactors;
}

float getSFPNet(float pT_, const std::string& period_,  const std::string& scale,  const std::string& WP, const int whichType){

    std::vector<float> SF = getSFmap(pT_, period_,whichType)[WP];

    if (scale == "Up"){
        return SF[0]+SF[1];
    }
    if (scale == "Down"){
        return SF[0]+SF[2];
    }
    return SF[0];
}
