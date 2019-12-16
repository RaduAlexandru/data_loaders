//My stuff
#include "easy_pbr/Viewer.h"
#include "mbzirc_challenge_1/SyntheticGenerator.h"


int main(int argc, char *argv[]) {

    std::string config_file="arena.cfg";

    std::shared_ptr<Viewer> view = Viewer::create(config_file); 

    std::shared_ptr<SyntheticGenerator> synth = SyntheticGenerator::create(config_file, view);

    while (true) {
        view->update();
    }

    return 0;
}
