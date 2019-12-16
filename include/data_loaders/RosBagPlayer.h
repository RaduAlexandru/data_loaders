#pragma once

//c++
#include <iosfwd>
#include <sstream>
#include <memory>
#include <stdarg.h>

//My stuff
#include "tiny-process-library/process.hpp"


class RosBagPlayer : public std::enable_shared_from_this<RosBagPlayer>{
public:
    template <class ...Args>
    static std::shared_ptr<RosBagPlayer> create( Args&& ...args ){
        return std::shared_ptr<RosBagPlayer>( new RosBagPlayer(std::forward<Args>(args)...) );
    }
    ~RosBagPlayer();
    void play(std::string args);
    void pause();
    void play();
    void reset();
    bool is_paused();
    bool is_finished(); //when the bag finishes running this will return true
    void kill();

    bool m_player_should_do_one_step;
    bool m_player_should_continue_after_step;
    int m_nr_resets; //nr of resets that were performed. Sometimes it's nice to know how many were done by the user so that I can have code that does different things after each reset

private:
    RosBagPlayer(const std::string config_file);
    void init_params(const std::string config_file);

    std::shared_ptr<TinyProcessLib::Process> m_rosbag; //make it a shared ptr so we can later use make_shared on the whole RosBagPlayer class
    std::string m_cmd;
    bool m_paused;


};