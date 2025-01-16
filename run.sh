#!/bin/bash

# fairQCM
# python dqn-donut.py -ep 1000 -nexp 10 -sm binary -cf True --reward_type nsw
# python dqn-donut.py -ep 1000 -nexp 10 -sm binary -cf True --reward_type utilitarian
# python dqn-donut.py -ep 1000 -nexp 10 -sm binary -cf True --reward_type rawlsian
# python dqn-donut.py -ep 1000 -nexp 10 -sm binary -cf True --reward_type egalitarian
# python dqn-donut.py -ep 1000 -nexp 10 -sm binary -cf True --reward_type gini

# # full
# python dqn-donut.py -ep 1000 -nexp 10 -sm binary --reward_type nsw
# python dqn-donut.py -ep 1000 -nexp 10 -sm binary --reward_type utilitarian
# python dqn-donut.py -ep 1000 -nexp 10 -sm binary --reward_type rawlsian
# python dqn-donut.py -ep 1000 -nexp 10 -sm binary --reward_type egalitarian
# python dqn-donut.py -ep 1000 -nexp 10 -sm binary --reward_type gini

# min
# python dqn-donut.py -ep 1000 -nexp 10 -sm reset-binary --reward_type nsw 
# python dqn-donut.py -ep 1000 -nexp 10 -sm reset-binary --reward_type utilitarian
# python dqn-donut.py -ep 1000 -nexp 10 -sm reset-binary --reward_type rawlsian
# python dqn-donut.py -ep 1000 -nexp 10 -sm reset-binary --reward_type egalitarian
# python dqn-donut.py -ep 1000 -nexp 10 -sm reset-binary --reward_type gini

# # reset
# python dqn-donut.py -ep 1000 -nexp 10 -sm equal-binary --reward_type nsw
# python dqn-donut.py -ep 1000 -nexp 10 -sm equal-binary --reward_type utilitarian
# python dqn-donut.py -ep 1000 -nexp 10 -sm equal-binary --reward_type rawlsian
# python dqn-donut.py -ep 1000 -nexp 10 -sm equal-binary --reward_type egalitarian
# python dqn-donut.py -ep 1000 -nexp 10 -sm equal-binary --reward_type gini

# rnn
# python dqn-recurrent.py -ep 1000 -nexp 10 -sm rnn --reward_type gini
# python dqn-recurrent.py -ep 1000 -nexp 10 -sm rnn --reward_type nsw
# python dqn-recurrent.py -ep 1000 -nexp 10 -sm rnn --reward_type egalitarian
# python dqn-recurrent.py -ep 1000 -nexp 10 -sm rnn --reward_type rawlsian
# python dqn-recurrent.py -ep 1000 -nexp 10 -sm rnn --reward_type utilitarian

python main.py -ep 1000 -nexp 10 -env donut -net rnn -rt gini