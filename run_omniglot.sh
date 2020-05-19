#!/bin/bash

python3.7 init.py
cd ./data
git clone https://github.com/brendenlake/omniglot.git
mv omniglot Omniglot_Raw
cd ..
python3.7 prepare_omniglot.py