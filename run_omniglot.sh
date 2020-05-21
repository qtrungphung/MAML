#!/bin/bash

python3 init.py
cd ./data
git clone https://github.com/brendenlake/omniglot.git
mv omniglot Omniglot_Raw
cd ..
python3 prepare_omniglot.py