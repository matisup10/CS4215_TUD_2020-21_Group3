#!/bin/bash -i

source ~/.bashrc
declare -a ModelTypes=("cnn")
pwd ~

# Iterate the string array using for loop
for epochs in `seq 4 1 4`; do
       for iteration in `seq 0 1 5`; do
		 for max_words in `seq 5400 200 5500`; do
                        for val in ${ModelTypes[@]}; do
				for min_words in `seq 0 10 50`; do
                           		echo "Starting file: epochs:$epoch model_type:$val (iterator:$iteration)"
                           		/home/woutermorssink/analytics-zoo/scripts/spark-submit-python-with-zoo.sh --master local[*] sentiment.py --model_type $val --epochs $epochs --iteration_number $iteration --n_train 3800 --n_test 4000 --max_words $max_words --min_words $min_words
                        	done
			done
                done
        done
done

