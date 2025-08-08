# seed 912 (default), manual rename in graph_eval.py to seeds [913, 914, 915, 916]
python3 graph_eval.py --exp_method GCAM  --dataset mutag --model GIN --accuracy --faithfulness
python3 graph_eval.py --exp_method RAND  --dataset mutag --model GIN --accuracy --faithfulness