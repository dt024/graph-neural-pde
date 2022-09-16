for time in  8 6 4 1 10 12 14 16; do
    for coup in 0.5 0.7 0.9 1; do
        for run in 1 2 3 4 5 6 7 8 9 10; do
            	#python3 run_grand_ex.py --depth $num_depth --discritize_type norm --step_size $step --trunc_alpha $alpha --dataset Cora
            python3 run_GNN.py --time $time --coupling_strength $coup --method adaptive_heun --function transformer --dataset Cora --no_early
	done
    done
done
