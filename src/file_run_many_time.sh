for num_depth in 1024; do
    for step in 0.0001 0.001 0.1 0.9; do
        for alpha in 0.1 1.0 2.0 4.0; do
            for k in 1.0 2.0 4.0 6.0; do
            	#python3 run_grand_ex.py --depth $num_depth --discritize_type norm --step_size $step --trunc_alpha $alpha --dataset Cora
            	python3 run_grand_ex.py --depth $num_depth --discritize_type norm --step_size $step --trunc_alpha $alpha --k $k --dataset Citeseer
            done
	done
    done
done
